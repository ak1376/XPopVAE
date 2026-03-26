#!/usr/bin/env python3
# snakemake_scripts/diagnose_allelefreq_vs_ld.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------
# project paths & local imports
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvVAE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether a genotype VAE relies more on allele-frequency "
            "information or LD information."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to saved model checkpoint, e.g. best_model.pt",
    )
    parser.add_argument(
        "--train-genotype-npy",
        type=Path,
        required=True,
        help="Training genotype matrix .npy, shape (n_individuals, n_snps)",
    )
    parser.add_argument(
        "--eval-genotype-npy",
        type=Path,
        required=True,
        help="Evaluation genotype matrix .npy, shape (n_individuals, n_snps)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for outputs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for reconstruction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for SNP shuffling",
    )
    parser.add_argument(
        "--maf-bins",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="MAF bin edges. Default: 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5",
    )
    return parser


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> ConvVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vae_config = checkpoint["vae_config"]
    input_length = int(checkpoint["input_length"])

    pheno_hidden_dim = vae_config.get("phenotype", {}).get("pheno_hidden_dim", None)
    pheno_latent_dim=vae_config["phenotype"].get("pheno_latent_dim", None)

    model = ConvVAE(
        input_length=input_length,
        in_channels=1,
        hidden_channels=vae_config["model"]["hidden_channels"],
        kernel_size=int(vae_config["model"]["kernel_size"]),
        stride=int(vae_config["model"]["stride"]),
        padding=int(vae_config["model"]["padding"]),
        latent_dim=int(vae_config["model"]["latent_dim"]),
        use_batchnorm=bool(vae_config["model"].get("use_batchnorm", False)),
        activation=vae_config["model"].get("activation", "elu"),
        pheno_dim=1,
        pheno_hidden_dim=pheno_hidden_dim,
        pheno_latent_dim=pheno_latent_dim
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def reconstruct_argmax_genotypes(
    model: torch.nn.Module,
    G: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    X = torch.tensor(G, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    recon_batches = []

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            logits, _, _, _, _ = model(x)   # (B,3,L)
            pred = torch.argmax(logits, dim=1)  # (B,L)
            recon_batches.append(pred.cpu().numpy())

    return np.concatenate(recon_batches, axis=0)


def compute_global_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = recall_score(
        y_true.ravel(),
        y_pred.ravel(),
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    return float(np.mean(recalls))


def compute_per_snp_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    n_snps = y_true.shape[1]
    out = np.full(n_snps, np.nan, dtype=float)

    for j in range(n_snps):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        recalls = recall_score(
            yt,
            yp,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
        out[j] = float(np.mean(recalls))

    return out


def compute_maf(G: np.ndarray) -> np.ndarray:
    """
    Compute per-SNP minor allele frequency for diploid genotype matrix coded 0/1/2.
    Returns shape (n_snps,)
    """
    alt_freq = G.mean(axis=0) / 2.0
    maf = np.minimum(alt_freq, 1.0 - alt_freq)
    return maf


def make_frequency_baseline_from_train(train_G: np.ndarray, eval_G: np.ndarray) -> np.ndarray:
    """
    Predict each SNP in eval set using rounded mean genotype from train set.
    Returns shape (n_eval_individuals, n_snps)
    """
    mean_genotype = train_G.mean(axis=0)
    baseline_per_snp = np.rint(mean_genotype).astype(int)
    baseline_per_snp = np.clip(baseline_per_snp, 0, 2)

    pred = np.tile(baseline_per_snp[None, :], (eval_G.shape[0], 1))
    return pred


def summarize_accuracy_by_maf_bins(
    maf: np.ndarray,
    per_snp_acc: np.ndarray,
    bin_edges: np.ndarray,
) -> dict:
    centers = []
    means = []
    counts = []

    for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i == len(bin_edges) - 2:
            mask = (maf >= left) & (maf <= right)
        else:
            mask = (maf >= left) & (maf < right)

        vals = per_snp_acc[mask]
        vals = vals[~np.isnan(vals)]

        centers.append((left + right) / 2.0)
        counts.append(len(vals))
        means.append(np.nan if len(vals) == 0 else np.mean(vals))

    return {
        "bin_edges": bin_edges,
        "bin_centers": np.array(centers),
        "mean_acc": np.array(means),
        "counts": np.array(counts),
    }


def plot_maf_accuracy_curves(
    maf_summary_vae: dict,
    maf_summary_shuffle: dict,
    maf_summary_baseline: dict,
    output_path: Path,
    title: str = "Per-SNP balanced accuracy vs MAF",
) -> None:
    plt.figure(figsize=(7.5, 5.5))

    plt.plot(
        maf_summary_vae["bin_centers"],
        maf_summary_vae["mean_acc"],
        marker="o",
        label="VAE reconstruction",
    )
    plt.plot(
        maf_summary_shuffle["bin_centers"],
        maf_summary_shuffle["mean_acc"],
        marker="o",
        label="VAE shuffled input",
    )
    plt.plot(
        maf_summary_baseline["bin_centers"],
        maf_summary_baseline["mean_acc"],
        marker="o",
        label="Frequency baseline",
    )

    plt.xlabel("Minor allele frequency")
    plt.ylabel("Mean per-SNP balanced accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved MAF accuracy plot to: {output_path}")


def save_maf_summary_table(
    maf_summary_vae: dict,
    maf_summary_shuffle: dict,
    maf_summary_baseline: dict,
    output_path: Path,
) -> None:
    with open(output_path, "w") as f:
        f.write(
            "maf_left\tmaf_right\tbin_center\tn_snps\t"
            "vae_mean_bal_acc\tshuffled_mean_bal_acc\tbaseline_mean_bal_acc\n"
        )
        edges = maf_summary_vae["bin_edges"]
        for i in range(len(edges) - 1):
            f.write(
                f"{edges[i]:.5f}\t"
                f"{edges[i+1]:.5f}\t"
                f"{maf_summary_vae['bin_centers'][i]:.5f}\t"
                f"{maf_summary_vae['counts'][i]}\t"
                f"{maf_summary_vae['mean_acc'][i]:.6f}\t"
                f"{maf_summary_shuffle['mean_acc'][i]:.6f}\t"
                f"{maf_summary_baseline['mean_acc'][i]:.6f}\n"
            )
    print(f"Saved MAF summary table to: {output_path}")


def save_summary_text(
    output_path: Path,
    checkpoint: Path,
    train_path: Path,
    eval_path: Path,
    acc_vae: float,
    acc_shuffle: float,
    acc_baseline: float,
) -> None:
    with open(output_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Train genotype matrix: {train_path}\n")
        f.write(f"Eval genotype matrix: {eval_path}\n\n")

        f.write(f"Global balanced accuracy (VAE, original SNP order): {acc_vae:.6f}\n")
        f.write(f"Global balanced accuracy (VAE, shuffled SNP order): {acc_shuffle:.6f}\n")
        f.write(f"Global balanced accuracy (frequency baseline):      {acc_baseline:.6f}\n\n")

        f.write("Interpretation guide:\n")
        f.write("- If shuffled accuracy is close to original accuracy, the model relies more on allele-frequency-like information.\n")
        f.write("- If shuffled accuracy drops substantially, the model relies more on LD/local SNP order.\n")
        f.write("- If VAE barely beats the frequency baseline, much of its predictive power may come from allele-frequency information.\n")
        f.write("- If VAE clearly beats the baseline, it is using additional information beyond per-SNP mean genotype.\n")
    print(f"Saved text summary to: {output_path}")


def make_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    prefix: str,
    title_prefix: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = np.array([0, 1, 2])

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    recalls = recall_score(
        y_true_flat,
        y_pred_flat,
        labels=classes,
        average=None,
        zero_division=0,
    )
    bal_acc = float(np.mean(recalls))

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=classes)

    cm_row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        cm_row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=cm_row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted genotype",
        ylabel="True genotype",
        title=f"{title_prefix}\nBalanced Acc = {bal_acc:.3f}",
    )

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_normalized[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    np.save(output_dir / f"{prefix}_confusion_matrix_raw.npy", cm)
    np.save(output_dir / f"{prefix}_confusion_matrix_normalized.npy", cm_normalized)

    print(f"{title_prefix} balanced accuracy: {bal_acc:.6f}")
    for cls, rec in zip(classes, recalls):
        print(f"{title_prefix} recall for class {cls}: {rec:.6f}")

    return {
        "balanced_accuracy": bal_acc,
        "recalls": {int(cls): float(rec) for cls, rec in zip(classes, recalls)},
        "confusion_matrix_raw": cm,
        "confusion_matrix_normalized": cm_normalized,
    }


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device)

    print(f"Loading training genotype matrix: {args.train_genotype_npy}")
    G_train = np.load(args.train_genotype_npy)
    print(f"Training matrix shape: {G_train.shape}")

    print(f"Loading evaluation genotype matrix: {args.eval_genotype_npy}")
    G_eval = np.load(args.eval_genotype_npy)
    print(f"Evaluation matrix shape: {G_eval.shape}")

    if G_train.ndim != 2 or G_eval.ndim != 2:
        raise ValueError("Expected both genotype matrices to have shape (n_individuals, n_snps)")

    if G_train.shape[1] != G_eval.shape[1]:
        raise ValueError("Training and evaluation matrices must have the same number of SNPs")

    # ------------------------------------------------------------
    # 1. Original-order VAE reconstruction
    # ------------------------------------------------------------
    print("Reconstructing evaluation genotypes with original SNP order...")
    G_eval_vae = reconstruct_argmax_genotypes(
        model=model,
        G=G_eval,
        device=device,
        batch_size=args.batch_size,
    )
    acc_vae = compute_global_balanced_accuracy(G_eval, G_eval_vae)
    print(f"Global balanced accuracy (VAE, original order): {acc_vae:.6f}")

    np.save(args.output_dir / "reconstructed_eval_argmax.npy", G_eval_vae)
    print(f"Saved original-order reconstructed genotypes to: {args.output_dir / 'reconstructed_eval_argmax.npy'}")

    make_and_save_confusion_matrix(
        y_true=G_eval,
        y_pred=G_eval_vae,
        output_dir=args.output_dir,
        prefix="vae_original",
        title_prefix="Normalized Confusion Matrix: VAE original input",
    )

    # ------------------------------------------------------------
    # 2. SNP-order shuffle test
    # ------------------------------------------------------------
    print("Running SNP-order shuffle test...")
    perm = rng.permutation(G_eval.shape[1])

    G_eval_shuffled = G_eval[:, perm]
    G_eval_vae_shuffled_order = reconstruct_argmax_genotypes(
        model=model,
        G=G_eval_shuffled,
        device=device,
        batch_size=args.batch_size,
    )

    inv_perm = np.argsort(perm)
    G_eval_vae_shuffled = G_eval_vae_shuffled_order[:, inv_perm]

    acc_shuffle = compute_global_balanced_accuracy(G_eval, G_eval_vae_shuffled)
    print(f"Global balanced accuracy (VAE, shuffled SNP order): {acc_shuffle:.6f}")

    np.save(args.output_dir / "reconstructed_eval_argmax_shuffled_input.npy", G_eval_vae_shuffled)
    np.save(args.output_dir / "snp_permutation.npy", perm)
    print(f"Saved shuffled-order reconstructed genotypes to: {args.output_dir / 'reconstructed_eval_argmax_shuffled_input.npy'}")
    print(f"Saved SNP permutation to: {args.output_dir / 'snp_permutation.npy'}")

    make_and_save_confusion_matrix(
        y_true=G_eval,
        y_pred=G_eval_vae_shuffled,
        output_dir=args.output_dir,
        prefix="vae_shuffled_input",
        title_prefix="Normalized Confusion Matrix: VAE shuffled input",
    )

    # ------------------------------------------------------------
    # 3. Frequency baseline
    # ------------------------------------------------------------
    print("Computing frequency baseline from training set...")
    G_eval_baseline = make_frequency_baseline_from_train(G_train, G_eval)
    acc_baseline = compute_global_balanced_accuracy(G_eval, G_eval_baseline)
    print(f"Global balanced accuracy (frequency baseline): {acc_baseline:.6f}")

    np.save(args.output_dir / "reconstructed_eval_frequency_baseline.npy", G_eval_baseline)
    print(f"Saved frequency baseline predictions to: {args.output_dir / 'reconstructed_eval_frequency_baseline.npy'}")

    make_and_save_confusion_matrix(
        y_true=G_eval,
        y_pred=G_eval_baseline,
        output_dir=args.output_dir,
        prefix="frequency_baseline",
        title_prefix="Normalized Confusion Matrix: frequency baseline",
    )

    # ------------------------------------------------------------
    # 4. Per-SNP balanced accuracy vs MAF
    # ------------------------------------------------------------
    print("Computing per-SNP balanced accuracy vs MAF...")
    maf_eval = compute_maf(G_eval)

    per_snp_acc_vae = compute_per_snp_balanced_accuracy(G_eval, G_eval_vae)
    per_snp_acc_shuffle = compute_per_snp_balanced_accuracy(G_eval, G_eval_vae_shuffled)
    per_snp_acc_baseline = compute_per_snp_balanced_accuracy(G_eval, G_eval_baseline)

    np.save(args.output_dir / "maf_eval.npy", maf_eval)
    np.save(args.output_dir / "per_snp_bal_acc_vae.npy", per_snp_acc_vae)
    np.save(args.output_dir / "per_snp_bal_acc_shuffle.npy", per_snp_acc_shuffle)
    np.save(args.output_dir / "per_snp_bal_acc_baseline.npy", per_snp_acc_baseline)
    print(f"Saved per-SNP arrays to: {args.output_dir}")

    maf_bin_edges = np.array(args.maf_bins, dtype=float)
    if maf_bin_edges[0] != 0.0:
        raise ValueError("MAF bins should start at 0.0")
    if maf_bin_edges[-1] > 0.5:
        raise ValueError("MAF bins should not exceed 0.5")

    maf_summary_vae = summarize_accuracy_by_maf_bins(
        maf=maf_eval,
        per_snp_acc=per_snp_acc_vae,
        bin_edges=maf_bin_edges,
    )
    maf_summary_shuffle = summarize_accuracy_by_maf_bins(
        maf=maf_eval,
        per_snp_acc=per_snp_acc_shuffle,
        bin_edges=maf_bin_edges,
    )
    maf_summary_baseline = summarize_accuracy_by_maf_bins(
        maf=maf_eval,
        per_snp_acc=per_snp_acc_baseline,
        bin_edges=maf_bin_edges,
    )

    plot_maf_accuracy_curves(
        maf_summary_vae=maf_summary_vae,
        maf_summary_shuffle=maf_summary_shuffle,
        maf_summary_baseline=maf_summary_baseline,
        output_path=args.output_dir / "balanced_accuracy_vs_maf.png",
        title="Per-SNP balanced accuracy vs MAF",
    )

    save_maf_summary_table(
        maf_summary_vae=maf_summary_vae,
        maf_summary_shuffle=maf_summary_shuffle,
        maf_summary_baseline=maf_summary_baseline,
        output_path=args.output_dir / "maf_accuracy_summary.tsv",
    )

    # ------------------------------------------------------------
    # 5. Top-level summary
    # ------------------------------------------------------------
    save_summary_text(
        output_path=args.output_dir / "diagnostic_summary.txt",
        checkpoint=args.checkpoint,
        train_path=args.train_genotype_npy,
        eval_path=args.eval_genotype_npy,
        acc_vae=acc_vae,
        acc_shuffle=acc_shuffle,
        acc_baseline=acc_baseline,
    )

    np.savez(
        args.output_dir / "diagnostic_summary.npz",
        global_bal_acc_vae=acc_vae,
        global_bal_acc_shuffle=acc_shuffle,
        global_bal_acc_baseline=acc_baseline,
    )
    print(f"Saved compact numeric summary to: {args.output_dir / 'diagnostic_summary.npz'}")


if __name__ == "__main__":
    main()