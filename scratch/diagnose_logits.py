#!/usr/bin/env python3
# snakemake_scripts/diagnose_logits.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_model_from_checkpoint

'''
python snakemake_scripts/diagnose_logits.py \
    --checkpoint /sietch_colab/akapoor/XPopVAE/experiments/OOA/vae/vae_gamma__gamma0p0/vae_outputs/checkpoints/best_model.pt \
    --ceu-genotype-npy /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/genotype_matrices/discovery_train.npy \
    --yri-genotype-npy /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/genotype_matrices/target_held_out.npy \
    --output-dir /sietch_colab/akapoor/XPopVAE/experiments/OOA/vae/vae_gamma__gamma0p0/vae_outputs/diagnose_logits \
    --batch-size 128 \
    --freq-bins 0.0 0.05 0.15 0.35 0.5
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose VAE logit distributions stratified by allele frequency."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ceu-genotype-npy", type=Path, required=True)
    parser.add_argument("--yri-genotype-npy", type=Path, required=True)
    parser.add_argument("--output-dir",       type=Path, required=True)
    parser.add_argument("--batch-size",        type=int,  default=128)
    parser.add_argument(
        "--freq-bins",
        type=float, nargs="+",
        default=[0.0, 0.05, 0.15, 0.35, 0.5],
        help="MAF bin edges. Default: 0 0.05 0.15 0.35 0.5",
    )
    return parser


# ------------------------------------------------------------------
# logit extraction
# ------------------------------------------------------------------
def extract_logits(
    model: torch.nn.Module,
    G: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Returns logits of shape (n_individuals, 3, n_snps).
    Logit[:, k, j] = unnormalized score for genotype k at SNP j for individual i.
    """
    X = torch.tensor(G, dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

    all_logits = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            logits, _, _, _, _, _ = model(x)  # (B, 3, L)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)  # (N, 3, L)


# ------------------------------------------------------------------
# MAF binning
# ------------------------------------------------------------------
def compute_maf(G: np.ndarray) -> np.ndarray:
    alt_freq = G.sum(axis=0) / (2.0 * G.shape[0])
    return np.minimum(alt_freq, 1.0 - alt_freq)


def bin_sites_by_maf(
    maf: np.ndarray,
    bin_edges: np.ndarray,
) -> list[np.ndarray]:
    """Returns list of site index arrays, one per MAF bin."""
    bins = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (maf >= lo) & (maf < hi) if i < len(bin_edges) - 2 else (maf >= lo) & (maf <= hi)
        bins.append(np.where(mask)[0])
    return bins


# ------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------
def plot_logit_distributions(
    logits: np.ndarray,
    G_truth: np.ndarray,
    maf: np.ndarray,
    bin_edges: np.ndarray,
    site_bins: list[np.ndarray],
    pop_label: str,
    output_dir: Path,
) -> None:
    """
    For each MAF bin, plot the distribution of logits for each genotype class (0, 1, 2).
    One row per MAF bin, three columns (one per genotype class).
    """
    n_bins = len(site_bins)
    genotype_labels = ["Homozygous ref (0)", "Heterozygous (1)", "Homozygous alt (2)"]
    colors = ["steelblue", "darkorange", "forestgreen"]

    fig, axes = plt.subplots(n_bins, 3, figsize=(15, 3 * n_bins), sharex=False)
    if n_bins == 1:
        axes = axes[np.newaxis, :]

    for bin_idx, site_idx in enumerate(site_bins):
        lo, hi = bin_edges[bin_idx], bin_edges[bin_idx + 1]
        bin_label = f"MAF [{lo:.2f}, {hi:.2f})  n={len(site_idx)} sites"

        if len(site_idx) == 0:
            for k in range(3):
                axes[bin_idx, k].set_visible(False)
            continue

        # logits for this bin: shape (n_individuals, 3, n_sites_in_bin)
        logits_bin = logits[:, :, site_idx]  # (N, 3, n_sites)

        # flatten over individuals and sites → one distribution per genotype class
        for k in range(3):
            ax = axes[bin_idx, k]
            vals = logits_bin[:, k, :].ravel()  # (N * n_sites,)

            ax.hist(vals, bins=50, color=colors[k], alpha=0.75, density=True)
            ax.axvline(np.median(vals), color="black", linestyle="--",
                       linewidth=1.0, label=f"median={np.median(vals):.2f}")
            ax.set_title(f"{genotype_labels[k]}\n{bin_label}", fontsize=8)
            ax.set_xlabel("logit value", fontsize=7)
            ax.set_ylabel("density", fontsize=7)
            ax.legend(fontsize=7)

    fig.suptitle(f"{pop_label} — logit distributions by MAF bin", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"logit_distributions_{pop_label.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved logit distribution plot to: {out_path}")


def plot_logit_softmax_by_maf(
    logits: np.ndarray,
    site_bins: list[np.ndarray],
    bin_edges: np.ndarray,
    pop_label: str,
    output_dir: Path,
) -> None:
    """
    For each MAF bin, show mean softmax probability for each genotype class.
    This reveals degeneracies — if class 0 dominates everywhere, the model
    has collapsed to predicting the majority class.
    """
    from scipy.special import softmax as scipy_softmax

    n_bins = len(site_bins)
    genotype_labels = ["P(0)", "P(1)", "P(2)"]
    colors = ["steelblue", "darkorange", "forestgreen"]

    mean_probs = np.full((n_bins, 3), np.nan)

    for bin_idx, site_idx in enumerate(site_bins):
        if len(site_idx) == 0:
            continue
        logits_bin = logits[:, :, site_idx]          # (N, 3, n_sites)
        logits_flat = logits_bin.transpose(0, 2, 1)  # (N, n_sites, 3)
        logits_2d = logits_flat.reshape(-1, 3)        # (N*n_sites, 3)
        probs = scipy_softmax(logits_2d, axis=1)      # (N*n_sites, 3)
        mean_probs[bin_idx] = probs.mean(axis=0)

    bin_labels = [
        f"[{bin_edges[i]:.2f},{bin_edges[i+1]:.2f})"
        for i in range(n_bins)
    ]
    x = np.arange(n_bins)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(3):
        ax.bar(x + k * width, mean_probs[:, k], width=width,
               label=genotype_labels[k], color=colors[k], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(bin_labels, rotation=15)
    ax.set_ylabel("Mean softmax probability")
    ax.set_xlabel("MAF bin")
    ax.set_title(f"{pop_label} — mean predicted genotype probabilities by MAF bin")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / f"logit_softmax_by_maf_{pop_label.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved softmax probability plot to: {out_path}")


def plot_ceu_vs_yri_logits(
    logits_ceu: np.ndarray,
    logits_yri: np.ndarray,
    site_bins_ceu: list[np.ndarray],
    site_bins_yri: list[np.ndarray],
    bin_edges: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Side-by-side median logit per genotype class per MAF bin for CEU vs YRI.
    Reveals whether the model treats the two populations differently.
    """
    n_bins = len(site_bins_ceu)
    genotype_labels = ["class 0", "class 1", "class 2"]
    colors = ["steelblue", "darkorange", "forestgreen"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for k in range(3):
        ax = axes[k]
        medians_ceu, medians_yri = [], []

        for bin_idx in range(n_bins):
            idx_ceu = site_bins_ceu[bin_idx]
            idx_yri = site_bins_yri[bin_idx]

            if len(idx_ceu) > 0:
                medians_ceu.append(np.median(logits_ceu[:, k, idx_ceu]))
            else:
                medians_ceu.append(np.nan)

            if len(idx_yri) > 0:
                medians_yri.append(np.median(logits_yri[:, k, idx_yri]))
            else:
                medians_yri.append(np.nan)

        bin_labels = [
            f"[{bin_edges[i]:.2f},{bin_edges[i+1]:.2f})"
            for i in range(n_bins)
        ]
        x = np.arange(n_bins)

        ax.plot(x, medians_ceu, marker="o", label="CEU", color="steelblue")
        ax.plot(x, medians_yri, marker="^", label="YRI", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=15, fontsize=8)
        ax.set_ylabel("Median logit")
        ax.set_title(f"Logit {genotype_labels[k]}: CEU vs YRI by MAF bin")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("CEU vs YRI median logits by MAF bin and genotype class")
    fig.tight_layout()
    out_path = output_dir / "logit_ceu_vs_yri_by_maf.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved CEU vs YRI logit comparison to: {out_path}")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model_from_checkpoint(args.checkpoint, device)

    G_ceu = np.load(args.ceu_genotype_npy)
    G_yri = np.load(args.yri_genotype_npy)
    print(f"CEU shape: {G_ceu.shape}  YRI shape: {G_yri.shape}")

    print("Extracting CEU logits...")
    logits_ceu = extract_logits(model, G_ceu, device, args.batch_size)
    print("Extracting YRI logits...")
    logits_yri = extract_logits(model, G_yri, device, args.batch_size)

    maf_ceu = compute_maf(G_ceu)
    maf_yri = compute_maf(G_yri)

    bin_edges = np.array(args.freq_bins)
    site_bins_ceu = bin_sites_by_maf(maf_ceu, bin_edges)
    site_bins_yri = bin_sites_by_maf(maf_yri, bin_edges)

    for pop_label, logits, G, site_bins, maf in [
        ("CEU", logits_ceu, G_ceu, site_bins_ceu, maf_ceu),
        ("YRI", logits_yri, G_yri, site_bins_yri, maf_yri),
    ]:
        print(f"\n--- {pop_label} ---")
        for i, (idx, lo, hi) in enumerate(zip(
            site_bins, bin_edges[:-1], bin_edges[1:]
        )):
            print(f"  MAF [{lo:.2f}, {hi:.2f}): {len(idx)} sites")

        plot_logit_distributions(
            logits=logits,
            G_truth=G,
            maf=maf,
            bin_edges=bin_edges,
            site_bins=site_bins,
            pop_label=pop_label,
            output_dir=args.output_dir,
        )

        plot_logit_softmax_by_maf(
            logits=logits,
            site_bins=site_bins,
            bin_edges=bin_edges,
            pop_label=pop_label,
            output_dir=args.output_dir,
        )

    plot_ceu_vs_yri_logits(
        logits_ceu=logits_ceu,
        logits_yri=logits_yri,
        site_bins_ceu=site_bins_ceu,
        site_bins_yri=site_bins_yri,
        bin_edges=bin_edges,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()