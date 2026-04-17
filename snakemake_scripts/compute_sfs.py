#!/usr/bin/env python3
# snakemake_scripts/compare_sfs.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


'''
python snakemake_scripts/compute_sfs.py \
    --checkpoint /sietch_colab/akapoor/XPopVAE/experiments/OOA/vae/default/vae_outputs/checkpoints/best_model.pt \
    --ceu-genotype-npy /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/genotype_matrices/discovery_train.npy \
    --yri-genotype-npy /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/genotype_matrices/target_held_out.npy \
    --output-dir /sietch_colab/akapoor/XPopVAE/experiments/OOA/vae/default/vae_outputs/sfs_comparison \
    --label "Reconstruction SFS comparison"



'''



# ------------------------------------------------------------------
# project paths & local imports
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_model_from_checkpoint, reconstruct_argmax_genotypes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SFS between ground-truth and VAE-reconstructed genotype matrices."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to saved checkpoint (.pt), e.g. best_model.pt",
    )
    parser.add_argument(
        "--ceu-genotype-npy",
        type=Path,
        required=True,
        help="Path to CEU genotype matrix .npy of shape (n_individuals, n_snps)",
    )
    parser.add_argument(
        "--yri-genotype-npy",
        type=Path,
        required=True,
        help="Path to YRI genotype matrix .npy of shape (n_individuals, n_snps)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for reconstruction.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="eval",
        help="Short label for this evaluation.",
    )
    return parser


# ------------------------------------------------------------------
# SFS computation
# ------------------------------------------------------------------
def compute_sfs_from_genotype_matrix(G: np.ndarray) -> np.ndarray:
    """
    Compute the 1D SFS from a diploid genotype matrix.

    Parameters
    ----------
    G : np.ndarray, shape (n_individuals, n_snps)
        Genotype matrix coded 0/1/2.

    Returns
    -------
    sfs : np.ndarray, shape (2*n_individuals + 1,)
        sfs[i] = number of sites where derived allele count == i.
    """
    n_individuals = G.shape[0]
    n_chrom = 2 * n_individuals

    allele_counts = G.sum(axis=0).astype(int)  # (n_snps,)
    sfs = np.bincount(allele_counts, minlength=n_chrom + 1)
    return sfs


# ------------------------------------------------------------------
# SFS plotting
# ------------------------------------------------------------------
def plot_sfs_comparison(
    sfs_ceu_truth: np.ndarray,
    sfs_ceu_recon: np.ndarray,
    sfs_yri_truth: np.ndarray,
    sfs_yri_recon: np.ndarray,
    output_path: Path,
    title: str = "SFS: truth vs reconstructed",
    max_count: int = 100,
    n_bins: int = 20,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sfs_truth, sfs_recon, pop_label in zip(
        axes,
        [sfs_ceu_truth, sfs_yri_truth],
        [sfs_ceu_recon, sfs_yri_recon],
        ["CEU", "YRI"],
    ):
        end = min(max_count + 1, len(sfs_truth) - 1)

        # integer bin edges so every count is captured exactly once
        bin_edges = np.linspace(1, end, n_bins + 1).astype(int)
        bin_edges = np.unique(bin_edges)
        n_bins_actual = len(bin_edges) - 1
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = (bin_centers[1] - bin_centers[0]) * 0.4 if n_bins_actual > 1 else 1.0

        truth_binned = np.array([
            sfs_truth[bin_edges[i]:bin_edges[i + 1]].sum()
            for i in range(n_bins_actual)
        ])
        recon_binned = np.array([
            sfs_recon[bin_edges[i]:bin_edges[i + 1]].sum()
            for i in range(n_bins_actual)
        ])

        ax.bar(bin_centers - width / 2, truth_binned, width=width,
               color="steelblue", alpha=0.8, label="truth")
        ax.bar(bin_centers + width / 2, recon_binned, width=width,
               color="darkorange", alpha=0.8, label="reconstructed")

        ax.set_yscale("log")
        ax.set_xlabel("Derived allele count")
        ax.set_ylabel("Number of sites (log scale)")
        ax.set_title(f"{pop_label} SFS: truth vs reconstructed")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved SFS comparison plot to: {output_path}")

def compute_sfs_metrics(
    sfs_truth: np.ndarray,
    sfs_recon: np.ndarray,
) -> dict[str, float]:
    # exclude fixed bins
    t = sfs_truth[1:-1].astype(float)
    r = sfs_recon[1:-1].astype(float)

    diff = r - t
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))

    total = t.sum()
    t_norm = t / total if total > 0 else t
    r_norm = r / r.sum() if r.sum() > 0 else r
    kl = float(np.sum(t_norm * np.log((t_norm + 1e-10) / (r_norm + 1e-10))))

    return {"mae": mae, "rmse": rmse, "bias": bias, "kl_div": kl}


def save_summary(
    output_path: Path,
    label: str,
    checkpoint: Path,
    ceu_path: Path,
    yri_path: Path,
    metrics_ceu: dict[str, float],
    metrics_yri: dict[str, float],
) -> None:
    with open(output_path, "w") as f:
        f.write(f"Label:      {label}\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"CEU input:  {ceu_path}\n")
        f.write(f"YRI input:  {yri_path}\n\n")

        for pop, metrics in [("CEU", metrics_ceu), ("YRI", metrics_yri)]:
            f.write(f"{pop} SFS metrics (truth vs reconstructed)\n")
            f.write(f"  MAE:    {metrics['mae']:.8f}\n")
            f.write(f"  RMSE:   {metrics['rmse']:.8f}\n")
            f.write(f"  Bias:   {metrics['bias']:.8f}\n")
            f.write(f"  KL div: {metrics['kl_div']:.8f}\n\n")

    print(f"Saved summary to: {output_path}")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device)

    print(f"Loading CEU genotype matrix: {args.ceu_genotype_npy}")
    G_ceu_truth = np.load(args.ceu_genotype_npy)
    print(f"Loading YRI genotype matrix: {args.yri_genotype_npy}")
    G_yri_truth = np.load(args.yri_genotype_npy)

    for name, G in [("CEU", G_ceu_truth), ("YRI", G_yri_truth)]:
        if G.ndim != 2:
            raise ValueError(
                f"Expected {name} genotype matrix of shape (n_individuals, n_snps), got {G.shape}"
            )
    if G_ceu_truth.shape[1] != G_yri_truth.shape[1]:
        raise ValueError(
            f"CEU and YRI matrices must have the same number of SNPs: "
            f"{G_ceu_truth.shape[1]} vs {G_yri_truth.shape[1]}"
        )

    print("Reconstructing CEU genotypes...")
    G_ceu_recon = reconstruct_argmax_genotypes(
        model=model, G=G_ceu_truth, device=device, batch_size=args.batch_size,
    )
    print("Reconstructing YRI genotypes...")
    G_yri_recon = reconstruct_argmax_genotypes(
        model=model, G=G_yri_truth, device=device, batch_size=args.batch_size,
    )

    np.save(args.output_dir / "reconstructed_ceu_argmax.npy", G_ceu_recon)
    np.save(args.output_dir / "reconstructed_yri_argmax.npy", G_yri_recon)

    # compute SFS
    sfs_ceu_truth = compute_sfs_from_genotype_matrix(G_ceu_truth)
    sfs_ceu_recon = compute_sfs_from_genotype_matrix(G_ceu_recon)
    sfs_yri_truth = compute_sfs_from_genotype_matrix(G_yri_truth)
    sfs_yri_recon = compute_sfs_from_genotype_matrix(G_yri_recon)

    np.save(args.output_dir / "sfs_ceu_truth.npy", sfs_ceu_truth)
    np.save(args.output_dir / "sfs_ceu_recon.npy", sfs_ceu_recon)
    np.save(args.output_dir / "sfs_yri_truth.npy", sfs_yri_truth)
    np.save(args.output_dir / "sfs_yri_recon.npy", sfs_yri_recon)

    print(f'[DEBUG]')
    print("CEU truth first 10 counts:", sfs_ceu_truth[1:11])
    print("CEU recon first 10 counts:", sfs_ceu_recon[1:11])

    # plot
    plot_sfs_comparison(
        sfs_ceu_truth=sfs_ceu_truth,
        sfs_ceu_recon=sfs_ceu_recon,
        sfs_yri_truth=sfs_yri_truth,
        sfs_yri_recon=sfs_yri_recon,
        output_path=args.output_dir / "sfs_truth_vs_reconstructed.png",
    )

    # metrics + summary
    metrics_ceu = compute_sfs_metrics(sfs_ceu_truth, sfs_ceu_recon)
    metrics_yri = compute_sfs_metrics(sfs_yri_truth, sfs_yri_recon)

    np.savez(
        args.output_dir / "sfs_metrics.npz",
        ceu_mae=metrics_ceu["mae"],   ceu_rmse=metrics_ceu["rmse"],
        ceu_bias=metrics_ceu["bias"], ceu_kl=metrics_ceu["kl_div"],
        yri_mae=metrics_yri["mae"],   yri_rmse=metrics_yri["rmse"],
        yri_bias=metrics_yri["bias"], yri_kl=metrics_yri["kl_div"],
    )

    save_summary(
        output_path=args.output_dir / "sfs_summary.txt",
        label=args.label,
        checkpoint=args.checkpoint,
        ceu_path=args.ceu_genotype_npy,
        yri_path=args.yri_genotype_npy,
        metrics_ceu=metrics_ceu,
        metrics_yri=metrics_yri,
    )

    print(f"[{args.label}] CEU SFS — MAE={metrics_ceu['mae']:.6f}  KL={metrics_ceu['kl_div']:.6f}")
    print(f"[{args.label}] YRI SFS — MAE={metrics_yri['mae']:.6f}  KL={metrics_yri['kl_div']:.6f}")


if __name__ == "__main__":
    main()