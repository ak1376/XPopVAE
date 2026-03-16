#!/usr/bin/env python3
# snakemake_scripts/compare_ld_decay.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
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
        description="Compare LD decay curves between ground-truth and VAE-reconstructed genotype matrices."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to saved checkpoint (.pt), e.g. best_model.pt",
    )
    parser.add_argument(
        "--genotype-npy",
        type=Path,
        required=True,
        help="Path to genotype matrix .npy of shape (n_individuals, n_snps)",
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
        "--max-distance",
        type=int,
        default=100,
        help="Maximum SNP distance (lag) to include in LD decay.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="LD decay: truth vs reconstructed",
        help="Plot title.",
    )
    return parser


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> ConvVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vae_config = checkpoint["vae_config"]
    input_length = int(checkpoint["input_length"])

    model = ConvVAE(
        input_length=input_length,
        in_channels=1,
        hidden_channels=vae_config["model"]["hidden_channels"],
        kernel_size=int(vae_config["model"]["kernel_size"]),
        stride=int(vae_config["model"]["stride"]),
        padding=int(vae_config["model"]["padding"]),
        latent_dim=int(vae_config["model"]["latent_dim"]),
        use_batchnorm=False,
        activation="elu",
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
    """
    Reconstruct genotype matrix using argmax genotype calls.

    Parameters
    ----------
    model : torch.nn.Module
        Trained ConvVAE
    G : np.ndarray
        Input genotype matrix, shape (n_individuals, n_snps)
    device : torch.device
    batch_size : int

    Returns
    -------
    np.ndarray
        Reconstructed hard genotype calls, shape (n_individuals, n_snps),
        values in {0,1,2}
    """
    X = torch.tensor(G, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    recon_batches = []

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            logits, mu, logvar, z = model(x)          # (B,3,L)
            pred = torch.argmax(logits, dim=1)        # (B,L)
            recon_batches.append(pred.cpu().numpy())

    return np.concatenate(recon_batches, axis=0)


def compute_ld_decay_by_lag(G: np.ndarray, max_distance: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean r^2 by SNP lag in a vectorized way.

    For each lag d:
      compare G[:, :-d] vs G[:, d:]
      compute column-wise Pearson r^2 across individuals
      average across SNP pairs at that lag

    Parameters
    ----------
    G : np.ndarray
        Genotype matrix of shape (n_individuals, n_snps)
    max_distance : int
        Maximum SNP lag

    Returns
    -------
    distances : np.ndarray
        Array [1, 2, ..., max_distance]
    mean_r2 : np.ndarray
        Mean r^2 at each lag
    """
    G = np.asarray(G, dtype=np.float32)
    n_individuals, n_snps = G.shape

    max_distance = min(max_distance, n_snps - 1)
    distances = np.arange(1, max_distance + 1, dtype=int)
    mean_r2 = np.full(max_distance, np.nan, dtype=np.float64)

    # loop over lag only; inside each lag is vectorized over all SNP pairs
    for idx, d in enumerate(distances):
        X = G[:, :-d]   # shape (N, M)
        Y = G[:, d:]    # shape (N, M)
        # M = number of SNP pairs at this lag

        # column means
        X_mean = X.mean(axis=0, keepdims=True)
        Y_mean = Y.mean(axis=0, keepdims=True)

        # centered
        Xc = X - X_mean
        Yc = Y - Y_mean

        # column variances / stds
        X_var = np.mean(Xc * Xc, axis=0)
        Y_var = np.mean(Yc * Yc, axis=0)

        valid = (X_var > 0) & (Y_var > 0)
        if not np.any(valid):
            continue

        cov = np.mean(Xc[:, valid] * Yc[:, valid], axis=0)
        r = cov / np.sqrt(X_var[valid] * Y_var[valid])
        r2 = r ** 2

        mean_r2[idx] = np.mean(r2)

    return distances, mean_r2


def save_ld_curve_arrays(
    output_path: Path,
    distances: np.ndarray,
    truth_r2: np.ndarray,
    recon_r2: np.ndarray,
) -> None:
    np.savez(
        output_path,
        distances=distances,
        truth_mean_r2=truth_r2,
        recon_mean_r2=recon_r2,
    )
    print(f"Saved LD curve arrays to: {output_path}")


def plot_ld_decay(
    distances: np.ndarray,
    truth_r2: np.ndarray,
    recon_r2: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(distances, truth_r2, label="Ground truth", marker="o", markersize=3)
    plt.plot(distances, recon_r2, label="Reconstructed (argmax)", marker="o", markersize=3)
    plt.xlabel("SNP distance")
    plt.ylabel("Mean $r^2$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved LD decay plot to: {output_path}")


def main():
    args = build_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device)

    print(f"Loading genotype matrix: {args.genotype_npy}")
    G_truth = np.load(args.genotype_npy)
    if G_truth.ndim != 2:
        raise ValueError(
            f"Expected genotype matrix of shape (n_individuals, n_snps), got {G_truth.shape}"
        )

    print(f"Genotype matrix shape: {G_truth.shape}")

    print("Reconstructing genotype matrix with argmax...")
    G_recon = reconstruct_argmax_genotypes(
        model=model,
        G=G_truth,
        device=device,
        batch_size=args.batch_size,
    )

    recon_path = args.output_dir / "reconstructed_genotypes_argmax.npy"
    np.save(recon_path, G_recon)
    print(f"Saved reconstructed genotype matrix to: {recon_path}")

    print("Computing LD decay for ground truth...")
    distances, truth_r2 = compute_ld_decay_by_lag(
        G_truth,
        max_distance=args.max_distance,
    )

    print("Computing LD decay for reconstructed matrix...")
    distances2, recon_r2 = compute_ld_decay_by_lag(
        G_recon,
        max_distance=args.max_distance,
    )

    if not np.array_equal(distances, distances2):
        raise RuntimeError("Distance arrays do not match.")

    curves_npz = args.output_dir / "ld_decay_curves.npz"
    save_ld_curve_arrays(
        output_path=curves_npz,
        distances=distances,
        truth_r2=truth_r2,
        recon_r2=recon_r2,
    )

    plot_path = args.output_dir / "ld_decay_truth_vs_reconstructed.png"
    plot_ld_decay(
        distances=distances,
        truth_r2=truth_r2,
        recon_r2=recon_r2,
        output_path=plot_path,
        title=args.title,
    )

    # small text summary
    summary_path = args.output_dir / "ld_decay_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Genotype input: {args.genotype_npy}\n")
        f.write(f"Shape: {G_truth.shape}\n")
        f.write(f"Max distance: {args.max_distance}\n")
        f.write("\n")
        f.write("First 10 distances / truth / recon:\n")
        for d, t, r in zip(distances[:10], truth_r2[:10], recon_r2[:10]):
            f.write(f"{d}\t{t:.6f}\t{r:.6f}\n")

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()