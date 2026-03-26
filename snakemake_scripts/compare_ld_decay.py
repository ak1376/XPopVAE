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
        "--variant-positions-npy",
        type=Path,
        default=None,
        help="Path to variant_positions_bp.npy. Required for --distance-mode bp.",
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
        "--distance-mode",
        type=str,
        choices=["lag", "bp"],
        default="bp",
        help="How to compute LD decay: by SNP lag or by physical distance in bp.",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=100,
        help="Maximum SNP lag to include when --distance-mode lag.",
    )
    parser.add_argument(
        "--max-bp-distance",
        type=float,
        default=50_000.0,
        help="Maximum physical distance in bp to include when --distance-mode bp.",
    )
    parser.add_argument(
        "--bp-bin-size",
        type=float,
        default=1_000.0,
        help="Bin width in bp when --distance-mode bp.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="LD decay: truth vs reconstructed",
        help="Base plot title.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="eval",
        help="Short label for this evaluation, e.g. discovery_val or target_yri",
    )
    parser.add_argument(
        "--include-metrics-in-title",
        action="store_true",
        help="If set, append MAE / RMSE / bias to the plot title.",
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


def _precompute_centered_stats(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    G = np.asarray(G, dtype=np.float32)
    means = G.mean(axis=0, keepdims=True)
    Gc = G - means
    var = np.mean(Gc * Gc, axis=0)
    return Gc, var


def compute_ld_decay_by_lag(
    G: np.ndarray,
    max_distance: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    G = np.asarray(G, dtype=np.float32)
    _, n_snps = G.shape

    Gc, var = _precompute_centered_stats(G)

    max_distance = min(max_distance, n_snps - 1)
    distances = np.arange(1, max_distance + 1, dtype=int)
    mean_r2 = np.full(max_distance, np.nan, dtype=np.float64)

    for idx, d in enumerate(distances):
        Xc = Gc[:, :-d]
        Yc = Gc[:, d:]

        X_var = var[:-d]
        Y_var = var[d:]

        valid = (X_var > 0) & (Y_var > 0)
        if not np.any(valid):
            continue

        cov = np.mean(Xc[:, valid] * Yc[:, valid], axis=0)
        r = cov / np.sqrt(X_var[valid] * Y_var[valid])
        r2 = r**2

        mean_r2[idx] = np.mean(r2)

    return distances, mean_r2


def compute_ld_decay_by_bp(
    G: np.ndarray,
    positions_bp: np.ndarray,
    max_bp_distance: float = 50_000.0,
    bp_bin_size: float = 1_000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LD decay by physical distance bins.

    Returns
    -------
    bin_centers : (n_bins,)
        Center of each bp distance bin.
    mean_r2 : (n_bins,)
        Mean r^2 in each bp distance bin.
    pair_counts : (n_bins,)
        Number of SNP pairs contributing to each bin.
    """
    G = np.asarray(G, dtype=np.float32)
    positions_bp = np.asarray(positions_bp, dtype=np.float64)

    n_individuals, n_snps = G.shape
    if positions_bp.ndim != 1 or positions_bp.shape[0] != n_snps:
        raise ValueError(
            f"positions_bp must have shape ({n_snps},), got {positions_bp.shape}"
        )
    if bp_bin_size <= 0:
        raise ValueError(f"bp_bin_size must be > 0, got {bp_bin_size}")
    if max_bp_distance <= 0:
        raise ValueError(f"max_bp_distance must be > 0, got {max_bp_distance}")

    Gc, var = _precompute_centered_stats(G)

    n_bins = int(np.ceil(max_bp_distance / bp_bin_size))
    bin_edges = np.arange(0, n_bins + 1, dtype=np.float64) * bp_bin_size
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    sum_r2 = np.zeros(n_bins, dtype=np.float64)
    count_r2 = np.zeros(n_bins, dtype=np.int64)

    # Iterate over lag, but assign each pair to a physical-distance bin
    for lag in range(1, n_snps):
        pair_dist_bp = positions_bp[lag:] - positions_bp[:-lag]

        # positions are sorted, so once all distances exceed max, we can stop
        if pair_dist_bp.size == 0:
            break
        if pair_dist_bp.min() > max_bp_distance:
            break

        within = pair_dist_bp <= max_bp_distance
        if not np.any(within):
            continue

        Xc = Gc[:, :-lag]
        Yc = Gc[:, lag:]

        X_var = var[:-lag]
        Y_var = var[lag:]

        valid = within & (X_var > 0) & (Y_var > 0)
        if not np.any(valid):
            continue

        cov = np.mean(Xc[:, valid] * Yc[:, valid], axis=0)
        r = cov / np.sqrt(X_var[valid] * Y_var[valid])
        r2 = r**2

        valid_dist = pair_dist_bp[valid]
        bin_idx = np.floor(valid_dist / bp_bin_size).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        # accumulate by bin
        np.add.at(sum_r2, bin_idx, r2)
        np.add.at(count_r2, bin_idx, 1)

    mean_r2 = np.full(n_bins, np.nan, dtype=np.float64)
    nonzero = count_r2 > 0
    mean_r2[nonzero] = sum_r2[nonzero] / count_r2[nonzero]

    return bin_centers, mean_r2, count_r2


def compute_curve_metrics(truth_r2: np.ndarray, recon_r2: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(truth_r2) & np.isfinite(recon_r2)
    if not np.any(valid):
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "max_abs_err": np.nan,
            "corr": np.nan,
        }

    diff = recon_r2[valid] - truth_r2[valid]

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    max_abs_err = float(np.max(np.abs(diff)))

    if np.sum(valid) >= 2:
        corr = float(np.corrcoef(truth_r2[valid], recon_r2[valid])[0, 1])
    else:
        corr = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "max_abs_err": max_abs_err,
        "corr": corr,
    }


def save_ld_curve_arrays(
    output_path: Path,
    distances: np.ndarray,
    truth_r2: np.ndarray,
    recon_r2: np.ndarray,
    metrics: dict[str, float],
    distance_mode: str,
    pair_counts: np.ndarray | None = None,
) -> None:
    payload = dict(
        distances=distances,
        truth_mean_r2=truth_r2,
        recon_mean_r2=recon_r2,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        bias=metrics["bias"],
        max_abs_err=metrics["max_abs_err"],
        corr=metrics["corr"],
        distance_mode=distance_mode,
    )
    if pair_counts is not None:
        payload["pair_counts"] = pair_counts

    np.savez(output_path, **payload)
    print(f"Saved LD curve arrays to: {output_path}")


def make_title(base_title: str, metrics: dict[str, float], include_metrics: bool) -> str:
    if not include_metrics:
        return base_title
    return (
        f"{base_title}\n"
        f"MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
        f"bias={metrics['bias']:.4f} | corr={metrics['corr']:.4f}"
    )


def plot_ld_decay(
    distances: np.ndarray,
    truth_r2: np.ndarray,
    recon_r2: np.ndarray,
    output_path: Path,
    title: str,
    distance_mode: str,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(distances, truth_r2, label="Ground truth", marker="o", markersize=3)
    plt.plot(distances, recon_r2, label="Reconstructed (argmax)", marker="o", markersize=3)

    if distance_mode == "bp":
        plt.xlabel("Physical distance (bp)")
    else:
        plt.xlabel("SNP distance (lag)")

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

    if args.distance_mode == "bp" and args.variant_positions_npy is None:
        raise ValueError("--variant-positions-npy is required when --distance-mode bp")

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

    positions_bp = None
    if args.distance_mode == "bp":
        print(f"Loading variant positions: {args.variant_positions_npy}")
        positions_bp = np.load(args.variant_positions_npy)
        if positions_bp.shape[0] != G_truth.shape[1]:
            raise ValueError(
                "Number of variant positions does not match number of SNP columns: "
                f"{positions_bp.shape[0]} vs {G_truth.shape[1]}"
            )

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

    pair_counts = None

    if args.distance_mode == "bp":
        print("Computing LD decay for ground truth by physical distance...")
        distances, truth_r2, pair_counts = compute_ld_decay_by_bp(
            G_truth,
            positions_bp=positions_bp,
            max_bp_distance=args.max_bp_distance,
            bp_bin_size=args.bp_bin_size,
        )

        print("Computing LD decay for reconstructed matrix by physical distance...")
        distances2, recon_r2, pair_counts_recon = compute_ld_decay_by_bp(
            G_recon,
            positions_bp=positions_bp,
            max_bp_distance=args.max_bp_distance,
            bp_bin_size=args.bp_bin_size,
        )

        if not np.array_equal(distances, distances2):
            raise RuntimeError("Distance arrays do not match.")
        if not np.array_equal(pair_counts, pair_counts_recon):
            print("Warning: pair count arrays differ between truth and recon. Using truth counts in output.")
    else:
        print("Computing LD decay for ground truth by SNP lag...")
        distances, truth_r2 = compute_ld_decay_by_lag(
            G_truth,
            max_distance=args.max_distance,
        )

        print("Computing LD decay for reconstructed matrix by SNP lag...")
        distances2, recon_r2 = compute_ld_decay_by_lag(
            G_recon,
            max_distance=args.max_distance,
        )

        if not np.array_equal(distances, distances2):
            raise RuntimeError("Distance arrays do not match.")

    metrics = compute_curve_metrics(truth_r2, recon_r2)

    curves_npz = args.output_dir / "ld_decay_curves.npz"
    save_ld_curve_arrays(
        output_path=curves_npz,
        distances=distances,
        truth_r2=truth_r2,
        recon_r2=recon_r2,
        metrics=metrics,
        distance_mode=args.distance_mode,
        pair_counts=pair_counts,
    )

    full_title = make_title(
        base_title=args.title,
        metrics=metrics,
        include_metrics=args.include_metrics_in_title,
    )

    plot_path = args.output_dir / "ld_decay_truth_vs_reconstructed.png"
    plot_ld_decay(
        distances=distances,
        truth_r2=truth_r2,
        recon_r2=recon_r2,
        output_path=plot_path,
        title=full_title,
        distance_mode=args.distance_mode,
    )

    summary_path = args.output_dir / "ld_decay_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Label: {args.label}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Genotype input: {args.genotype_npy}\n")
        f.write(f"Shape: {G_truth.shape}\n")
        f.write(f"Distance mode: {args.distance_mode}\n")
        if args.distance_mode == "bp":
            f.write(f"Variant positions: {args.variant_positions_npy}\n")
            f.write(f"Max bp distance: {args.max_bp_distance}\n")
            f.write(f"BP bin size: {args.bp_bin_size}\n")
        else:
            f.write(f"Max lag distance: {args.max_distance}\n")
        f.write("\n")
        f.write("Curve discorrespondence metrics\n")
        f.write(f"MAE:         {metrics['mae']:.8f}\n")
        f.write(f"RMSE:        {metrics['rmse']:.8f}\n")
        f.write(f"Bias:        {metrics['bias']:.8f}\n")
        f.write(f"MaxAbsError: {metrics['max_abs_err']:.8f}\n")
        f.write(f"Correlation: {metrics['corr']:.8f}\n")
        f.write("\n")
        if pair_counts is not None:
            f.write("First 10 bins / pair counts / truth / recon / diff:\n")
            for d, c, t, r in zip(distances[:10], pair_counts[:10], truth_r2[:10], recon_r2[:10]):
                f.write(f"{d}\t{c}\t{t:.6f}\t{r:.6f}\t{(r-t):.6f}\n")
        else:
            f.write("First 10 distances / truth / recon / diff:\n")
            for d, t, r in zip(distances[:10], truth_r2[:10], recon_r2[:10]):
                f.write(f"{d}\t{t:.6f}\t{r:.6f}\t{(r-t):.6f}\n")

    print(f"Saved summary to: {summary_path}")
    print(
        f"[{args.label}] LD metrics: "
        f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
        f"bias={metrics['bias']:.6f}, corr={metrics['corr']:.6f}"
    )


if __name__ == "__main__":
    main()