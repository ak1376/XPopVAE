#!/usr/bin/env python3
"""
probe_overfitting.py

Checks whether the val R² gap vs gBLUP is due to outliers or genuine overfitting,
by comparing train vs val R² and plotting scatter plots for both.

Usage:
    python probe_overfitting.py \
        --checkpoint experiments/IM_symmetric/vae/default/vae_outputs/checkpoints/best_model.pt \
        --train_geno experiments/IM_symmetric/processed_data/0/rep0/discovery_train.npy \
        --val_geno   experiments/IM_symmetric/processed_data/0/rep0/discovery_val.npy \
        --train_pheno phenotype_creation/simulated_phenotype_train.npy \
        --val_pheno   phenotype_creation/simulated_phenotype_val.npy \
        --out_dir     probe_overfitting
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvVAE


# ------------------------------------------------------------------
# model loading
# ------------------------------------------------------------------
def load_model(checkpoint_path: Path, device: torch.device) -> ConvVAE:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["vae_config"]
    model = ConvVAE(
        input_length=ckpt["input_length"],
        in_channels=1,
        hidden_channels=cfg["model"]["hidden_channels"],
        kernel_size=int(cfg["model"]["kernel_size"]),
        stride=int(cfg["model"]["stride"]),
        padding=int(cfg["model"]["padding"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")
    return model


# ------------------------------------------------------------------
# inference
# ------------------------------------------------------------------
@torch.no_grad()
def get_predictions(model: ConvVAE, geno: np.ndarray, device: torch.device,
                    batch_size: int = 256) -> np.ndarray:
    preds = []
    x = torch.tensor(geno, dtype=torch.float32).unsqueeze(1)
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size].to(device)
        _, mu, _, _, pheno_pred = model(batch)
        preds.append(pheno_pred.cpu().numpy())
    return np.concatenate(preds, axis=0).squeeze(1)


# ------------------------------------------------------------------
# scatter plot helper
# ------------------------------------------------------------------
def scatter_panel(ax, y_true, y_pred, title, trimmed_std_threshold=3.0):
    r2_full = r2_score(y_true, y_pred)

    mask = np.abs(y_true - y_true.mean()) < trimmed_std_threshold * y_true.std()
    r2_trim = r2_score(y_true[mask], y_pred[mask])
    n_outliers = int((~mask).sum())

    ax.scatter(y_true[mask],  y_pred[mask],  alpha=0.4, s=8,  color="steelblue", label="inliers")
    ax.scatter(y_true[~mask], y_pred[~mask], alpha=0.9, s=30, color="tomato",    label=f"outliers (n={n_outliers})")

    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True phenotype")
    ax.set_ylabel("Predicted phenotype")
    ax.set_title(f"{title}\nR²={r2_full:.4f}  |  R² (trimmed)={r2_trim:.4f}")
    ax.legend(fontsize=8)

    return {"r2_full": r2_full, "r2_trimmed": r2_trim, "n_outliers": n_outliers}


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   type=Path, required=True)
    ap.add_argument("--train_geno",   type=Path, required=True)
    ap.add_argument("--val_geno",     type=Path, required=True)
    ap.add_argument("--train_pheno",  type=Path, required=True)
    ap.add_argument("--val_pheno",    type=Path, required=True)
    ap.add_argument("--out_dir",      type=Path, default=Path("probe_overfitting"))
    ap.add_argument("--batch_size",   type=int,  default=256)
    ap.add_argument("--std_thresh",   type=float, default=3.0,
                    help="Threshold (in std devs) for outlier trimming")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load
    model = load_model(args.checkpoint, device)

    X_train = np.load(args.train_geno)
    X_val   = np.load(args.val_geno)
    y_train = np.load(args.train_pheno)
    y_val   = np.load(args.val_pheno)

    # normalize phenotype the same way training did
    train_mean, train_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - train_mean) / train_std
    y_val_norm   = (y_val   - train_mean) / train_std

    # predictions (in normalized space)
    print("Running inference on train set...")
    pred_train = get_predictions(model, X_train, device, args.batch_size)
    print("Running inference on val set...")
    pred_val   = get_predictions(model, X_val,   device, args.batch_size)

    # un-normalize back to original scale for interpretability
    pred_train_orig = pred_train * train_std + train_mean
    pred_val_orig   = pred_val   * train_std + train_mean

    # ------------------------------------------------------------------
    # print summary
    # ------------------------------------------------------------------
    r2_train = r2_score(y_train, pred_train_orig)
    r2_val   = r2_score(y_val,   pred_val_orig)

    mask_train = np.abs(y_train - y_train.mean()) < args.std_thresh * y_train.std()
    mask_val   = np.abs(y_val   - y_val.mean())   < args.std_thresh * y_val.std()

    r2_train_trim = r2_score(y_train[mask_train], pred_train_orig[mask_train])
    r2_val_trim   = r2_score(y_val[mask_val],     pred_val_orig[mask_val])

    print()
    print("=" * 55)
    print("OVERFITTING DIAGNOSTIC")
    print("=" * 55)
    print(f"{'':30} {'Train':>10} {'Val':>10}")
    print(f"{'-'*50}")
    print(f"{'R² (full)':30} {r2_train:>10.4f} {r2_val:>10.4f}")
    print(f"{'R² (trimmed, >{args.std_thresh}σ removed)':30} {r2_train_trim:>10.4f} {r2_val_trim:>10.4f}")
    print(f"{'N outliers':30} {int((~mask_train).sum()):>10} {int((~mask_val).sum()):>10}")
    print(f"{'Train-Val R² gap (full)':30} {r2_train - r2_val:>10.4f}")
    print(f"{'Train-Val R² gap (trimmed)':30} {r2_train_trim - r2_val_trim:>10.4f}")
    print()

    if r2_train - r2_val > 0.05:
        print(">> Genuine overfitting likely — train/val gap > 0.05")
    else:
        print(">> No strong overfitting signal — train/val gap is small")

    if r2_val_trim - r2_val > 0.03:
        print(">> Outliers are driving the R² gap — trimmed R² is meaningfully higher")
    else:
        print(">> Outliers are not the main issue")

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter_panel(axes[0], y_train, pred_train_orig,
                  "Train (CEU)", trimmed_std_threshold=args.std_thresh)
    scatter_panel(axes[1], y_val,   pred_val_orig,
                  "Validation (CEU)", trimmed_std_threshold=args.std_thresh)

    plt.suptitle("Overfitting diagnostic: Train vs Val", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = args.out_dir / "train_vs_val_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to: {out_path}")


if __name__ == "__main__":
    main()