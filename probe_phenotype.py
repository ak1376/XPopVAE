#!/usr/bin/env python3
"""
probe_phenotype.py

Probes how much phenotype information lives in the VAE's mu vectors,
without any joint training. The encoder is frozen; mu vectors are extracted
and a ridge regression is fit on top.

Splits
------
  Train probe  : CEU discovery_train   (in-population, labeled)
  Eval (main)  : CEU discovery_validation
  Eval (xpop)  : YRI target_train or target_held_out (cross-population transfer, optional)
                 If target_train is empty (target_held_out_frac=1.0), automatically
                 falls back to target_held_out.npy from the same directory.

Usage
-----
    python probe_phenotype.py \
        --checkpoint   /sietch_colab/akapoor/XPopVAE/experiments/OOA/vae/default/vae_outputs/checkpoints/best_model.pt \
        --disc-train-geno   /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/genotype_matrices/discovery_train.npy \
        --disc-val-geno     /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/genotype_matrices/discovery_validation.npy \
        --target-train-geno /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/genotype_matrices/target_train.npy \
        --disc-train-pheno  /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/phenotypes/discovery_train_pheno.npy \
        --disc-val-pheno    /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/phenotypes/discovery_validation_pheno.npy \
        --target-train-pheno /sietch_colab/akapoor/XPopVAE/experiments/OOA/processed_data/0/rep0/phenotypes/target_train_pheno.npy \
        --output-dir   probe_outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# project path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvVAE


# =============================================================================
# Helpers
# =============================================================================

def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt["vae_config"]
    m    = cfg["model"]
    da   = cfg.get("domain_adaptation", {})
    ph   = cfg.get("phenotype", {})

    model = ConvVAE(
        input_length    = ckpt["input_length"],
        in_channels     = 1,
        hidden_channels = m["hidden_channels"],
        kernel_size     = int(m["kernel_size"]),
        stride          = int(m["stride"]),
        padding         = int(m["padding"]),
        latent_dim      = int(m["latent_dim"]),
        use_batchnorm   = bool(m.get("use_batchnorm", False)),
        activation      = m.get("activation", "elu"),
        pheno_dim       = 1,
        pheno_hidden_dim= ph.get("pheno_hidden_dim", None),
        use_grl         = bool(da.get("use_grl", False)),
        grl_hidden_dim  = da.get("grl_hidden_dim", None),
        num_domains     = 2,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loaded checkpoint  : {path}")
    print(f"  saved at epoch   : {ckpt.get('epoch', '?')}")
    print(f"  val stop metric  : {ckpt.get('val_loss', '?'):.6f}")
    print(f"  input_length     : {ckpt['input_length']}")
    print(f"  latent_dim       : {m['latent_dim']}")
    return model


@torch.no_grad()
def extract_mu(model: ConvVAE, geno: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """Pass genotypes through the frozen encoder; return mu as (N, latent_dim)."""
    geno_t = torch.tensor(geno, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
    mus = []
    for start in range(0, len(geno_t), batch_size):
        batch = geno_t[start : start + batch_size].to(device)
        _, mu, _, _, _, _ = model(batch)
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def fit_probe(
    mu_train : np.ndarray,
    y_train  : np.ndarray,
    alphas   : tuple = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
) -> tuple[RidgeCV, StandardScaler]:
    """Fit a RidgeCV probe on mu_train -> y_train (phenotypes)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(mu_train)
    probe = RidgeCV(alphas=alphas, cv=5)
    probe.fit(X, y_train)
    print(f"  Ridge best alpha : {probe.alpha_:.4g}")
    return probe, scaler


def evaluate_probe(
    probe   : RidgeCV,
    scaler  : StandardScaler,
    mu      : np.ndarray,
    y_true  : np.ndarray,
    split   : str,
) -> dict:
    X     = scaler.transform(mu)
    y_hat = probe.predict(X)
    rmse  = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    r2    = float(r2_score(y_true, y_hat))
    corr  = float(np.corrcoef(y_true, y_hat)[0, 1])
    print(f"  [{split:30s}]  R²={r2:.4f}  RMSE={rmse:.4f}  r={corr:.4f}")
    return {"split": split, "r2": r2, "rmse": rmse, "pearson_r": corr}


def scatter_plot(
    probe      : RidgeCV,
    scaler     : StandardScaler,
    mu         : np.ndarray,
    y_true     : np.ndarray,
    split      : str,
    out_path   : Path,
    color      : str = "steelblue",
):
    X     = scaler.transform(mu)
    y_hat = probe.predict(X)
    r2    = r2_score(y_true, y_hat)
    corr  = np.corrcoef(y_true, y_hat)[0, 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_hat, alpha=0.35, s=10, color=color, rasterized=True)

    lo = min(y_true.min(), y_hat.min())
    hi = max(y_true.max(), y_hat.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")

    ax.set_xlabel("True phenotype")
    ax.set_ylabel("Predicted phenotype")
    ax.set_title(f"{split}\nR²={r2:.3f}  r={corr:.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metrics(metrics_list: list[dict], out_path: Path):
    with open(out_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"\nMetrics saved to : {out_path}")


def resolve_yri_split(
    target_train_geno_path  : str | None,
    target_train_pheno_path : str | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Resolve which YRI split to use for cross-population evaluation.

    Priority:
      1. target_train if non-empty
      2. target_held_out from the same directory (fallback when target_held_out_frac=1.0)
      3. None if neither is available

    Returns (geno, pheno, split_label).
    """
    if target_train_geno_path is None or target_train_pheno_path is None:
        return None, None, ""

    geno_path  = Path(target_train_geno_path)
    pheno_path = Path(target_train_pheno_path)

    geno = np.load(geno_path)

    # FIX: if target_train is non-empty, use it directly
    if len(geno) > 0:
        pheno = np.load(pheno_path).astype(np.float32).ravel()
        print(f"  target_train    : {geno.shape}  (using target_train)")
        return geno, pheno, "YRI target_train (cross-pop)"

    # FIX: fallback to target_held_out from the same genotype_matrices/ directory
    held_out_geno_path  = geno_path.parent  / "target_held_out.npy"
    held_out_pheno_path = pheno_path.parent / "target_held_out_pheno.npy"

    if held_out_geno_path.exists() and held_out_pheno_path.exists():
        geno  = np.load(held_out_geno_path)
        pheno = np.load(held_out_pheno_path).astype(np.float32).ravel()
        if len(geno) > 0:
            print(f"  target_train empty — falling back to target_held_out: {geno.shape}")
            return geno, pheno, "YRI target_held_out (cross-pop)"

    print("  target_train empty and target_held_out not found — skipping cross-pop eval")
    return None, None, ""


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model (frozen)
    # ------------------------------------------------------------------
    model = load_checkpoint(Path(args.checkpoint), device)

    # ------------------------------------------------------------------
    # Load genotypes
    # ------------------------------------------------------------------
    print("\nLoading genotypes…")
    disc_train_geno = np.load(args.disc_train_geno)
    disc_val_geno   = np.load(args.disc_val_geno)
    print(f"  disc_train  : {disc_train_geno.shape}")
    print(f"  disc_val    : {disc_val_geno.shape}")

    # Resolve YRI split (target_train or target_held_out fallback)
    target_geno, target_pheno_raw, yri_split_label = resolve_yri_split(
        args.target_train_geno,
        args.target_train_pheno,
    )
    has_yri = target_geno is not None

    # ------------------------------------------------------------------
    # Load phenotypes (raw, will normalise using disc_train stats)
    # ------------------------------------------------------------------
    print("\nLoading phenotypes…")
    disc_train_pheno = np.load(args.disc_train_pheno).astype(np.float32).ravel()
    disc_val_pheno   = np.load(args.disc_val_pheno).astype(np.float32).ravel()

    mean_ = disc_train_pheno.mean()
    std_  = disc_train_pheno.std()
    print(f"  Pheno normalisation — mean={mean_:.4f}  std={std_:.4f}  (CEU disc_train)")

    y_disc_train = (disc_train_pheno - mean_) / std_
    y_disc_val   = (disc_val_pheno   - mean_) / std_

    if has_yri:
        y_target = (target_pheno_raw - mean_) / std_

    # ------------------------------------------------------------------
    # Extract mu vectors
    # ------------------------------------------------------------------
    print("\nExtracting mu vectors…")
    mu_disc_train = extract_mu(model, disc_train_geno, args.batch_size, device)
    mu_disc_val   = extract_mu(model, disc_val_geno,   args.batch_size, device)
    print(f"  mu disc_train  : {mu_disc_train.shape}")
    print(f"  mu disc_val    : {mu_disc_val.shape}")

    if has_yri:
        mu_target = extract_mu(model, target_geno, args.batch_size, device)
        print(f"  mu target      : {mu_target.shape}")

    if args.save_mus:
        np.save(out / "mu_disc_train.npy", mu_disc_train)
        np.save(out / "mu_disc_val.npy",   mu_disc_val)
        if has_yri:
            np.save(out / "mu_target.npy", mu_target)
        print("  mu arrays saved.")

    # ------------------------------------------------------------------
    # Fit probe on CEU disc_train
    # ------------------------------------------------------------------
    print("\nFitting ridge probe on CEU disc_train mu…")
    probe, scaler = fit_probe(mu_disc_train, y_disc_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\nEvaluating probe…")
    all_metrics = []

    m = evaluate_probe(probe, scaler, mu_disc_train, y_disc_train, "CEU disc_train (in-sample)")
    all_metrics.append(m)
    scatter_plot(probe, scaler, mu_disc_train, y_disc_train,
                 "CEU disc_train (in-sample)", out / "scatter_disc_train.png",
                 color="steelblue")

    m = evaluate_probe(probe, scaler, mu_disc_val, y_disc_val, "CEU disc_val (held-out)")
    all_metrics.append(m)
    scatter_plot(probe, scaler, mu_disc_val, y_disc_val,
                 "CEU disc_val (held-out)", out / "scatter_disc_val.png",
                 color="steelblue")

    if has_yri:
        m = evaluate_probe(probe, scaler, mu_target, y_target, yri_split_label)
        all_metrics.append(m)
        scatter_plot(probe, scaler, mu_target, y_target,
                     yri_split_label, out / "scatter_target.png",
                     color="darkorange")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    save_metrics(all_metrics, out / "probe_metrics.json")
    print("\nDone.")


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Probe phenotype prediction from frozen VAE mu vectors."
    )
    p.add_argument("--checkpoint",        type=str, required=True,
                   help="Path to best_model.pt checkpoint")

    g = p.add_argument_group("genotypes")
    g.add_argument("--disc-train-geno",   type=str, required=True)
    g.add_argument("--disc-val-geno",     type=str, required=True)
    g.add_argument("--target-train-geno", type=str, default=None,
                   help="YRI target_train genotypes (falls back to target_held_out if empty)")

    ph = p.add_argument_group("phenotypes")
    ph.add_argument("--disc-train-pheno",   type=str, required=True)
    ph.add_argument("--disc-val-pheno",     type=str, required=True)
    ph.add_argument("--target-train-pheno", type=str, default=None,
                    help="YRI target_train phenotypes (falls back to target_held_out if empty)")

    p.add_argument("--output-dir",  type=str, required=True)
    p.add_argument("--batch-size",  type=int, default=256,
                   help="Batch size for mu extraction (default: 256)")
    p.add_argument("--save-mus",    action="store_true",
                   help="Save raw mu arrays as .npy files in output-dir")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)