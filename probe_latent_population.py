#!/usr/bin/env python3
"""
probe_latent_population.py

Asks: are CEU and YRI separable in the VAE latent space?

Analyses:
  0. Raw genotype PCA      — baseline: does PCA on raw genotypes separate populations?
  1. Latent mu heatmap     — visual inspection of mu vectors, CEU and YRI stacked
  2. Per-dimension means   — mean mu per latent dim for CEU vs YRI
  3. LDA                   — finds the single linear direction maximally separating populations
  4. Linear probe          — logistic regression accuracy, the definitive separability test

Usage
-----
    python probe_latent_population.py \
        --checkpoint  path/to/best_model.pt \
        --ceu-geno    path/to/discovery_train.npy \
        --yri-geno    path/to/target_held_out.npy \
        --output-dir  results/population_probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvVAE


# =============================================================================
# Model loading + mu extraction
# =============================================================================

def load_checkpoint(path: Path, device: torch.device) -> ConvVAE:
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt["vae_config"]
    m    = cfg["model"]
    da   = cfg.get("domain_adaptation", {})
    ph   = cfg.get("phenotype", {})

    model = ConvVAE(
        input_length     = ckpt["input_length"],
        in_channels      = 1,
        hidden_channels  = m["hidden_channels"],
        kernel_size      = int(m["kernel_size"]),
        stride           = int(m["stride"]),
        padding          = int(m["padding"]),
        latent_dim       = int(m["latent_dim"]),
        use_batchnorm    = bool(m.get("use_batchnorm", False)),
        activation       = m.get("activation", "elu"),
        pheno_dim        = 1,
        pheno_hidden_dim = ph.get("pheno_hidden_dim", None),
        use_grl          = bool(da.get("use_grl", False)),
        grl_hidden_dim   = da.get("grl_hidden_dim", None),
        num_domains      = 2,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loaded checkpoint : {path}  (epoch {ckpt.get('epoch', '?')})")
    return model


@torch.no_grad()
def extract_mu(model: ConvVAE, geno: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    from torch.utils.data import DataLoader, TensorDataset
    geno_t = torch.tensor(geno, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(geno_t), batch_size=batch_size, shuffle=False)
    mus = []
    for (batch,) in loader:
        _, mu, _, _, _, _ = model(batch.to(device))
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


# =============================================================================
# Analysis 0: Raw genotype PCA baseline
# =============================================================================

def run_raw_genotype_pca(
    ceu_geno : np.ndarray,
    yri_geno : np.ndarray,
    out_dir  : Path,
) -> dict:
    print("  Running raw genotype PCA (fit on CEU, project YRI)…")

    scaler  = StandardScaler()
    ceu_s   = scaler.fit_transform(ceu_geno.astype(np.float32))
    yri_s   = scaler.transform(yri_geno.astype(np.float32))

    pca     = PCA(n_components=10)
    ceu_pca = pca.fit_transform(ceu_s)
    yri_pca = pca.transform(yri_s)
    ev      = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xi, yi, xlabel, ylabel) in zip(axes, [
        (0, 1, f"PC1 ({ev[0]*100:.2f}% var)", f"PC2 ({ev[1]*100:.2f}% var)"),
        (0, 2, f"PC1 ({ev[0]*100:.2f}% var)", f"PC3 ({ev[2]*100:.2f}% var)"),
    ]):
        ax.scatter(ceu_pca[:, xi], ceu_pca[:, yi],
                   alpha=0.5, s=8, label="CEU", color="steelblue", rasterized=True)
        ax.scatter(yri_pca[:, xi], yri_pca[:, yi],
                   alpha=0.5, s=8, label="YRI", color="darkorange", rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, markerscale=3)
    fig.suptitle("Raw genotype PCA\nfit on CEU, YRI projected", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "raw_genotype_pca.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'raw_genotype_pca.png'}")

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(range(1, len(ev) + 1), ev * 100, color="steelblue", alpha=0.8)
    ax2.set_xlabel("PC")
    ax2.set_ylabel("Variance explained (%)")
    ax2.set_title("Raw genotype PCA scree (top 10 PCs)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "raw_genotype_pca_scree.png", dpi=150)
    plt.close(fig2)

    print("  Running linear probe on raw genotypes…")
    all_s  = np.concatenate([ceu_s, yri_s], axis=0)
    labels = np.array([0] * len(ceu_geno) + [1] * len(yri_geno))
    clf    = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, all_s, labels, cv=cv, scoring="balanced_accuracy")
    mean_acc = float(scores.mean())
    print(f"  Raw genotype linear probe balanced acc: {mean_acc:.4f}")

    np.save(out_dir / "raw_ceu_pca.npy", ceu_pca)
    np.save(out_dir / "raw_yri_pca.npy", yri_pca)

    return {
        "raw_geno_pc1_var":          float(ev[0]),
        "raw_geno_pc2_var":          float(ev[1]),
        "raw_geno_linear_probe_acc": mean_acc,
    }


# =============================================================================
# Analysis 1: Latent mu heatmap + per-dimension mean comparison
# =============================================================================

def run_mu_heatmap(
    ceu_mu  : np.ndarray,
    yri_mu  : np.ndarray,
    out_dir : Path,
    max_individuals : int = 200,
):
    """
    Plot 1: Heatmap of mu vectors, CEU and YRI stacked with a dividing line.
            Rows = individuals (subsampled for visibility), cols = latent dims.
            If the populations look the same, the heatmap will show no banding.

    Plot 2: Per-dimension mean mu for CEU vs YRI, and their difference.
            This directly shows any systematic per-dimension shift between populations.
    """
    print("  Plotting latent mu heatmap…")

    # Subsample for visibility — keep random rows from each population
    rng = np.random.default_rng(42)
    n_ceu = min(max_individuals, len(ceu_mu))
    n_yri = min(max_individuals, len(yri_mu))
    ceu_idx = rng.choice(len(ceu_mu), n_ceu, replace=False)
    yri_idx = rng.choice(len(yri_mu), n_yri, replace=False)

    ceu_sub = ceu_mu[ceu_idx]   # (n_ceu, latent_dim)
    yri_sub = yri_mu[yri_idx]   # (n_yri, latent_dim)

    # Stack CEU on top, YRI on bottom
    stacked = np.concatenate([ceu_sub, yri_sub], axis=0)   # (n_ceu+n_yri, latent_dim)

    # Clip for visual clarity (outliers can dominate colorscale)
    vmax = np.percentile(np.abs(stacked), 98)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        stacked,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    # Dividing line between CEU and YRI
    ax.axhline(n_ceu - 0.5, color="black", linewidth=1.5, linestyle="--")
    ax.text(-5, n_ceu / 2, "CEU", ha="right", va="center", fontsize=9, color="steelblue")
    ax.text(-5, n_ceu + n_yri / 2, "YRI", ha="right", va="center", fontsize=9, color="darkorange")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Individual")
    ax.set_title(f"Latent mu heatmap\nCEU (top {n_ceu}) and YRI (bottom {n_yri}), "
                 f"dashed line = population boundary")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="mu value")
    fig.tight_layout()
    fig.savefig(out_dir / "latent_mu_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'latent_mu_heatmap.png'}")

    # ------------------------------------------------------------------
    # Per-dimension mean comparison
    # ------------------------------------------------------------------
    print("  Plotting per-dimension mean mu…")

    ceu_mean = ceu_mu.mean(axis=0)   # (latent_dim,)
    yri_mean = yri_mu.mean(axis=0)   # (latent_dim,)
    diff     = yri_mean - ceu_mean   # positive = YRI higher

    dims = np.arange(len(ceu_mean))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(dims, ceu_mean, color="steelblue",  alpha=0.8, lw=0.8, label="CEU mean")
    axes[0].plot(dims, yri_mean, color="darkorange",  alpha=0.8, lw=0.8, label="YRI mean")
    axes[0].axhline(0, color="gray", lw=0.5, linestyle="--")
    axes[0].set_ylabel("Mean mu")
    axes[0].set_title("Per-dimension mean mu: CEU vs YRI")
    axes[0].legend(fontsize=9)

    axes[1].bar(dims, diff, color=np.where(diff > 0, "darkorange", "steelblue"),
                alpha=0.6, width=1.0)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Latent dimension")
    axes[1].set_ylabel("YRI mean − CEU mean")
    axes[1].set_title(f"Per-dimension mean shift (YRI − CEU)\n"
                      f"max |shift| = {np.abs(diff).max():.4f}  "
                      f"mean |shift| = {np.abs(diff).mean():.4f}")

    fig.tight_layout()
    fig.savefig(out_dir / "latent_mu_per_dim_means.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'latent_mu_per_dim_means.png'}")

    np.save(out_dir / "ceu_mu_mean_per_dim.npy", ceu_mean)
    np.save(out_dir / "yri_mu_mean_per_dim.npy", yri_mean)
    np.save(out_dir / "mu_mean_diff_yri_minus_ceu.npy", diff)


# =============================================================================
# Analysis 2: LDA projection
# =============================================================================

def run_lda(ceu_mu: np.ndarray, yri_mu: np.ndarray, out_dir: Path) -> dict:
    print("  Running LDA…")
    all_mu = np.concatenate([ceu_mu, yri_mu], axis=0)
    labels = np.array([0] * len(ceu_mu) + [1] * len(yri_mu))

    scaler   = StandardScaler()
    all_mu_s = scaler.fit_transform(all_mu)

    lda        = LinearDiscriminantAnalysis()
    projection = lda.fit_transform(all_mu_s, labels).ravel()

    ceu_proj   = projection[labels == 0]
    yri_proj   = projection[labels == 1]
    separation = abs(ceu_proj.mean() - yri_proj.mean()) / projection.std()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(projection.min(), projection.max(), 60)
    ax.hist(ceu_proj, bins=bins, alpha=0.6, label="CEU", color="steelblue", density=True)
    ax.hist(yri_proj, bins=bins, alpha=0.6, label="YRI", color="darkorange", density=True)
    ax.set_xlabel("LDA projection")
    ax.set_ylabel("Density")
    ax.set_title(f"LDA: max-separation direction\nd′ = {separation:.3f}  "
                 f"(CEU μ={ceu_proj.mean():.2f}, YRI μ={yri_proj.mean():.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "lda_projection.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'lda_projection.png'}  (d′={separation:.3f})")

    np.save(out_dir / "lda_projection.npy", projection)
    np.save(out_dir / "lda_labels.npy",     labels)

    return {"lda_separation_dprime": float(separation),
            "ceu_lda_mean":          float(ceu_proj.mean()),
            "yri_lda_mean":          float(yri_proj.mean())}


# =============================================================================
# Analysis 3: Linear classifier probe
# =============================================================================

def run_linear_probe(ceu_mu: np.ndarray, yri_mu: np.ndarray, out_dir: Path) -> dict:
    print("  Running linear probe (logistic regression, 5-fold CV)…")
    all_mu = np.concatenate([ceu_mu, yri_mu], axis=0)
    labels = np.array([0] * len(ceu_mu) + [1] * len(yri_mu))

    scaler   = StandardScaler()
    all_mu_s = scaler.fit_transform(all_mu)

    clf    = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, all_mu_s, labels, cv=cv, scoring="balanced_accuracy")

    mean_acc = float(scores.mean())
    std_acc  = float(scores.std())
    print(f"  Balanced accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    clf.fit(all_mu_s, labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"], scores,
           color="steelblue", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="chance")
    ax.axhline(mean_acc, color="red", linestyle="-", lw=1.5,
               label=f"mean={mean_acc:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Linear probe: CEU vs YRI separability\n(logistic regression, 5-fold CV)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "linear_probe_cv.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'linear_probe_cv.png'}")

    return {
        "linear_probe_balanced_acc_mean": mean_acc,
        "linear_probe_balanced_acc_std":  std_acc,
        "linear_probe_cv_scores":         scores.tolist(),
    }


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(Path(args.checkpoint), device)

    print("\nLoading genotypes…")
    ceu_geno = np.load(args.ceu_geno)
    yri_geno = np.load(args.yri_geno)
    print(f"  CEU : {ceu_geno.shape}")
    print(f"  YRI : {yri_geno.shape}")

    print("\nExtracting mu vectors…")
    ceu_mu = extract_mu(model, ceu_geno, args.batch_size, device)
    yri_mu = extract_mu(model, yri_geno, args.batch_size, device)
    print(f"  CEU mu : {ceu_mu.shape}")
    print(f"  YRI mu : {yri_mu.shape}")

    if args.save_mus:
        np.save(out / "ceu_mu.npy", ceu_mu)
        np.save(out / "yri_mu.npy", yri_mu)

    metrics = {}

    print("\n[0/3] Raw genotype PCA baseline")
    metrics.update(run_raw_genotype_pca(ceu_geno, yri_geno, out))

    print("\n[1/3] Latent mu heatmap + per-dimension means")
    run_mu_heatmap(ceu_mu, yri_mu, out, max_individuals=args.max_heatmap_individuals)

    print("\n[2/3] LDA")
    metrics.update(run_lda(ceu_mu, yri_mu, out))

    print("\n[3/3] Linear probe")
    metrics.update(run_linear_probe(ceu_mu, yri_mu, out))

    with open(out / "population_probe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Summary ---")
    print(f"  Raw genotype PC1 var          : {metrics['raw_geno_pc1_var']*100:.2f}%")
    print(f"  Raw genotype linear probe acc : {metrics['raw_geno_linear_probe_acc']:.4f}")
    print(f"  LDA d′ (latent)               : {metrics['lda_separation_dprime']:.4f}")
    print(f"  Linear probe acc (latent)     : {metrics['linear_probe_balanced_acc_mean']:.4f}"
          f" ± {metrics['linear_probe_balanced_acc_std']:.4f}")
    print(f"\nAll outputs saved to: {out}")


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Probe CEU/YRI population separability in VAE latent space."
    )
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--ceu-geno",    type=str, required=True,
                   help="CEU genotype matrix .npy  (N, L)")
    p.add_argument("--yri-geno",    type=str, required=True,
                   help="YRI genotype matrix .npy  (N, L)")
    p.add_argument("--output-dir",  type=str, required=True)
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--max-heatmap-individuals", type=int, default=200,
                   help="Max individuals per population to show in heatmap (default 200)")
    p.add_argument("--save-mus",    action="store_true",
                   help="Save raw mu arrays to output-dir")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)