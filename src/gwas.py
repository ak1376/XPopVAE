#!/usr/bin/env python3
# src/gwas.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from src.utils import calculate_normalization_statistics, normalize


"""
Traditional GWAS via marginal OLS regression.

For each SNP j, fits:
    y = alpha_j + beta_j * g_j + epsilon

Outputs:
    beta_panel.npy   : per-SNP effect sizes (p,)
    t_map.npy        : per-SNP t-statistics (p,)
    p_map.npy        : per-SNP p-values     (p,)
    manhattan.png    : Manhattan plot
"""



# =============================================================================
# Marginal OLS
# =============================================================================

def marginal_OLS(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized marginal OLS across all SNPs simultaneously.

    X : (n, p) normalized genotype matrix — already centered and scaled
    y : (n,)   normalized phenotype vector — already centered and scaled

    Returns
    -------
    beta : (p,) effect sizes
    t    : (p,) t-statistics
    p    : (p,) two-sided p-values
    """
    n, p = X.shape

    # beta_j = (X_j . y) / ||X_j||^2
    numerator   = X.T @ y                        # (p,)
    denominator = (X ** 2).sum(axis=0)           # (p,)
    beta        = numerator / denominator        # (p,)

    # residuals per SNP: y_hat_j = X_j * beta_j
    y_hat   = X * beta[np.newaxis, :]            # (n, p) broadcast
    resid   = y[:, np.newaxis] - y_hat           # (n, p)
    rss     = (resid ** 2).sum(axis=0)           # (p,)

    # residual variance: s^2 = RSS / (n - 2)  [2 params: intercept + slope]
    s2      = rss / (n - 2)                      # (p,)

    # SE(beta_j) = sqrt(s^2 / ||X_j||^2)
    se      = np.sqrt(s2 / denominator)          # (p,)

    # t-statistic
    t       = beta / se                          # (p,)

    # two-sided p-value from t distribution with (n-2) df
    pval    = 2 * stats.t.sf(np.abs(t), df=n - 2)  # (p,)

    return beta, t, pval


# =============================================================================
# Saving
# =============================================================================

def save_beta_panel(beta: np.ndarray, path: Path) -> None:
    np.save(path, beta)


def save_t_map(t: np.ndarray, path: Path) -> None:
    np.save(path, t)


def save_p_map(pval: np.ndarray, path: Path) -> None:
    np.save(path, pval)


# =============================================================================
# Manhattan plot
# =============================================================================

def plot_manhattan(
    pval: np.ndarray,
    variant_positions_bp: np.ndarray | None,
    out_path: Path,
) -> None:
    """
    Simple Manhattan plot. If variant_positions_bp is provided, uses physical
    positions on the x-axis; otherwise uses SNP index.
    """
    bonferroni_threshold = 0.05 / len(pval)

    log_p = -np.log10(np.clip(pval, 1e-300, 1.0))
    x     = variant_positions_bp if variant_positions_bp is not None else np.arange(len(pval))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(x, log_p, s=2, alpha=0.5, color="steelblue", rasterized=True)

    sig_line = -np.log10(bonferroni_threshold)
    ax.axhline(sig_line, color="red", linewidth=0.8, linestyle="--",
               label=f"Bonferroni p={bonferroni_threshold:.2e} (n={len(pval)} SNPs)")

    ax.set_xlabel("Position (bp)" if variant_positions_bp is not None else "SNP index")
    ax.set_ylabel("$-\\log_{10}(p)$")
    ax.set_title("Manhattan Plot")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[GWAS] Manhattan plot saved: {out_path}")


# =============================================================================
# High-level runner (importable by snakemake scripts)
# =============================================================================

def run_gwas(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
    variant_positions: np.ndarray | None = None,
) -> dict:
    """
    Run full GWAS pipeline and return PRS R² on val and test sets.

    Fits normalization on training data only, runs marginal OLS, saves
    outputs, and scores PRS via Pearson R².
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geno_mean, geno_std = calculate_normalization_statistics(X_train)
    X_train_n = normalize(X_train, geno_mean, geno_std)
    X_val_n   = normalize(X_val,   geno_mean, geno_std)
    X_test_n  = normalize(X_test,  geno_mean, geno_std)

    pheno_mean, pheno_std = calculate_normalization_statistics(y_train)
    y_train_n = normalize(y_train, pheno_mean, pheno_std)
    y_val_n   = normalize(y_val,   pheno_mean, pheno_std)
    y_test_n  = normalize(y_test,  pheno_mean, pheno_std)

    beta, t, pval = marginal_OLS(X_train_n, y_train_n)
    print(f"[GWAS] beta range: [{beta.min():.4f}, {beta.max():.4f}]")
    print(f"[GWAS] significant SNPs (p<5e-8): {(pval < 5e-8).sum()}")

    save_beta_panel(beta, out_dir / "beta_panel.npy")
    save_t_map(t,         out_dir / "t_map.npy")
    save_p_map(pval,      out_dir / "p_map.npy")
    plot_manhattan(pval, variant_positions, out_dir / "manhattan.png")

    prs_val  = X_val_n  @ beta
    prs_test = X_test_n @ beta

    val_r2  = float(stats.pearsonr(prs_val,  y_val_n)[0]  ** 2)
    test_r2 = float(stats.pearsonr(prs_test, y_test_n)[0] ** 2)

    return {"val_r2": val_r2, "test_r2": test_r2}

