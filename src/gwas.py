#!/usr/bin/env python3
# src/gwas.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Helpers
# =============================================================================


def log(msg: str) -> None:
    print(f"[GWAS] {msg}", flush=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return a dict of scalar metrics for one split."""
    return {"r2": r2_score(y_true, y_pred)}


def report(
    model_name: str, val_metrics: Dict, test_metrics: Dict, extra: str = ""
) -> None:
    log(f"{model_name}{' | ' + extra if extra else ''}")
    log(f"  Val  R²: {val_metrics['r2']:.4f}")
    log(f"  Test R²: {test_metrics['r2']:.4f}")


# =============================================================================
# Preprocessing
# =============================================================================


def standardization(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using mean and std of the training set only."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


# =============================================================================
# Plotting
# =============================================================================


def plot_scatter(
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    """Save a 1×2 scatter plot (val | test) for a given model."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y_true, y_pred, split in zip(
        axes,
        [y_val, y_test],
        [y_val_pred, y_test_pred],
        ["Val", "Test"],
    ):
        r2 = r2_score(y_true, y_pred)
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolors="none")
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("True phenotype")
        ax.set_ylabel("Predicted phenotype")
        ax.set_title(f"{model_name} — {split}  ($R^2$={r2:.4f})")

    fig.tight_layout()
    fname = (
        model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        + "_scatter.png"
    )
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    log(f"Scatter plot saved: {out_dir / fname}")


# =============================================================================
# Models
# =============================================================================


def ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> Dict[str, float]:
    """
    Tune Ridge alpha on validation R², refit, evaluate on val + test,
    save scatter plot, and return metrics dict.
    """
    log("Fitting Ridge Regression...")
    best_r2, best_alpha = -np.inf, 1.0

    for alpha in np.logspace(-3, 6, 80):
        r2 = r2_score(y_val, Ridge(alpha=alpha).fit(X_train, y_train).predict(X_val))
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    ridge = Ridge(alpha=best_alpha).fit(X_train, y_train)
    y_val_pred = ridge.predict(X_val)
    y_test_pred = ridge.predict(X_test)

    val_metrics = evaluate(y_val, y_val_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    report(
        "Ridge Regression",
        val_metrics,
        test_metrics,
        extra=f"best alpha={best_alpha:.4g}",
    )
    plot_scatter(y_val, y_val_pred, y_test, y_test_pred, "Ridge Regression", out_dir)

    return {
        "val_r2": val_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "best_alpha": best_alpha,
    }


# =============================================================================
# Main entry point
# =============================================================================


def run_gwas(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> Dict[str, float]:
    """
    Full GWAS pipeline:
      1. Standardize genotype matrices
      2. Run Ridge regression
      3. Save scatter plots
      4. Return metrics dict
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_s, X_val_s, X_test_s = standardization(X_train, X_val, X_test)

    metrics = ridge_regression(
        X_train_s,
        y_train,
        X_val_s,
        y_val,
        X_test_s,
        y_test,
        out_dir,
    )

    return metrics
