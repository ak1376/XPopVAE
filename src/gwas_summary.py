#!/usr/bin/env python3
# src/gwas_summary.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Data loading
# =============================================================================


def load_results(
    proc_basedir: Path,
    sim_numbers: List[str],
    replicates: List[str],
) -> pd.DataFrame:
    """
    Walk every (sim_number, replicate) combination and collect:
      - T_OOA value from sampled_params.pkl
      - val_r2 and test_r2 from gwas/gwas_summary.json

    Returns a tidy DataFrame with columns:
        sim_number, replicate, T_OOA, val_r2, test_r2
    """
    rows = []
    for sim in sim_numbers:
        for rep in replicates:
            rep_dir = proc_basedir / sim / f"rep{rep}"

            params_path = (
                rep_dir
                / "../../../simulations"
                / sim
                / f"rep{rep}"
                / "sampled_params.pkl"
            )
            gwas_path = rep_dir / "gwas" / "gwas_summary.json"

            # sampled_params.pkl lives in the simulations tree, not processed_data
            # we accept either location for flexibility
            sim_params_path = (
                rep_dir.parent.parent.parent
                / "simulations"
                / sim
                / f"rep{rep}"
                / "sampled_params.pkl"
            )

            if not gwas_path.exists():
                print(f"[gwas_summary] Missing GWAS results: {gwas_path} — skipping")
                continue
            if not sim_params_path.exists():
                print(
                    f"[gwas_summary] Missing sampled params: {sim_params_path} — skipping"
                )
                continue

            with open(sim_params_path, "rb") as f:
                params = pickle.load(f)

            metrics = json.loads(gwas_path.read_text())

            rows.append(
                {
                    "sim_number": int(sim),
                    "replicate": int(rep),
                    "T_OOA": float(params["T_OOA"]),
                    "val_r2": float(metrics["val_r2"]),
                    "test_r2": float(metrics["test_r2"]),
                }
            )

    if not rows:
        raise RuntimeError(
            "No results found — check that GWAS has been run for all sim/rep combinations."
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(["T_OOA", "sim_number", "replicate"])
        .reset_index(drop=True)
    )
    print(
        f"[gwas_summary] Loaded {len(df)} rows across {df['T_OOA'].nunique()} unique T_OOA values"
    )
    return df


# =============================================================================
# Plotting
# =============================================================================


def _plot_r2_vs_tooa(
    df: pd.DataFrame,
    r2_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Core plotting logic.
    - Single replicate  → scatter only
    - Multiple replicates → scatter (light) + mean line + std error bars
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    n_reps = df["replicate"].nunique()
    grouped = df.groupby("T_OOA")[r2_col]

    if n_reps == 1:
        # Just scatter individual points
        ax.scatter(
            df["T_OOA"], df[r2_col], color="steelblue", s=60, zorder=3, label="R²"
        )
    else:
        means = grouped.mean()
        stds = grouped.std()
        tooa = means.index.values

        # Scatter individual replicates (light)
        ax.scatter(
            df["T_OOA"],
            df[r2_col],
            color="steelblue",
            alpha=0.35,
            s=30,
            zorder=2,
            label="Individual replicates",
        )
        # Mean line
        ax.plot(
            tooa, means.values, color="steelblue", linewidth=2, zorder=3, label="Mean"
        )
        # ± 1 std band
        ax.fill_between(
            tooa,
            means.values - stds.values,
            means.values + stds.values,
            color="steelblue",
            alpha=0.15,
            label="± 1 SD",
        )
        ax.legend(fontsize=9)

    ax.set_xlabel("T_OOA (generations)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gwas_summary] Saved plot: {out_path}")


def plot_discovery_r2(df: pd.DataFrame, out_dir: Path) -> None:
    _plot_r2_vs_tooa(
        df,
        r2_col="val_r2",
        ylabel="Discovery R² (validation)",
        title="GWAS Discovery R² vs T_OOA",
        out_path=out_dir / "discovery_r2_vs_tooa.png",
    )


def plot_target_r2(df: pd.DataFrame, out_dir: Path) -> None:
    _plot_r2_vs_tooa(
        df,
        r2_col="test_r2",
        ylabel="Target R² (held-out)",
        title="GWAS Target R² vs T_OOA",
        out_path=out_dir / "target_r2_vs_tooa.png",
    )


def plot_af_diff_vs_r2_gap(
    df: pd.DataFrame,
    proc_basedir: Path,
    out_dir: Path,
) -> None:
    """
    For each row in df (a single sim's replicates), compute
    mean |V_CEU - V_YRI| at causal SNPs, where V = effect² * p(1-p),
    and plot against R² gap (val_r2 - test_r2).
    Called once per sim via the diagnose_replicates wildcard; out_dir is the
    per-sim subdirectory.
    """
    from scipy import stats

    out_dir.mkdir(parents=True, exist_ok=True)
    sim_number = int(df["sim_number"].iloc[0])

    rows = []
    for _, row in df.iterrows():
        sim, rep = str(int(row["sim_number"])), int(row["replicate"])
        proc = proc_basedir / sim / f"rep{rep}"

        trait_df = pd.read_pickle(proc / "trait_df.pkl")
        biallelic_ids = np.load(proc / "biallelic_site_ids.npy")
        disc_train = np.load(proc / "genotype_matrices/discovery_train.npy")
        target_ho = np.load(proc / "genotype_matrices/target_held_out.npy")

        causal_set = set(trait_df["site_id"].unique())
        idx = np.array([i for i, s in enumerate(biallelic_ids) if s in causal_set])

        # align effect sizes to the same order as idx
        effect_by_site = trait_df.groupby("site_id")["effect_size"].mean()
        effects = np.array([effect_by_site[s] for s in biallelic_ids[idx]])

        p_ceu = disc_train.mean(axis=0)[idx] / 2.0
        p_yri = target_ho.mean(axis=0)[idx] / 2.0

        v_ceu = effects**2 * p_ceu * (1.0 - p_ceu)
        v_yri = effects**2 * p_yri * (1.0 - p_yri)
        mean_v_diff = float(np.abs(v_ceu - v_yri).mean())

        rows.append(
            {
                "label": f"rep{rep}",
                "r2_disc": row["val_r2"],
                "r2_target": row["test_r2"],
                "r2_gap": row["val_r2"] - row["test_r2"],
                "mean_v_diff_causal": mean_v_diff,
            }
        )

    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(out_dir / "replicate_stats.csv", index=False)

    x = diag_df["mean_v_diff_causal"].values
    y = diag_df["r2_gap"].values
    mask = ~(np.isnan(x) | np.isnan(y))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        x[mask],
        y[mask],
        s=80,
        color="#E53935",
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )
    for _, r in diag_df[mask].iterrows():
        ax.annotate(
            r["label"],
            (r["mean_v_diff_causal"], r["r2_gap"]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=9,
        )

    if mask.sum() > 2:
        r_val, p_val = stats.pearsonr(x[mask], y[mask])
        m, b = np.polyfit(x[mask], y[mask], 1)
        xline = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xline, m * xline + b, "k--", lw=1, alpha=0.5)
        ax.set_title(
            f"Variance contribution gap at causal SNPs vs R² gap  (Sim {sim_number})\n"
            f"Pearson r={r_val:.3f}, p={p_val:.3f}  (n={mask.sum()} replicates)",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("mean( |V_CEU − V_YRI| ) at causal SNPs,  V = β² p(1−p)", fontsize=11)
    ax.set_ylabel("R² gap  (R²_CEU − R²_YRI)", fontsize=11)
    fig.suptitle(
        f"Do variance contributions at causal SNPs\n"
        f"explain the CEU→YRI portability gap?  (Sim {sim_number})",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "af_diff_vs_r2_gap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gwas_summary] Saved plot: {out_path}")


# =============================================================================
# Main entry point
# =============================================================================


def run_gwas_summary(
    proc_basedir: Path,
    sim_numbers: List[str],
    replicates: List[str],
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(proc_basedir, sim_numbers, replicates)

    plot_discovery_r2(df, out_dir)
    plot_target_r2(df, out_dir)

    csv_path = out_dir / "gwas_summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[gwas_summary] Summary table saved: {csv_path}")

    return df
