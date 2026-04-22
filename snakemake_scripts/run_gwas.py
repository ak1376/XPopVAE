#!/usr/bin/env python3
# snakemake_scripts/run_gwas.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gwas import run_gwas


def main():
    cli = argparse.ArgumentParser(description="Run GWAS (Ridge) for one sim/replicate")
    cli.add_argument(
        "--disc-train-geno", type=Path, required=True, help="discovery_train.npy"
    )
    cli.add_argument(
        "--disc-train-pheno", type=Path, required=True, help="discovery_train_pheno.npy"
    )
    cli.add_argument(
        "--disc-val-geno", type=Path, required=True, help="discovery_validation.npy"
    )
    cli.add_argument(
        "--disc-val-pheno",
        type=Path,
        required=True,
        help="discovery_validation_pheno.npy",
    )
    cli.add_argument(
        "--target-geno", type=Path, required=True, help="target_held_out.npy"
    )
    cli.add_argument(
        "--target-pheno", type=Path, required=True, help="target_held_out_pheno.npy"
    )
    cli.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    cli.add_argument(
        "--summary-out", type=Path, required=True, help="Path to write summary JSON"
    )
    args = cli.parse_args()

    # ── load arrays ───────────────────────────────────────────────────────────
    X_train = np.load(args.disc_train_geno)
    y_train = np.load(args.disc_train_pheno).ravel()
    X_val = np.load(args.disc_val_geno)
    y_val = np.load(args.disc_val_pheno).ravel()
    X_test = np.load(args.target_geno)
    y_test = np.load(args.target_pheno).ravel()

    print(f"[run_gwas] X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"[run_gwas] X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"[run_gwas] X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # ── run ───────────────────────────────────────────────────────────────────
    metrics = run_gwas(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        out_dir=args.out_dir,
    )

    # ── write summary ─────────────────────────────────────────────────────────
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(metrics, indent=2))
    print(f"[run_gwas] Summary written to {args.summary_out}")
    print(
        f"[run_gwas] Done. Val R²={metrics['val_r2']:.4f}  Test R²={metrics['test_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
