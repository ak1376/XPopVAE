#!/usr/bin/env python3
# snakemake_scripts/run_standard_prs.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prs import run_standard_prs


def main():
    cli = argparse.ArgumentParser(description="Run standard (GWAS-beta) PRS for one sim/replicate")
    cli.add_argument("--disc-train-geno",  type=Path, required=True, help="discovery_train.npy")
    cli.add_argument("--disc-train-pheno", type=Path, required=True, help="discovery_train_pheno.npy")
    cli.add_argument("--disc-val-geno",    type=Path, required=True, help="discovery_validation.npy")
    cli.add_argument("--disc-val-pheno",   type=Path, required=True, help="discovery_validation_pheno.npy")
    cli.add_argument("--target-geno",      type=Path, required=True, help="target_held_out.npy")
    cli.add_argument("--target-pheno",     type=Path, required=True, help="target_held_out_pheno.npy")
    cli.add_argument("--beta-panel",       type=Path, required=True, help="beta_panel.npy from GWAS step")
    cli.add_argument("--out-dir",          type=Path, required=True, help="Output directory")
    cli.add_argument("--summary-out",      type=Path, required=True, help="Path to write summary JSON")
    args = cli.parse_args()

    X_train = np.load(args.disc_train_geno).astype(np.float32)
    y_train = np.load(args.disc_train_pheno).ravel().astype(np.float32)
    X_val   = np.load(args.disc_val_geno).astype(np.float32)
    y_val   = np.load(args.disc_val_pheno).ravel().astype(np.float32)
    X_test  = np.load(args.target_geno).astype(np.float32)
    y_test  = np.load(args.target_pheno).ravel().astype(np.float32)
    beta    = np.load(args.beta_panel)

    print(f"[run_standard_prs] X_train: {X_train.shape}  beta: {beta.shape}")
    print(f"[run_standard_prs] X_val:   {X_val.shape}")
    print(f"[run_standard_prs] X_test:  {X_test.shape}")

    metrics = run_standard_prs(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        beta,
        out_dir=args.out_dir,
    )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(metrics, indent=2))
    print(f"[run_standard_prs] Summary written to {args.summary_out}")
    print(f"[run_standard_prs] Done. Val R²={metrics['val_r2']:.4f}  Test R²={metrics['test_r2']:.4f}")


if __name__ == "__main__":
    main()
