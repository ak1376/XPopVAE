#!/usr/bin/env python3
# snakemake_scripts/run_gwas_summary.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gwas_summary import plot_af_diff_vs_r2_gap, run_gwas_summary


def main():
    cli = argparse.ArgumentParser(
        description="Summarise GWAS R² results across all simulations"
    )
    cli.add_argument(
        "--proc-basedir",
        type=Path,
        required=True,
        help="Base processed_data directory, e.g. experiments/OOA/processed_data",
    )
    cli.add_argument(
        "--sim-numbers",
        nargs="+",
        required=True,
        help="List of simulation numbers e.g. 0 1 2 ... 9",
    )
    cli.add_argument(
        "--replicates",
        nargs="+",
        required=True,
        help="List of replicate indices e.g. 0",
    )
    cli.add_argument(
        "--out-dir", type=Path, required=True, help="Output directory for plots and CSV"
    )
    cli.add_argument(
        "--af-diff-plot",
        action="store_true",
        help="Produce the per-sim AF-diff vs R² gap plot (used by diagnose_replicates)",
    )
    args = cli.parse_args()

    df = run_gwas_summary(
        proc_basedir=args.proc_basedir,
        sim_numbers=args.sim_numbers,
        replicates=args.replicates,
        out_dir=args.out_dir,
    )

    if args.af_diff_plot:
        plot_af_diff_vs_r2_gap(df, args.proc_basedir, args.out_dir)

    print("[run_gwas_summary] Done.")


if __name__ == "__main__":
    main()
