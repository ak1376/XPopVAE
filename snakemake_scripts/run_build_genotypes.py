#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.build_genotypes import BuildGenotypesArgs, build_genotypes_for_vae

"""
python -u snakemake_scripts/run_build_genotypes.py \
  --tree experiments/IM_symmetric/simulations/0/rep0/tree_sequence.trees \
  --phenotype experiments/IM_symmetric/simulations/0/rep0/phenotype.pkl \
  --outdir genotypes/0/rep0 \
  --experiment-config-json config_files/experiment_config_IM_symmetric.json \
  --maf-threshold 0.05 \
  --subset-mode random \
  --subset-seed 0 \
  --discovery-val-frac 0.2 \
  --target-test-frac 0.2 \
  --split-seed 0 \
  --subset-snps 5000
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Snakemake wrapper: build genotype arrays + discovery train/val + target train/test split."
    )

    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--phenotype", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config with maf_threshold under data.maf_threshold",
    )
    ap.add_argument("--maf-threshold", type=float, default=None)

    ap.add_argument(
        "--experiment-config-json",
        type=Path,
        default=None,
        help="Optional experiment config JSON; reads top-level key 'discovery' unless --discovery-pop is provided.",
    )

    # subsetting
    ap.add_argument("--subset-snps", type=int, default=5000)
    ap.add_argument("--subset-bp", type=float, default=None)
    ap.add_argument(
        "--subset-mode",
        type=str,
        default="first",
        choices=["first", "middle", "random"],
    )
    ap.add_argument("--subset-seed", type=int, default=0)

    # splits
    ap.add_argument(
        "--discovery-val-frac",
        type=float,
        default=0.2,
        help="Fraction of discovery population held out for validation.",
    )
    ap.add_argument(
        "--target-test-frac",
        type=float,
        default=0.2,
        help="Fraction of target population held out for test.",
    )
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument(
        "--discovery-pop",
        type=str,
        default=None,
        help="Override discovery population label. If omitted, read from --experiment-config-json; else default CEU.",
    )

    ap.add_argument("--norm-eps", type=float, default=1e-6)
    ap.add_argument("--norm-clip-std-min", type=float, default=1e-3)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    a = BuildGenotypesArgs(
        tree=args.tree,
        phenotype=args.phenotype,
        outdir=args.outdir,
        config=args.config,
        maf_threshold=args.maf_threshold,
        experiment_config_json=args.experiment_config_json,
        subset_snps=int(args.subset_snps),
        subset_bp=args.subset_bp,
        subset_mode=str(args.subset_mode),
        subset_seed=int(args.subset_seed),
        discovery_val_frac=float(args.discovery_val_frac),
        target_test_frac=float(args.target_test_frac),
        split_seed=int(args.split_seed),
        discovery_pop=args.discovery_pop,
        norm_eps=float(args.norm_eps),
        norm_clip_std_min=float(args.norm_clip_std_min),
    )

    summary = build_genotypes_for_vae(a)
    print("[build_genotypes_for_vae wrapper] summary:", summary)


if __name__ == "__main__":
    main()