#!/usr/bin/env python3
# snakemake_scripts/run_build_genotypes.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.build_genotypes import (
    set_seed,
    load_trees_sequence,
    build_haploid_matrix,
    filter_multiallelic_and_monomorphic,
    apply_maf_filter_on_discovery,
    diploid_matrix,
    pop_metadata,
    simulate_trait,
    simulate_phenotype,
)

'''
python -u snakemake_scripts/run_build_genotypes.py \
  --tree experiments/OOA/simulations/0/rep0/tree_sequence.trees \
  --outdir experiments/OOA/simulations/0/rep0/processed_data \
  --experiment-config-json config_files/experiment_config_OOA.json
'''


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Simulate phenotypes + build genotype arrays for XPopVAE."
    )
    ap.add_argument("--tree",                   type=Path, required=True)
    ap.add_argument("--outdir",                 type=Path, required=True)
    ap.add_argument("--experiment-config-json", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ── load config ───────────────────────────────────────────────────────────
    cfg = json.loads(Path(args.experiment_config_json).read_text())

    maf_threshold        = float(cfg.get("maf_threshold", 0.0))
    num_causal           = int(cfg["num_causal_variants"])
    h2                   = float(cfg["heritability"])
    distribution         = str(cfg.get("trait_distribution", "normal"))
    alpha                = float(cfg.get("alpha", 0.0))
    seed                 = int(cfg.get("seed", 42))
    disc_train_frac      = float(cfg.get("discovery_train_frac", 0.8))
    target_held_out_frac = float(cfg.get("target_held_out_frac", 0.8))
    discovery_pop        = str(cfg.get("discovery", "CEU"))

    print(f"[run_build_genotypes] config:")
    print(f"  discovery_pop        : {discovery_pop}")
    print(f"  maf_threshold        : {maf_threshold}")
    print(f"  num_causal           : {num_causal}")
    print(f"  h2={h2}  alpha={alpha}")
    print(f"  disc_train_frac      : {disc_train_frac}")
    print(f"  target_held_out_frac : {target_held_out_frac}")
    print(f"  seed                 : {seed}")

    # ── load + filter multiallelic and monomorphic ────────────────────────────
    rng = set_seed(seed)
    ts  = load_trees_sequence(args.tree)

    print(f"\n[run_build_genotypes] loaded ts: {args.tree}")
    print(f"  num_individuals : {ts.num_individuals}")
    print(f"  num_samples     : {ts.num_samples}")
    print(f"  num_sites       : {ts.num_sites}")

    G_hap, biallelic_site_ids = filter_multiallelic_and_monomorphic(
        ts, build_haploid_matrix(ts)
    )
    ind_id_to_pop = pop_metadata(ts)

    # ── simulate trait from ALL biallelic polymorphic sites ───────────────────
    trait_df, causal_site_ids = simulate_trait(
        ts, num_causal, biallelic_site_ids, rng, distribution=distribution,
    )

    # ── split indices ─────────────────────────────────────────────────────────
    train_discovery_inds, validation_discovery_inds = [], []
    train_target_inds,    test_target_inds          = [], []

    for ind_id, pop in ind_id_to_pop.items():
        if pop == discovery_pop:
            (train_discovery_inds if rng.random() < disc_train_frac
             else validation_discovery_inds).append(ind_id)
        else:
            (test_target_inds if rng.random() < target_held_out_frac
             else train_target_inds).append(ind_id)

    train_discovery_inds      = np.array(train_discovery_inds)
    validation_discovery_inds = np.array(validation_discovery_inds)
    train_target_inds         = np.array(train_target_inds)
    test_target_inds          = np.array(test_target_inds)

    print(f"\n[run_build_genotypes] splits:")
    print(f"  train_discovery      : {len(train_discovery_inds)}")
    print(f"  validation_discovery : {len(validation_discovery_inds)}")
    print(f"  train_target         : {len(train_target_inds)}")
    print(f"  test_target          : {len(test_target_inds)}")

    # ── simulate phenotype from ALL biallelic polymorphic sites ───────────────
    (train_discovery_pheno, validation_discovery_pheno,
     train_target_pheno,    test_target_pheno) = simulate_phenotype(
        ts, distribution, causal_site_ids,
        train_discovery_inds, validation_discovery_inds,
        train_target_inds,    test_target_inds,
        alpha=alpha, h2=h2, random_seed=seed,
    )

    # ── build full diploid matrix before MAF filter ───────────────────────────
    diploid = diploid_matrix(G_hap)

    # ── apply MAF filter using discovery train individuals only ───────────────
    maf_mask, biallelic_site_ids, maf_report = apply_maf_filter_on_discovery(
        diploid, biallelic_site_ids, train_discovery_inds,
        maf_threshold, causal_site_ids,
    )

    # apply mask to full diploid
    diploid              = diploid[:, maf_mask]
    variant_positions_bp = np.array(
        [ts.site(s).position for s in biallelic_site_ids], dtype=np.float64
    )

    print(f"  diploid shape after MAF filter : {diploid.shape}")

    # ── slice diploid into splits ─────────────────────────────────────────────
    train_discovery_diploid      = diploid[train_discovery_inds, :].astype(np.float32)
    validation_discovery_diploid = diploid[validation_discovery_inds, :].astype(np.float32)
    train_target_diploid         = diploid[train_target_inds, :].astype(np.float32)
    test_target_diploid          = diploid[test_target_inds, :].astype(np.float32)

    # ── save ──────────────────────────────────────────────────────────────────
    outdir    = Path(args.outdir)
    geno_dir  = outdir / "genotype_matrices"
    pheno_dir = outdir / "phenotypes"
    for d in (outdir, geno_dir, pheno_dir):
        d.mkdir(parents=True, exist_ok=True)

    # genotype matrices
    training_diploid = np.concatenate(
        [train_discovery_diploid, train_target_diploid], axis=0
    )
    np.save(geno_dir / "training.npy",             training_diploid)
    np.save(geno_dir / "discovery_train.npy",      train_discovery_diploid)
    np.save(geno_dir / "discovery_validation.npy", validation_discovery_diploid)
    np.save(geno_dir / "target_train.npy",         train_target_diploid)
    np.save(geno_dir / "target_held_out.npy",      test_target_diploid)

    # phenotype arrays
    def pheno_values(df: pd.DataFrame) -> np.ndarray:
        return df.sort_values("individual_id")["phenotype"].to_numpy().astype(np.float32)

    np.save(pheno_dir / "training_pheno.npy",
            np.concatenate([pheno_values(train_discovery_pheno),
                            pheno_values(train_target_pheno)]))
    np.save(pheno_dir / "discovery_train_pheno.npy",      pheno_values(train_discovery_pheno))
    np.save(pheno_dir / "discovery_validation_pheno.npy", pheno_values(validation_discovery_pheno))
    np.save(pheno_dir / "target_train_pheno.npy",         pheno_values(train_target_pheno))
    np.save(pheno_dir / "target_held_out_pheno.npy",      pheno_values(test_target_pheno))

    # full phenotype table
    full_pheno = pd.concat([
        train_discovery_pheno, validation_discovery_pheno,
        train_target_pheno,    test_target_pheno,
    ]).sort_values("individual_id").reset_index(drop=True)
    pop_id_to_name     = {pop.id: pop.metadata["name"] for pop in ts.populations()}
    ind_id_to_pop_name = {ind.id: pop_id_to_name[ind.population] for ind in ts.individuals()}
    full_pheno["population"] = full_pheno["individual_id"].map(ind_id_to_pop_name)
    full_pheno.to_pickle(outdir / "phenotype.pkl")

    trait_df.to_pickle(outdir / "trait_df.pkl")

    # MAF filter report
    with open(outdir / "maf_filter_report.txt", "w") as f:
        f.write("MAF Filter Report (applied on discovery train only)\n")
        f.write("=" * 40 + "\n")
        for k, v in maf_report.items():
            f.write(f"{k}: {v}\n")

    # split index arrays
    np.save(outdir / "discovery_train_idx.npy",  train_discovery_inds.astype(np.int64))
    np.save(outdir / "discovery_val_idx.npy",    validation_discovery_inds.astype(np.int64))
    np.save(outdir / "target_train_idx.npy",     train_target_inds.astype(np.int64))
    np.save(outdir / "target_held_out_idx.npy",  test_target_inds.astype(np.int64))

    # variant positions and biallelic site ids
    np.save(outdir / "variant_positions_bp.npy", variant_positions_bp)
    np.save(outdir / "biallelic_site_ids.npy",   biallelic_site_ids.astype(np.int64))

    print(f"\n[run_build_genotypes] files written to {outdir}:")
    for split, arr in [
        ("training",             training_diploid),
        ("discovery_train",      train_discovery_diploid),
        ("discovery_validation", validation_discovery_diploid),
        ("target_train",         train_target_diploid),
        ("target_held_out",      test_target_diploid),
    ]:
        print(f"  genotype_matrices/{split}.npy  {arr.shape}")


if __name__ == "__main__":
    main()