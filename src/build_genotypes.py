#!/usr/bin/env python3
# src/build_genotypes.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tskit
import yaml


# =============================================================================
# Public API
# =============================================================================

@dataclass(frozen=True)
class BuildGenotypesArgs:
    tree: Path
    phenotype: Path
    outdir: Path

    # Optional YAML config (maf_threshold under cfg["data"]["maf_threshold"])
    config: Optional[Path] = None
    maf_threshold: Optional[float] = None

    # Optional experiment config JSON (reads "discovery" + "maf_threshold")
    experiment_config_json: Optional[Path] = None

    # SNP subsetting
    subset_snps: int = 5000
    subset_bp: Optional[float] = None
    subset_mode: str = "first"   # "first" | "middle" | "random"
    subset_seed: int = 0

    # split
    disc_train_frac: float = 0.8   # fraction of discovery pop -> discovery_train
    target_held_out_frac: float = 0.8  # fraction of target pop -> target_held_out
    split_seed: int = 0
    discovery_pop: Optional[str] = None  # if None, read from experiment_config_json, else default "CEU"

    # Always normalize by HWE (af_residual)
    norm_eps: float = 1e-6
    norm_clip_std_min: float = 1e-3


def build_genotypes_for_vae(a: BuildGenotypesArgs) -> Dict[str, Any]:
    """
    End-to-end builder.

    Output directory layout
    -----------------------
    processed_data/
      genotype_matrices/
        training.npy                  # discovery_train + target_train stacked
        discovery_train.npy           # discovery pop, 80 %
        target_train.npy              # target pop,    20 %
        validation.npy                # discovery_validation only
        discovery_validation.npy      # discovery pop, 20 %
        held_out.npy                  # target_held_out only
        target_held_out.npy           # target pop,    80 %

      phenotypes/
        training_pheno.npy
        discovery_train_pheno.npy
        target_train_pheno.npy
        validation_pheno.npy
        discovery_validation_pheno.npy
        held_out_pheno.npy
        target_held_out_pheno.npy

      # auxiliary (unchanged filenames)
      meta.pkl
      hap_meta.pkl
      all_individuals.npy
      hap1.npy / hap2.npy
      snp_index.npy
      variant_positions_bp.npy
      variant_site_ids.npy
      ts_individual_ids.npy
      *_idx.npy          (row indices into all_individuals for each split)
      site_filter_report.txt
      genotype_site_stats.txt
      train_mono_filter_report.txt
    """
    outdir = Path(a.outdir)
    geno_dir = outdir / "genotype_matrices"
    pheno_dir = outdir / "phenotypes"
    for d in (outdir, geno_dir, pheno_dir):
        d.mkdir(parents=True, exist_ok=True)

    maf_from_yaml = _load_maf_from_config(a.config)
    maf_from_exp_json = _load_maf_from_experiment_json(a.experiment_config_json)

    maf_threshold = (
        a.maf_threshold
        if a.maf_threshold is not None
        else (
            maf_from_exp_json
            if maf_from_exp_json is not None
            else (maf_from_yaml if maf_from_yaml is not None else 0.0)
        )
    )

    discovery_pop = _resolve_discovery_pop(a.discovery_pop, a.experiment_config_json)

    print(
        "[build_genotypes_for_vae] "
        f"maf_threshold={maf_threshold} "
        "(biallelic/nonmissing/global-nonmonomorphic filter first; "
        "MAF filter applied on discovery_train only)"
    )
    print(f"[build_genotypes_for_vae] discovery_pop={discovery_pop}")
    print(f"[build_genotypes_for_vae] loading ts: {a.tree}")

    ts = tskit.load(str(a.tree))

    # -------------------------------------------------------------------------
    # Basic TS site stats (raw)
    # -------------------------------------------------------------------------
    stats = compute_site_stats(ts)
    (outdir / "genotype_site_stats.txt").write_text(_format_site_stats(a.tree, stats))

    # -------------------------------------------------------------------------
    # Extract/filter to:
    #   - biallelic
    #   - non-missing
    #   - globally non-monomorphic
    # -------------------------------------------------------------------------
    hap1, hap2, G, kept_ind_ids, filt, kept_positions_bp, kept_site_ids = _extract_haps_and_diploid(ts)
    (outdir / "site_filter_report.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in filt.items()]) + "\n"
    )

    # -------------------------------------------------------------------------
    # Load + align phenotype/meta
    # -------------------------------------------------------------------------
    pheno = pd.read_pickle(a.phenotype)
    pheno = _align_pheno_to_kept_inds(ts, pheno, kept_ind_ids, G.shape[0])

    for col in ["individual_id", "population", "phenotype"]:
        if col not in pheno.columns:
            raise ValueError(
                f"phenotype.pkl must include column '{col}'. Columns found: {list(pheno.columns)}"
            )

    meta = pheno[["individual_id", "population", "phenotype"]].copy()
    meta.to_pickle(outdir / "meta.pkl")

    hap_meta = _build_hap_meta(meta, has_hap2=(hap2 is not None))
    hap_meta.to_pickle(outdir / "hap_meta.pkl")

    # -------------------------------------------------------------------------
    # Split:
    #   discovery pop  -> 80% discovery_train  | 20% discovery_validation
    #   target pop     -> 20% target_train      | 80% target_held_out
    # -------------------------------------------------------------------------
    disc_train_idx, disc_val_idx, target_train_idx, target_held_out_idx = (
        _make_splits(
            meta=meta,
            discovery_pop=discovery_pop,
            disc_train_frac=float(a.disc_train_frac),
            target_held_out_frac=float(a.target_held_out_frac),
            seed=int(a.split_seed),
        )
    )

    # Save indices (row indices into all_individuals / G)
    np.save(outdir / "discovery_train_idx.npy",      disc_train_idx.astype(np.int64))
    np.save(outdir / "discovery_val_idx.npy",         disc_val_idx.astype(np.int64))
    np.save(outdir / "target_train_idx.npy",          target_train_idx.astype(np.int64))
    np.save(outdir / "target_held_out_idx.npy",       target_held_out_idx.astype(np.int64))

    # -------------------------------------------------------------------------
    # Subset SNP window (AFTER global structural filtering, BEFORE MAF)
    # -------------------------------------------------------------------------
    num_sites = G.shape[1]
    start, end = _choose_subset_indices(
        kept_positions_bp=kept_positions_bp,
        num_sites=num_sites,
        subset_snps=a.subset_snps,
        subset_bp=a.subset_bp,
        subset_mode=a.subset_mode,
        subset_seed=a.subset_seed,
    )
    if end <= start:
        raise RuntimeError(f"Subset window invalid: start={start}, end={end}, num_sites={num_sites}")

    G_subset    = G[:, start:end].astype(np.float32)
    hap1_subset = hap1[:, start:end].astype(np.float32)
    hap2_subset = None if hap2 is None else hap2[:, start:end].astype(np.float32)

    kept_positions_subset = kept_positions_bp[start:end].astype(np.float64)
    kept_site_ids_subset  = kept_site_ids[start:end].astype(np.int32)
    snp_idx               = np.arange(start, end, dtype=np.int64)

    # -------------------------------------------------------------------------
    # Discovery-train MAF filtering (defines the feature set)
    # -------------------------------------------------------------------------
    num_snps_before = int(G_subset.shape[1])

    G_disc_train_pre = G_subset[disc_train_idx]
    p_train  = G_disc_train_pre.mean(axis=0, dtype=np.float64) / 2.0
    maf_train = np.minimum(p_train, 1.0 - p_train)

    keep_cols = maf_train > 0.0
    num_mono_removed = int((maf_train == 0.0).sum())

    num_maf_removed = 0
    min_ac_implied  = None
    if maf_threshold is not None and maf_threshold > 0.0:
        n_disc_train_haps = 2 * int(G_disc_train_pre.shape[0])
        min_ac_implied    = int(math.ceil(float(maf_threshold) * n_disc_train_haps))
        before     = int(keep_cols.sum())
        keep_cols &= (maf_train >= float(maf_threshold))
        after      = int(keep_cols.sum())
        num_maf_removed = before - after

    if int(keep_cols.sum()) == 0:
        raise RuntimeError(
            "All SNPs were removed by discovery_train filtering. "
            f"subset_start={start}, subset_end={end}, maf_threshold={maf_threshold}"
        )

    G_subset    = G_subset[:, keep_cols]
    hap1_subset = hap1_subset[:, keep_cols]
    if hap2_subset is not None:
        hap2_subset = hap2_subset[:, keep_cols]
    kept_positions_subset = kept_positions_subset[keep_cols]
    kept_site_ids_subset  = kept_site_ids_subset[keep_cols]
    snp_idx               = snp_idx[keep_cols]
    num_snps_after        = int(G_subset.shape[1])

    (outdir / "train_mono_filter_report.txt").write_text(
        f"num_snps_before={num_snps_before}\n"
        f"num_snps_after={num_snps_after}\n"
        f"num_dropped={num_snps_before - num_snps_after}\n"
        f"num_monomorphic_removed={num_mono_removed}\n"
        f"num_maf_removed={num_maf_removed}\n"
        f"maf_threshold={maf_threshold}\n"
        f"num_discovery_train_individuals={G_disc_train_pre.shape[0]}\n"
        f"num_discovery_train_haplotypes={2 * G_disc_train_pre.shape[0]}\n"
        f"min_allele_count_implied={min_ac_implied}\n"
    )

    # -------------------------------------------------------------------------
    # Save auxiliary arrays (outdir root, unchanged filenames)
    # -------------------------------------------------------------------------
    np.save(outdir / "all_individuals.npy",      G_subset)
    np.save(outdir / "hap1.npy",                 hap1_subset)
    np.save(
        outdir / "hap2.npy",
        hap2_subset if hap2_subset is not None else np.zeros_like(hap1_subset, dtype=np.float32),
    )
    np.save(outdir / "snp_index.npy",            snp_idx)
    np.save(outdir / "variant_positions_bp.npy", kept_positions_subset)
    np.save(outdir / "variant_site_ids.npy",     kept_site_ids_subset)
    np.save(outdir / "ts_individual_ids.npy",    kept_ind_ids.astype(np.int64))

    # -------------------------------------------------------------------------
    # Build per-split genotype slices
    # -------------------------------------------------------------------------
    G_disc_train      = G_subset[disc_train_idx]
    G_disc_val        = G_subset[disc_val_idx]
    G_target_train    = G_subset[target_train_idx]
    G_target_held_out = G_subset[target_held_out_idx]

    # Composite arrays
    G_training   = np.concatenate([G_disc_train, G_target_train], axis=0).astype(np.float32)
    G_validation = G_disc_val.astype(np.float32)
    G_held_out   = G_target_held_out.astype(np.float32)

    # genotype_matrices/
    np.save(geno_dir / "training.npy",              G_training)
    np.save(geno_dir / "discovery_train.npy",       G_disc_train.astype(np.float32))
    np.save(geno_dir / "target_train.npy",          G_target_train.astype(np.float32))
    np.save(geno_dir / "validation.npy",            G_validation)
    np.save(geno_dir / "discovery_validation.npy",  G_disc_val.astype(np.float32))
    np.save(geno_dir / "held_out.npy",              G_held_out)
    np.save(geno_dir / "target_held_out.npy",       G_target_held_out.astype(np.float32))

    # -------------------------------------------------------------------------
    # Build per-split phenotype arrays
    # -------------------------------------------------------------------------
    y_all             = meta["phenotype"].to_numpy()
    y_disc_train      = y_all[disc_train_idx]
    y_disc_val        = y_all[disc_val_idx]
    y_target_train    = y_all[target_train_idx]
    y_target_held_out = y_all[target_held_out_idx]

    y_training   = np.concatenate([y_disc_train, y_target_train])
    y_validation = y_disc_val
    y_held_out   = y_target_held_out

    # phenotypes/
    np.save(pheno_dir / "training_pheno.npy",              y_training)
    np.save(pheno_dir / "discovery_train_pheno.npy",       y_disc_train)
    np.save(pheno_dir / "target_train_pheno.npy",          y_target_train)
    np.save(pheno_dir / "validation_pheno.npy",            y_validation)
    np.save(pheno_dir / "discovery_validation_pheno.npy",  y_disc_val)
    np.save(pheno_dir / "held_out_pheno.npy",              y_held_out)
    np.save(pheno_dir / "target_held_out_pheno.npy",       y_target_held_out)

    summary = {
        "outdir": str(outdir),
        "maf_threshold": float(maf_threshold),
        "subset_start": int(start),
        "subset_end": int(end),
        "num_inds": int(G_subset.shape[0]),
        "num_snps": int(G_subset.shape[1]),
        "disc_train_frac": float(a.disc_train_frac),
        "target_held_out_frac": float(a.target_held_out_frac),
        "split_seed": int(a.split_seed),
        "discovery_pop": str(discovery_pop),
        "n_discovery_train": int(disc_train_idx.size),
        "n_discovery_validation": int(disc_val_idx.size),
        "n_target_train": int(target_train_idx.size),
        "n_target_held_out": int(target_held_out_idx.size),
        "pop_counts": meta["population"].astype(str).value_counts().to_dict(),
        "normalize": False,
        "norm_mode": "af_residual",
        "num_snps_before_train_mono_filter": num_snps_before,
        "num_snps_after_train_mono_filter": num_snps_after,
    }
    return summary


# =============================================================================
# Config helpers
# =============================================================================

def _load_maf_from_experiment_json(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    cfg = json.loads(Path(path).read_text())
    maf = cfg.get("maf_threshold", None)
    return None if maf is None else float(maf)


def _load_discovery_from_experiment_json(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    cfg = json.loads(Path(path).read_text())
    disc = cfg.get("discovery", None)
    return None if disc is None else str(disc)


def _resolve_discovery_pop(cli_discovery: Optional[str], experiment_json: Optional[Path]) -> str:
    if cli_discovery is not None:
        return str(cli_discovery)
    disc = _load_discovery_from_experiment_json(experiment_json)
    return disc if disc is not None else "CEU"


def _load_maf_from_config(config_path: Optional[Path]) -> Optional[float]:
    if config_path is None:
        return None
    cfg = yaml.safe_load(Path(config_path).read_text())
    data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    maf = data.get("maf_threshold", None)
    return None if maf is None else float(maf)


def _format_site_stats(tree_path: Path, stats: Dict[str, Any]) -> str:
    return (
        f"Tree sequence: {tree_path}\n"
        f"Number of individuals_or_samples: {stats['num_individuals_or_samples']}\n"
        f"Total sites: {stats['num_sites_total']}\n"
        f"Segregating sites: {stats['num_segregating_sites']}\n"
        f"Multiallelic sites: {stats['num_multiallelic_sites']}\n"
        f"Biallelic segregating sites: {stats['num_biallelic_sites']}\n"
    )


# =============================================================================
# Subsetting helpers
# =============================================================================

def _choose_subset_indices(
    *,
    kept_positions_bp: np.ndarray,
    num_sites: int,
    subset_snps: int,
    subset_bp: Optional[float],
    subset_mode: str,
    subset_seed: int,
) -> Tuple[int, int]:
    if subset_bp is not None:
        return _choose_bp_window(
            kept_positions_bp,
            subset_bp=float(subset_bp),
            subset_mode=subset_mode,
            seed=subset_seed,
        )
    return _choose_contiguous_block(num_sites, subset_snps, subset_mode, seed=subset_seed)


def _choose_contiguous_block(
    num_sites: int,
    subset_snps: int,
    subset_mode: str,
    seed: int = 0,
) -> Tuple[int, int]:
    if subset_snps is None or subset_snps >= num_sites:
        return 0, num_sites
    if subset_mode == "first":
        start = 0
    elif subset_mode == "middle":
        start = max((num_sites - subset_snps) // 2, 0)
    elif subset_mode == "random":
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, num_sites - subset_snps + 1))
    else:
        raise ValueError(f"Unknown subset_mode: {subset_mode}")
    return start, start + subset_snps


def _choose_bp_window(
    positions_bp: np.ndarray,
    subset_bp: float,
    subset_mode: str,
    seed: int = 0,
) -> Tuple[int, int]:
    if positions_bp.size == 0:
        return 0, 0
    if subset_bp is None or subset_bp <= 0:
        return 0, positions_bp.size

    pos = positions_bp.astype(np.float64)
    max_start_idx = max(0, int(np.searchsorted(pos, pos[-1] - subset_bp, side="right")) - 1)
    max_start_idx = int(min(max_start_idx, pos.size - 1))

    if subset_mode == "first":
        start_idx = 0
    elif subset_mode == "middle":
        mid_bp    = 0.5 * (pos[0] + pos[-1])
        start_idx = int(max(0, min(int(np.searchsorted(pos, mid_bp - 0.5 * subset_bp, side="left")), max_start_idx)))
    elif subset_mode == "random":
        rng       = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, max_start_idx + 1)) if max_start_idx > 0 else 0
    else:
        raise ValueError(f"Unknown subset_mode: {subset_mode}")

    end_idx = int(np.searchsorted(pos, pos[start_idx] + subset_bp, side="right"))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, pos.size)
    return start_idx, end_idx


# =============================================================================
# Phenotype alignment
# =============================================================================

def _align_pheno_to_kept_inds(
    ts: tskit.TreeSequence,
    pheno: pd.DataFrame,
    kept_ind_ids: np.ndarray,
    n_rows_expected: int,
) -> pd.DataFrame:
    if "individual_id" in pheno.columns:
        pheno = pheno.sort_values("individual_id").reset_index(drop=True)

    if ts.num_individuals > 0:
        if "individual_id" not in pheno.columns:
            raise ValueError("phenotype.pkl must have an 'individual_id' column when ts has individuals.")
        pheno_indexed = pheno.set_index("individual_id", drop=False)
        missing = [i for i in kept_ind_ids.tolist() if i not in pheno_indexed.index]
        if missing:
            raise ValueError(f"Some kept tskit individual IDs missing from phenotype.pkl: {missing[:10]} ...")
        pheno = pheno_indexed.loc[kept_ind_ids].reset_index(drop=True)

    if n_rows_expected != len(pheno):
        raise ValueError(
            f"Genotype rows ({n_rows_expected}) and phenotype rows ({len(pheno)}) do not match after alignment."
        )
    return pheno


def _build_hap_meta(meta: pd.DataFrame, has_hap2: bool) -> pd.DataFrame:
    if not has_hap2:
        hm = meta.assign(hap_id=0).copy()
        hm["hap_index"] = np.arange(len(hm))
        return hm
    hm = pd.concat([meta.assign(hap_id=0), meta.assign(hap_id=1)], ignore_index=True)
    hm["hap_index"] = np.arange(len(hm))
    return hm


# =============================================================================
# Splitting
# =============================================================================

def _make_splits(
    *,
    meta: pd.DataFrame,
    discovery_pop: str,
    disc_train_frac: float,
    target_held_out_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (disc_train_idx, disc_val_idx, target_train_idx, target_held_out_idx).

    Discovery pop:
      disc_train_frac      -> discovery_train      (default 80 %)
      1 - disc_train_frac  -> discovery_validation (default 20 %)

    Target pop:
      1 - target_held_out_frac -> target_train    (default 20 %)
      target_held_out_frac     -> target_held_out (default 80 %)
    """
    if not (0.0 < disc_train_frac < 1.0):
        raise ValueError(f"disc_train_frac must be in (0,1). Got {disc_train_frac}")
    if not (0.0 < target_held_out_frac < 1.0):
        raise ValueError(f"target_held_out_frac must be in (0,1). Got {target_held_out_frac}")

    pops    = meta["population"].astype(str).to_numpy()
    all_idx = np.arange(len(meta), dtype=np.int64)

    idx_disc = all_idx[pops == discovery_pop]
    idx_targ = all_idx[pops != discovery_pop]

    if idx_disc.size < 2:
        raise ValueError(f"Need >=2 discovery ('{discovery_pop}') individuals. Got {idx_disc.size}.")
    if idx_targ.size < 2:
        raise ValueError(f"Need >=2 target individuals. Got {idx_targ.size}.")

    rng = np.random.default_rng(seed)

    # --- discovery split ---
    idx_disc = idx_disc.copy()
    rng.shuffle(idx_disc)
    n_disc_train = max(1, int(round(disc_train_frac * idx_disc.size)))
    n_disc_train = min(n_disc_train, idx_disc.size - 1)  # ensure >=1 in val
    disc_train = np.sort(idx_disc[:n_disc_train])
    disc_val   = np.sort(idx_disc[n_disc_train:])

    # --- target split ---
    idx_targ = idx_targ.copy()
    rng.shuffle(idx_targ)
    n_target_held_out = max(1, int(round(target_held_out_frac * idx_targ.size)))
    n_target_held_out = min(n_target_held_out, idx_targ.size - 1)  # ensure >=1 in train
    # held_out is the larger portion; we put it at the END of the shuffled array
    target_train    = np.sort(idx_targ[:idx_targ.size - n_target_held_out])
    target_held_out = np.sort(idx_targ[idx_targ.size - n_target_held_out:])

    return disc_train, disc_val, target_train, target_held_out


# =============================================================================
# Site stats + extraction/filtering
# =============================================================================

def compute_site_stats(ts: tskit.TreeSequence) -> dict:
    G_hap = ts.genotype_matrix()  # (sites, samples)
    num_inds = ts.num_individuals if ts.num_individuals > 0 else ts.num_samples
    segregating  = G_hap.max(axis=1) > 0
    multiallelic = G_hap.max(axis=1) > 1
    return {
        "num_individuals_or_samples": int(num_inds),
        "num_sites_total": int(ts.num_sites),
        "num_segregating_sites": int(segregating.sum()),
        "num_multiallelic_sites": int(multiallelic.sum()),
        "num_biallelic_sites": int((segregating & ~multiallelic).sum()),
    }


def _global_non_monomorphic_mask_from_haps(
    G_hap_biallelic: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    if G_hap_biallelic.size == 0:
        return np.zeros((0,), dtype=bool), {
            "num_sites_in": 0,
            "num_haplotypes": 0,
            "num_monomorphic_removed_global": 0,
            "num_sites_out": 0,
        }
    p    = G_hap_biallelic.mean(axis=1)
    maf  = np.minimum(p, 1.0 - p)
    keep = maf > 0.0
    return keep, {
        "num_sites_in": int(G_hap_biallelic.shape[0]),
        "num_haplotypes": int(G_hap_biallelic.shape[1]),
        "num_monomorphic_removed_global": int((maf == 0.0).sum()),
        "num_sites_out": int(keep.sum()),
    }


def _extract_haps_and_diploid(ts: tskit.TreeSequence):
    G_hap = ts.genotype_matrix()  # (sites, samples)
    filter_report: Dict[str, object] = {}

    site_positions_bp_all = ts.tables.sites.position.astype(np.float64)
    site_ids_all          = np.arange(ts.num_sites, dtype=np.int32)

    biallelic = G_hap.max(axis=1) <= 1
    if (G_hap.min(axis=1) < 0).any():
        biallelic &= G_hap.min(axis=1) >= 0

    filter_report["num_sites_raw"] = int(G_hap.shape[0])
    filter_report["num_sites_after_biallelic_nonmissing"] = int(biallelic.sum())

    G_hap             = G_hap[biallelic, :]
    site_positions_bp = site_positions_bp_all[biallelic]
    site_ids          = site_ids_all[biallelic]

    keep_global_poly, global_poly_info = _global_non_monomorphic_mask_from_haps(G_hap.astype(np.float32))
    filter_report.update(global_poly_info)

    G_hap             = G_hap[keep_global_poly, :]
    site_positions_bp = site_positions_bp[keep_global_poly]
    site_ids          = site_ids[keep_global_poly]

    if ts.num_individuals == 0:
        hap1 = G_hap.T.astype(np.float32)
        return hap1, None, hap1.copy(), np.arange(hap1.shape[0], dtype=np.int64), filter_report, site_positions_bp, site_ids

    samples      = ts.samples()
    node_to_col  = np.full(ts.num_nodes, -1, dtype=np.int32)
    node_to_col[samples] = np.arange(samples.size, dtype=np.int32)

    inds = list(ts.individuals())
    nodes2           = np.full((len(inds), 2), -1, dtype=np.int32)
    kept_ind_ids_all = np.full((len(inds),), -1, dtype=np.int64)

    non_diploid_ids: List[int] = []
    for i, ind in enumerate(inds):
        kept_ind_ids_all[i] = ind.id
        if len(ind.nodes) != 2:
            non_diploid_ids.append(ind.id)
            continue
        nodes2[i, 0] = ind.nodes[0]
        nodes2[i, 1] = ind.nodes[1]

    if non_diploid_ids:
        raise ValueError(
            "Found individuals that do not have exactly 2 nodes (diploid requirement). "
            f"First few IDs: {non_diploid_ids[:10]}"
        )

    cols2        = node_to_col[nodes2]
    valid        = (cols2[:, 0] >= 0) & (cols2[:, 1] >= 0)
    cols2        = cols2[valid]
    kept_ind_ids = kept_ind_ids_all[valid]

    hap1 = G_hap[:, cols2[:, 0]].T.astype(np.float32)
    hap2 = G_hap[:, cols2[:, 1]].T.astype(np.float32)
    dip  = (hap1 + hap2).astype(np.float32)

    return hap1, hap2, dip, kept_ind_ids, filter_report, site_positions_bp, site_ids