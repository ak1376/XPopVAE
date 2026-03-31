#!/usr/bin/env python3
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
    subset_mode: str = "first"  # "first" | "middle" | "random"
    subset_seed: int = 0

    # Discovery train/val split (val_frac of discovery population)
    discovery_val_frac: float = 0.2
    # Target train/test split (test_frac of target population)
    target_test_frac: float = 0.2

    split_seed: int = 0
    discovery_pop: Optional[str] = (
        None  # if None, read from experiment_config_json, else default "CEU"
    )

    # Always normalize by HWE (af_residual)
    norm_eps: float = 1e-6
    norm_clip_std_min: float = 1e-3


def build_genotypes_for_vae(a: BuildGenotypesArgs) -> Dict[str, Any]:
    """
    End-to-end builder:
      - Load TS + phenotype
      - Filter sites globally for structural validity only:
          * biallelic
          * non-missing
          * globally non-monomorphic
      - Extract diploid / haplotype matrices
      - Align phenotype to genotype rows
      - Split discovery into train/val and target into train/test (both 80/20 by default)
      - Subset window
      - Filter SNPs using discovery_train only:
          * remove monomorphic-in-discovery_train
          * if maf_threshold > 0, apply discovery_train MAF filter
      - Save arrays + metadata

    Saved .npy arrays
    -----------------
    discovery.npy             all discovery individuals (unsplit)
    target.npy                all target individuals (unsplit)
    train_discovery.npy       discovery training split
    validation_discovery.npy  discovery validation split
    train_target.npy          target training split
    test_target.npy           target test split

    Saved index DataFrames (columns: row_index, individual_id, population, dataset)
    -------------------------------------------------------------------------------
    discovery_split_index.pkl / .csv
    target_split_index.pkl   / .csv
    """
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
    #
    # No MAF filter here. MAF filtering happens later on discovery_train only.
    # -------------------------------------------------------------------------
    hap1, hap2, G, kept_ind_ids, filt, kept_positions_bp, kept_site_ids = (
        _extract_haps_and_diploid(ts)
    )
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

    # Meta aligned to genotype rows
    meta = pheno[["individual_id", "population", "phenotype"]].copy()
    meta.to_pickle(outdir / "meta.pkl")

    hap_meta = _build_hap_meta(meta, has_hap2=(hap2 is not None))
    hap_meta.to_pickle(outdir / "hap_meta.pkl")

    # -------------------------------------------------------------------------
    # Split: discovery -> train/val; target -> train/test
    # -------------------------------------------------------------------------
    disc_train_idx, disc_val_idx, targ_train_idx, targ_test_idx = _make_splits(
        meta=meta,
        discovery_pop=discovery_pop,
        discovery_val_frac=float(a.discovery_val_frac),
        target_test_frac=float(a.target_test_frac),
        seed=int(a.split_seed),
    )

    # Save raw index arrays (row positions into the aligned G matrix)
    np.save(outdir / "discovery_train_idx.npy", disc_train_idx.astype(np.int64))
    np.save(outdir / "discovery_val_idx.npy", disc_val_idx.astype(np.int64))
    np.save(outdir / "target_train_idx.npy", targ_train_idx.astype(np.int64))
    np.save(outdir / "target_test_idx.npy", targ_test_idx.astype(np.int64))

    # -------------------------------------------------------------------------
    # Subset window (AFTER global structural filtering, BEFORE discovery-train MAF)
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
        raise RuntimeError(
            f"Subset window invalid: start={start}, end={end}, num_sites={num_sites}"
        )

    G_subset = G[:, start:end].astype(np.float32)
    hap1_subset = hap1[:, start:end].astype(np.float32)
    hap2_subset = None if hap2 is None else hap2[:, start:end].astype(np.float32)

    kept_positions_subset = kept_positions_bp[start:end].astype(np.float64)
    kept_site_ids_subset = kept_site_ids[start:end].astype(np.int32)
    snp_idx = np.arange(start, end, dtype=np.int64)

    # -------------------------------------------------------------------------
    # CRITICAL: define feature set using discovery_train only.
    #
    # A site may be polymorphic globally but monomorphic in discovery_train.
    # We therefore:
    #   1) remove monomorphic sites in discovery_train
    #   2) if maf_threshold > 0, remove sites with low MAF in discovery_train
    # -------------------------------------------------------------------------
    num_snps_before = int(G_subset.shape[1])

    G_disc_train_pre = G_subset[disc_train_idx]  # diploid dosages 0/1/2
    p_train = G_disc_train_pre.mean(axis=0, dtype=np.float64) / 2.0
    maf_train = np.minimum(p_train, 1.0 - p_train)

    # Start by removing monomorphic-in-discovery_train columns
    keep_cols = maf_train > 0.0
    num_mono_removed = int((maf_train == 0.0).sum())

    # Then optionally apply discovery_train MAF threshold
    num_maf_removed = 0
    min_ac_implied = None
    if maf_threshold is not None and maf_threshold > 0.0:
        n_disc_train_haps = 2 * int(G_disc_train_pre.shape[0])
        min_ac_implied = int(math.ceil(float(maf_threshold) * n_disc_train_haps))

        before = int(keep_cols.sum())
        keep_cols &= maf_train >= float(maf_threshold)
        after = int(keep_cols.sum())
        num_maf_removed = before - after

    if int(keep_cols.sum()) == 0:
        raise RuntimeError(
            "All SNPs were removed by discovery_train filtering. "
            f"subset_start={start}, subset_end={end}, maf_threshold={maf_threshold}"
        )

    # Apply same SNP mask to all rows / outputs
    G_subset = G_subset[:, keep_cols]
    hap1_subset = hap1_subset[:, keep_cols]
    if hap2_subset is not None:
        hap2_subset = hap2_subset[:, keep_cols]

    kept_positions_subset = kept_positions_subset[keep_cols]
    kept_site_ids_subset = kept_site_ids_subset[keep_cols]
    snp_idx = snp_idx[keep_cols]

    num_snps_after = int(G_subset.shape[1])

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
    # Save SNP-position / site metadata
    # -------------------------------------------------------------------------
    np.save(outdir / "hap1.npy", hap1_subset)
    np.save(
        outdir / "hap2.npy",
        (
            hap2_subset
            if hap2_subset is not None
            else np.zeros_like(hap1_subset, dtype=np.float32)
        ),
    )
    np.save(outdir / "snp_index.npy", snp_idx)
    np.save(outdir / "variant_positions_bp.npy", kept_positions_subset)
    np.save(outdir / "variant_site_ids.npy", kept_site_ids_subset)
    np.save(outdir / "ts_individual_ids.npy", kept_ind_ids.astype(np.int64))

    # -------------------------------------------------------------------------
    # Build per-population genotype slices and save
    # -------------------------------------------------------------------------
    # Full unsplit arrays
    disc_all_idx = np.concatenate([disc_train_idx, disc_val_idx])
    disc_all_idx = np.sort(disc_all_idx)
    targ_all_idx = np.concatenate([targ_train_idx, targ_test_idx])
    targ_all_idx = np.sort(targ_all_idx)

    G_discovery = G_subset[disc_all_idx]
    G_target = G_subset[targ_all_idx]

    # Split arrays
    G_disc_train = G_subset[disc_train_idx]
    G_disc_val = G_subset[disc_val_idx]
    G_targ_train = G_subset[targ_train_idx]
    G_targ_test = G_subset[targ_test_idx]

    np.save(outdir / "discovery.npy", G_discovery.astype(np.float32))
    np.save(outdir / "target.npy", G_target.astype(np.float32))
    np.save(outdir / "train_discovery.npy", G_disc_train.astype(np.float32))
    np.save(outdir / "validation_discovery.npy", G_disc_val.astype(np.float32))
    np.save(outdir / "train_target.npy", G_targ_train.astype(np.float32))
    np.save(outdir / "test_target.npy", G_targ_test.astype(np.float32))

    # -------------------------------------------------------------------------
    # Phenotype arrays (aligned to each split)
    # -------------------------------------------------------------------------
    y_all = meta["phenotype"].to_numpy()

    np.save(outdir / "discovery_pheno.npy", y_all[disc_all_idx])
    np.save(outdir / "target_pheno.npy", y_all[targ_all_idx])
    np.save(outdir / "train_discovery_pheno.npy", y_all[disc_train_idx])
    np.save(outdir / "validation_discovery_pheno.npy", y_all[disc_val_idx])
    np.save(outdir / "train_target_pheno.npy", y_all[targ_train_idx])
    np.save(outdir / "test_target_pheno.npy", y_all[targ_test_idx])

    # -------------------------------------------------------------------------
    # Index DataFrames
    # Columns: row_index (position in the per-population array), individual_id,
    #          population, dataset (train / validation / test)
    # -------------------------------------------------------------------------
    disc_index_df = _build_split_index_df(
        all_sorted_idx=disc_all_idx,
        split_idx_map={
            "train": disc_train_idx,
            "validation": disc_val_idx,
        },
        meta=meta,
    )
    disc_index_df.to_pickle(outdir / "discovery_split_index.pkl")
    disc_index_df.to_csv(outdir / "discovery_split_index.csv", index=False)

    targ_index_df = _build_split_index_df(
        all_sorted_idx=targ_all_idx,
        split_idx_map={
            "train": targ_train_idx,
            "test": targ_test_idx,
        },
        meta=meta,
    )
    targ_index_df.to_pickle(outdir / "target_split_index.pkl")
    targ_index_df.to_csv(outdir / "target_split_index.csv", index=False)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    summary = {
        "outdir": str(outdir),
        "maf_threshold": float(maf_threshold),
        "subset_start": int(start),
        "subset_end": int(end),
        "num_inds": int(G_subset.shape[0]),
        "num_snps": int(G_subset.shape[1]),
        "discovery_val_frac": float(a.discovery_val_frac),
        "target_test_frac": float(a.target_test_frac),
        "split_seed": int(a.split_seed),
        "discovery_pop": str(discovery_pop),
        "n_discovery": int(disc_all_idx.size),
        "n_discovery_train": int(disc_train_idx.size),
        "n_discovery_val": int(disc_val_idx.size),
        "n_target": int(targ_all_idx.size),
        "n_target_train": int(targ_train_idx.size),
        "n_target_test": int(targ_test_idx.size),
        "pop_counts": meta["population"].astype(str).value_counts().to_dict(),
        "normalize": False,
        "norm_mode": "af_residual",
        "num_snps_before_train_mono_filter": num_snps_before,
        "num_snps_after_train_mono_filter": num_snps_after,
    }
    return summary


# =============================================================================
# Index DataFrame builder
# =============================================================================


def _build_split_index_df(
    all_sorted_idx: np.ndarray,
    split_idx_map: Dict[str, np.ndarray],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a DataFrame mapping each individual in a population group to its
    position in the saved per-population array, its tskit individual_id,
    its population label, and its dataset assignment.

    Parameters
    ----------
    all_sorted_idx : sorted array of row positions into `meta` / G_subset
                     for this population group (e.g. disc_all_idx)
    split_idx_map  : dict mapping dataset label -> array of row positions
                     (positions are into meta / G_subset, same coordinate
                     space as all_sorted_idx)
    meta           : aligned metadata DataFrame

    Returns
    -------
    DataFrame with columns:
        row_index     - position in the per-population saved array (0-based)
        individual_id - tskit individual id
        population    - population string
        dataset       - "train" | "validation" | "test"
    """
    # Build a lookup: global_row -> dataset label
    global_to_dataset: Dict[int, str] = {}
    for label, idx_arr in split_idx_map.items():
        for i in idx_arr.tolist():
            global_to_dataset[int(i)] = label

    records = []
    for row_index, global_row in enumerate(all_sorted_idx.tolist()):
        row = meta.iloc[global_row]
        records.append(
            {
                "row_index": row_index,
                "individual_id": int(row["individual_id"]),
                "population": str(row["population"]),
                "dataset": global_to_dataset[int(global_row)],
            }
        )

    return pd.DataFrame(
        records, columns=["row_index", "individual_id", "population", "dataset"]
    )


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


def _resolve_discovery_pop(
    cli_discovery: Optional[str], experiment_json: Optional[Path]
) -> str:
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
        start, end = _choose_bp_window(
            kept_positions_bp,
            subset_bp=float(subset_bp),
            subset_mode=subset_mode,
            seed=subset_seed,
        )
    else:
        start, end = _choose_contiguous_block(
            num_sites, subset_snps, subset_mode, seed=subset_seed
        )
    return start, end


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

    end = start + subset_snps
    return start, end


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
    max_start_idx = np.searchsorted(pos, pos[-1] - subset_bp, side="right") - 1
    max_start_idx = int(max(0, min(max_start_idx, pos.size - 1)))

    if subset_mode == "first":
        start_idx = 0
    elif subset_mode == "middle":
        mid_bp = 0.5 * (pos[0] + pos[-1])
        start_bp = mid_bp - 0.5 * subset_bp
        start_idx = int(np.searchsorted(pos, start_bp, side="left"))
        start_idx = int(max(0, min(start_idx, max_start_idx)))
    elif subset_mode == "random":
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, max_start_idx + 1)) if max_start_idx > 0 else 0
    else:
        raise ValueError(f"Unknown subset_mode: {subset_mode}")

    end_bp = pos[start_idx] + subset_bp
    end_idx = int(np.searchsorted(pos, end_bp, side="right"))

    if end_idx <= start_idx and pos.size > 0:
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
            raise ValueError(
                "phenotype.pkl must have an 'individual_id' column when ts has individuals."
            )
        pheno_indexed = pheno.set_index("individual_id", drop=False)
        missing = [i for i in kept_ind_ids.tolist() if i not in pheno_indexed.index]
        if missing:
            raise ValueError(
                f"Some kept tskit individual IDs missing from phenotype.pkl: {missing[:10]} ..."
            )
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
    discovery_val_frac: float,
    target_test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split individuals into four index arrays (positions into the aligned G matrix):
        disc_train_idx  - discovery training set
        disc_val_idx    - discovery validation set
        targ_train_idx  - target training set
        targ_test_idx   - target test set

    Discovery is split by `discovery_val_frac` (val fraction).
    Target is split by `target_test_frac` (test fraction).
    Both splits share the same `seed` but use independent RNG states so they
    don't interfere with each other.
    """
    pops = meta["population"].astype(str).to_numpy()
    n = len(meta)
    if n < 2:
        raise ValueError("Need at least 2 individuals to split.")

    all_idx = np.arange(n, dtype=np.int64)
    idx_disc = all_idx[pops == discovery_pop]
    idx_targ = all_idx[pops != discovery_pop]

    if idx_disc.size < 2:
        raise ValueError(
            f"Need >=2 individuals in discovery population '{discovery_pop}'. Got {idx_disc.size}."
        )
    if idx_targ.size < 2:
        raise ValueError(
            f"Need >=2 individuals outside discovery population '{discovery_pop}' for target. "
            f"Got {idx_targ.size}."
        )

    if not (0.0 < discovery_val_frac < 1.0):
        raise ValueError(
            f"discovery_val_frac must be in (0, 1). Got {discovery_val_frac}"
        )
    if not (0.0 < target_test_frac < 1.0):
        raise ValueError(f"target_test_frac must be in (0, 1). Got {target_test_frac}")

    # Discovery split
    rng_disc = np.random.default_rng(seed)
    idx_disc = idx_disc.copy()
    rng_disc.shuffle(idx_disc)

    n_disc_val = max(1, int(round(discovery_val_frac * idx_disc.size)))
    if idx_disc.size - n_disc_val < 1:
        n_disc_val = idx_disc.size - 1

    disc_val = np.sort(idx_disc[:n_disc_val])
    disc_train = np.sort(idx_disc[n_disc_val:])

    # Target split — use a derived seed so it's independent from disc shuffle
    rng_targ = np.random.default_rng(seed + 1)
    idx_targ = idx_targ.copy()
    rng_targ.shuffle(idx_targ)

    n_targ_test = max(1, int(round(target_test_frac * idx_targ.size)))
    if idx_targ.size - n_targ_test < 1:
        n_targ_test = idx_targ.size - 1

    targ_test = np.sort(idx_targ[:n_targ_test])
    targ_train = np.sort(idx_targ[n_targ_test:])

    return disc_train, disc_val, targ_train, targ_test


# =============================================================================
# Site stats + extraction/filtering
# =============================================================================


def compute_site_stats(ts: tskit.TreeSequence) -> dict:
    G_hap = ts.genotype_matrix()  # (sites, samples)
    num_sites_total = ts.num_sites
    num_inds = ts.num_individuals if ts.num_individuals > 0 else ts.num_samples

    segregating = G_hap.max(axis=1) > 0
    multiallelic = G_hap.max(axis=1) > 1

    return {
        "num_individuals_or_samples": int(num_inds),
        "num_sites_total": int(num_sites_total),
        "num_segregating_sites": int(segregating.sum()),
        "num_multiallelic_sites": int(multiallelic.sum()),
        "num_biallelic_sites": int((segregating & ~multiallelic).sum()),
    }


def _global_non_monomorphic_mask_from_haps(
    G_hap_biallelic: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """
    Keep sites that are globally non-monomorphic across all haplotypes.
    This is structural filtering only, not discovery-train feature selection.
    """
    if G_hap_biallelic.size == 0:
        keep = np.zeros((0,), dtype=bool)
        return keep, {
            "num_sites_in": 0,
            "num_haplotypes": 0,
            "num_monomorphic_removed_global": 0,
            "num_sites_out": 0,
        }

    p = G_hap_biallelic.mean(axis=1)
    maf = np.minimum(p, 1.0 - p)
    keep = maf > 0.0

    info = {
        "num_sites_in": int(G_hap_biallelic.shape[0]),
        "num_haplotypes": int(G_hap_biallelic.shape[1]),
        "num_monomorphic_removed_global": int((maf == 0.0).sum()),
        "num_sites_out": int(keep.sum()),
    }
    return keep, info


def _extract_haps_and_diploid(ts: tskit.TreeSequence):
    """
    Extract haplotype and diploid genotype matrices after applying only:
      - biallelic filtering
      - non-missing filtering
      - global non-monomorphic filtering

    No MAF filtering is done here.
    """
    G_hap = ts.genotype_matrix()  # (sites, samples)
    filter_report: Dict[str, object] = {}

    site_positions_bp_all = ts.tables.sites.position.astype(np.float64)
    site_ids_all = np.arange(ts.num_sites, dtype=np.int32)

    # biallelic + non-missing
    biallelic = G_hap.max(axis=1) <= 1
    if (G_hap.min(axis=1) < 0).any():
        biallelic &= G_hap.min(axis=1) >= 0

    filter_report["num_sites_raw"] = int(G_hap.shape[0])
    filter_report["num_sites_after_biallelic_nonmissing"] = int(biallelic.sum())

    G_hap = G_hap[biallelic, :]
    site_positions_bp = site_positions_bp_all[biallelic]
    site_ids = site_ids_all[biallelic]

    # Remove globally monomorphic sites only
    keep_global_poly, global_poly_info = _global_non_monomorphic_mask_from_haps(
        G_hap.astype(np.float32)
    )
    filter_report.update(global_poly_info)

    G_hap = G_hap[keep_global_poly, :]
    site_positions_bp = site_positions_bp[keep_global_poly]
    site_ids = site_ids[keep_global_poly]

    # If TS has no individuals, treat samples as haploid rows
    if ts.num_individuals == 0:
        hap1 = G_hap.T.astype(np.float32)
        dip = hap1.copy()
        kept_ind_ids = np.arange(hap1.shape[0], dtype=np.int64)
        return hap1, None, dip, kept_ind_ids, filter_report, site_positions_bp, site_ids

    # Map nodes -> sample columns
    samples = ts.samples()
    node_to_col = np.full(ts.num_nodes, -1, dtype=np.int32)
    node_to_col[samples] = np.arange(samples.size, dtype=np.int32)

    inds = list(ts.individuals())
    nodes2 = np.full((len(inds), 2), -1, dtype=np.int32)
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

    cols2 = node_to_col[nodes2]
    valid = (cols2[:, 0] >= 0) & (cols2[:, 1] >= 0)
    cols2 = cols2[valid]
    kept_ind_ids = kept_ind_ids_all[valid]

    hap1 = G_hap[:, cols2[:, 0]].T.astype(np.float32)
    hap2 = G_hap[:, cols2[:, 1]].T.astype(np.float32)
    dip = (hap1 + hap2).astype(np.float32)

    return hap1, hap2, dip, kept_ind_ids, filter_report, site_positions_bp, site_ids