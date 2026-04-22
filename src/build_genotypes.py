#!/usr/bin/env python3
# src/build_genotypes.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import tskit
import tstrait


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed=seed)


def load_trees_sequence(tree_path: Path) -> tskit.TreeSequence:
    return tskit.load(str(tree_path))


def build_haploid_matrix(ts: tskit.TreeSequence) -> np.ndarray:
    return ts.genotype_matrix()


def filter_multiallelic_and_monomorphic(
    ts: tskit.TreeSequence,
    genotype_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    biallelic_site_ids = np.array(
        [v.site.id for v in ts.variants() if len(v.alleles) == 2]
    )
    hap = genotype_matrix[biallelic_site_ids, :]
    print(f"  sites after biallelic filter   : {len(biallelic_site_ids)}")

    # FAST: for binary haploid matrix, polymorphic = not all same value
    # hap.any(axis=1) is True if at least one 1 exists (not all-zero)
    # ~hap.all(axis=1) is True if at least one 0 exists (not all-one)
    polymorphic_mask = hap.any(axis=1) & ~hap.all(axis=1)
    hap = hap[polymorphic_mask, :]
    biallelic_site_ids = biallelic_site_ids[polymorphic_mask]
    print(f"  sites after monomorphic filter : {len(biallelic_site_ids)}")

    return hap, biallelic_site_ids


def apply_maf_filter_on_discovery(
    diploid_full: np.ndarray,
    biallelic_site_ids: np.ndarray,
    discovery_inds: np.ndarray,
    maf_threshold: float,
    causal_site_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply MAF filter using only discovery train individuals, report causal site loss."""

    # compute MAF only in discovery train individuals
    discovery_diploid = diploid_full[discovery_inds, :]
    allele_freqs = discovery_diploid.mean(axis=0) / 2  # divide by 2 since diploid
    maf = np.minimum(allele_freqs, 1 - allele_freqs)
    maf_mask = maf >= maf_threshold

    causal_set = set(causal_site_ids)
    causal_before = causal_set & set(biallelic_site_ids)
    causal_after = causal_set & set(biallelic_site_ids[maf_mask])
    causal_lost = causal_before - causal_after

    maf_report = {
        "maf_threshold": maf_threshold,
        "sites_before_maf": len(biallelic_site_ids),
        "sites_after_maf": int(maf_mask.sum()),
        "causal_sites_before_maf": len(causal_before),
        "causal_sites_after_maf": len(causal_after),
        "causal_sites_lost": len(causal_lost),
        "pct_causal_lost": (
            100 * len(causal_lost) / len(causal_before) if causal_before else 0
        ),
    }

    print(f"  sites after MAF filter (CEU)   : {int(maf_mask.sum())}")
    print(
        f"  causal sites lost to MAF       : {maf_report['causal_sites_lost']}/{maf_report['causal_sites_before_maf']} ({maf_report['pct_causal_lost']:.1f}%)"
    )

    return maf_mask, biallelic_site_ids[maf_mask], maf_report


def diploid_matrix(hap: np.ndarray) -> np.ndarray:
    dosage = hap[:, 0::2] + hap[:, 1::2]
    diploid = dosage.T
    return diploid


def pop_metadata(ts: tskit.TreeSequence) -> dict[int, str]:
    pop_id_to_name = {pop.id: pop.metadata["name"] for pop in ts.populations()}
    return {ind.id: pop_id_to_name[ind.population] for ind in ts.individuals()}


def simulate_trait(
    ts: tskit.TreeSequence,
    num_causal: int,
    biallelic_site_ids: np.ndarray,
    rng: np.random.Generator,
    distribution: str = "normal",
    trait_seed: int = 42,  # <-- ADD THIS
) -> tuple:
    # Use a SEPARATE rng seeded only by trait_seed, not the shared rng
    # This will ensure that the same causal sites will be used across different replicates. Note that because of the differences in genealogy, the amount of causal SNPs that may be filtered out by the MAF filter will differ, but the same causal SNPs will be used to generate the phenotype.
    trait_rng = np.random.default_rng(trait_seed)
    causal_site_ids = trait_rng.choice(
        biallelic_site_ids, size=num_causal, replace=False
    ).tolist()
    model = tstrait.trait_model(distribution=distribution, mean=0, var=1)
    trait_df = tstrait.sim_trait(
        ts, causal_sites=causal_site_ids, model=model, alpha=0, random_seed=42
    )
    return trait_df, causal_site_ids


def simulate_phenotype(
    ts: tskit.TreeSequence,
    distribution: str,
    causal_site_ids: list[int],
    train_discovery_inds: np.ndarray,
    validation_discovery_inds: np.ndarray,
    train_target_inds: np.ndarray,
    test_target_inds: np.ndarray,
    alpha: float,
    h2: float,
    random_seed: int,
) -> tuple:
    model = tstrait.trait_model(distribution=distribution, mean=0, var=1)
    phenotype_df = tstrait.sim_phenotype(
        ts,
        model=model,
        causal_sites=causal_site_ids,
        alpha=alpha,
        h2=h2,
        random_seed=random_seed,
    )
    pheno = phenotype_df.phenotype

    return (
        pheno[pheno["individual_id"].isin(train_discovery_inds)],
        pheno[pheno["individual_id"].isin(validation_discovery_inds)],
        pheno[pheno["individual_id"].isin(train_target_inds)],
        pheno[pheno["individual_id"].isin(test_target_inds)],
    )
