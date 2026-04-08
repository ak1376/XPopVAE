#!/usr/bin/env python3
# simulate_phenotype_for_test.py

'''
Samples causal SNPs from the genomic window using the DISCOVERY TRAIN set,
fits a phenotype architecture on DISCOVERY TRAIN, and applies it to all splits:

    discovery_train      (CEU, 80%)  -- fitting cohort
    discovery_validation (CEU, 20%)  -- within-pop evaluation
    target_train         (YRI, 20%)  -- seen during training (mixed batch)
    target_held_out      (YRI, 80%)  -- final cross-pop evaluation

Phenotype model:
    y = G_beta + e
    where noise variance is set to match the requested heritability.

Standardization note:
    CEU splits are standardized using CEU discovery train mean/std (stored in
    the architecture). YRI splits are standardized using their own population
    mean/std (use_pop_specific_standardization=True). This prevents the
    systematic sign inversion that occurs when CEU allele frequencies are
    applied to YRI genotypes at sites with large frequency divergence.
    The effect sizes are still CEU-fit, so cross-population transfer remains
    genuinely hard — but not artificially impossible.

Evaluations:
    Oracle  R² (Ridge on causal SNPs only)  -- upper bound
    Baseline R² (Ridge on all SNPs / gBLUP) -- cross-population baseline to beat

Both are evaluated on discovery_validation, target_train, and target_held_out.
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import os


# =============================================================================
# Architecture helpers
# =============================================================================

def fit_phenotype_architecture(
    genotype_matrix,
    num_causal=100,
    heritability=0.7,
    seed=0,
    standardize=True,
):
    '''
    Fit the phenotype architecture on the training genotype matrix.

    Returns:
        architecture : dict with causal_snps, effect_sizes, mean, std,
                       noise_sd, heritability, standardize
        genetic_component_train : ndarray (n_individuals,)
    '''
    rng = np.random.default_rng(seed)
    n_individuals, n_snps = genotype_matrix.shape

    # choose causal SNPs from polymorphic sites in DISCOVERY TRAIN
    maf = genotype_matrix.mean(axis=0) / 2.0
    polymorphic = np.where((maf > 0) & (maf < 1))[0]

    if num_causal > len(polymorphic):
        raise ValueError(
            f"Requested {num_causal} causal SNPs but only {len(polymorphic)} "
            "polymorphic SNPs available in discovery_train."
        )

    causal_snps = rng.choice(polymorphic, size=num_causal, replace=False)
    X = genotype_matrix[:, causal_snps].astype(float)

    if standardize:
        mean = X.mean(axis=0)
        std  = X.std(axis=0, ddof=0)
        keep = std > 0
        X, causal_snps, mean, std = X[:, keep], causal_snps[keep], mean[keep], std[keep]
        if X.shape[1] == 0:
            raise ValueError("All selected causal SNPs became non-variable after filtering.")
        X = (X - mean) / std
    else:
        mean = np.zeros(X.shape[1], dtype=float)
        std  = np.ones(X.shape[1],  dtype=float)

    effect_sizes = rng.normal(loc=0.0, scale=1.0, size=X.shape[1])
    genetic_component_train = X @ effect_sizes
    var_g = np.var(genetic_component_train, ddof=0)
    if var_g == 0:
        raise ValueError("Genetic variance is zero on DISCOVERY TRAIN.")

    var_e    = var_g * (1.0 - heritability) / heritability
    noise_sd = np.sqrt(var_e)

    architecture = dict(
        causal_snps=causal_snps,
        effect_sizes=effect_sizes,
        mean=mean,
        std=std,
        noise_sd=noise_sd,
        heritability=heritability,
        standardize=standardize,
    )
    return architecture, genetic_component_train


def apply_phenotype_architecture(
    genotype_matrix,
    architecture,
    seed=0,
    use_pop_specific_standardization=False,
):
    '''
    Apply a pre-fit architecture to a genotype matrix.

    Parameters
    ----------
    genotype_matrix : (n_individuals, n_snps)
    architecture    : dict from fit_phenotype_architecture
    seed            : int
    use_pop_specific_standardization : bool
        If True, standardize causal SNPs using this population's own mean/std
        rather than the CEU discovery train mean/std stored in the architecture.
        Use this when applying to YRI to avoid inverting predictions due to
        allele frequency divergence at causal sites. Effect sizes remain
        CEU-fit, so cross-population transfer is still genuinely hard.

    Returns
    -------
    phenotype            : (n_individuals,)
    genetic_component    : (n_individuals,)
    environmental_noise  : (n_individuals,)
    '''
    rng = np.random.default_rng(seed)
    causal_snps  = architecture["causal_snps"]
    effect_sizes = architecture["effect_sizes"]
    noise_sd     = architecture["noise_sd"]
    standardize  = architecture["standardize"]

    X = genotype_matrix[:, causal_snps].astype(float)

    if standardize:
        if use_pop_specific_standardization:
            mean = X.mean(axis=0)
            std  = X.std(axis=0, ddof=0)
            std[std == 0] = 1.0
        else:
            mean = architecture["mean"]
            std  = architecture["std"]
        X = (X - mean) / std

    genetic_component   = X @ effect_sizes
    environmental_noise = rng.normal(loc=0.0, scale=noise_sd, size=genotype_matrix.shape[0])
    phenotype           = genetic_component + environmental_noise
    return phenotype, genetic_component, environmental_noise


# =============================================================================
# Main
# =============================================================================

def main():
    # -------------------------------------------------
    # paths
    # -------------------------------------------------
    output_path = Path("phenotype_creation")
    os.makedirs(output_path, exist_ok=True)

    base = Path(
        "/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0"
    )
    geno_dir = base / "genotype_matrices"

    disc_train_path      = geno_dir / "discovery_train.npy"
    disc_val_path        = geno_dir / "discovery_validation.npy"
    target_train_path    = geno_dir / "target_train.npy"
    target_held_out_path = geno_dir / "target_held_out.npy"

    # -------------------------------------------------
    # load genotypes
    # -------------------------------------------------
    X_disc_train      = np.load(disc_train_path)
    X_disc_val        = np.load(disc_val_path)
    X_target_train    = np.load(target_train_path)
    X_target_held_out = np.load(target_held_out_path)

    num_causal   = 100
    heritability = 1.0
    seed         = 295
    standardize  = True

    print("Loaded genotype matrices")
    print(f"  discovery_train      : {X_disc_train.shape}")
    print(f"  discovery_validation : {X_disc_val.shape}")
    print(f"  target_train         : {X_target_train.shape}")
    print(f"  target_held_out      : {X_target_held_out.shape}")
    print()

    # -------------------------------------------------
    # fit architecture on DISCOVERY TRAIN only
    # -------------------------------------------------
    architecture, g_disc_train_no_noise = fit_phenotype_architecture(
        X_disc_train,
        num_causal=num_causal,
        heritability=heritability,
        seed=seed,
        standardize=standardize,
    )
    causal_snps = architecture["causal_snps"]
    betas       = architecture["effect_sizes"]

    print(f"Causal SNPs selected: {len(causal_snps)}")
    print(f"CEU causal SNP MAF range: "
          f"{(X_disc_train[:, causal_snps].mean(axis=0) / 2.0).min():.3f} – "
          f"{(X_disc_train[:, causal_snps].mean(axis=0) / 2.0).max():.3f}")
    print(f"YRI causal SNP MAF range: "
          f"{(X_target_train[:, causal_snps].mean(axis=0) / 2.0).min():.3f} – "
          f"{(X_target_train[:, causal_snps].mean(axis=0) / 2.0).max():.3f}")
    print()

    # -------------------------------------------------
    # apply architecture to all splits
    # CEU splits: use CEU standardization (stored in architecture)
    # YRI splits: use population-specific standardization
    # -------------------------------------------------
    y_disc_train,      g_disc_train,      e_disc_train      = apply_phenotype_architecture(
        X_disc_train, architecture, seed=seed,
        use_pop_specific_standardization=False,
    )
    y_disc_val,        g_disc_val,        e_disc_val        = apply_phenotype_architecture(
        X_disc_val, architecture, seed=seed + 1,
        use_pop_specific_standardization=False,
    )
    y_target_train,    g_target_train,    e_target_train    = apply_phenotype_architecture(
        X_target_train, architecture, seed=seed + 2,
        use_pop_specific_standardization=True,
    )
    y_target_held_out, g_target_held_out, e_target_held_out = apply_phenotype_architecture(
        X_target_held_out, architecture, seed=seed + 3,
        use_pop_specific_standardization=True,
    )

    # -------------------------------------------------
    # Oracle: Ridge on causal SNPs only  (upper bound)
    # -------------------------------------------------
    print("=" * 60)
    print("ORACLE (causal SNPs only, trained on CEU disc_train)")
    print("=" * 60)

    X_disc_train_causal      = X_disc_train[:, causal_snps]
    X_disc_val_causal        = X_disc_val[:, causal_snps]
    X_target_train_causal    = X_target_train[:, causal_snps]
    X_target_held_out_causal = X_target_held_out[:, causal_snps]

    oracle_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(
        X_disc_train_causal, y_disc_train
    )

    y_pred_oracle_disc_val        = oracle_model.predict(X_disc_val_causal)
    y_pred_oracle_target_train    = oracle_model.predict(X_target_train_causal)
    y_pred_oracle_target_held_out = oracle_model.predict(X_target_held_out_causal)

    r2_oracle_disc_val        = r2_score(y_disc_val,        y_pred_oracle_disc_val)
    r2_oracle_target_train    = r2_score(y_target_train,    y_pred_oracle_target_train)
    r2_oracle_target_held_out = r2_score(y_target_held_out, y_pred_oracle_target_held_out)

    print(f"Oracle R² — discovery_validation (CEU within-pop) : {r2_oracle_disc_val:.4f}")
    print(f"Oracle R² — target_train         (YRI cross-pop)  : {r2_oracle_target_train:.4f}")
    print(f"Oracle R² — target_held_out      (YRI cross-pop)  : {r2_oracle_target_held_out:.4f}")
    print()

    # -------------------------------------------------
    # Baseline: Ridge on all SNPs  (gBLUP equivalent)
    # -------------------------------------------------
    print("=" * 60)
    print("BASELINE / gBLUP (all SNPs, trained on CEU disc_train)")
    print("=" * 60)

    baseline_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(
        X_disc_train, y_disc_train
    )

    y_pred_baseline_disc_val        = baseline_model.predict(X_disc_val)
    y_pred_baseline_target_train    = baseline_model.predict(X_target_train)
    y_pred_baseline_target_held_out = baseline_model.predict(X_target_held_out)

    r2_baseline_disc_val        = r2_score(y_disc_val,        y_pred_baseline_disc_val)
    r2_baseline_target_train    = r2_score(y_target_train,    y_pred_baseline_target_train)
    r2_baseline_target_held_out = r2_score(y_target_held_out, y_pred_baseline_target_held_out)

    print(f"Baseline R² — discovery_validation (CEU within-pop) : {r2_baseline_disc_val:.4f}")
    print(f"Baseline R² — target_train         (YRI cross-pop)  : {r2_baseline_target_train:.4f}")
    print(f"Baseline R² — target_held_out      (YRI cross-pop)  : {r2_baseline_target_held_out:.4f}")
    print()

    # -------------------------------------------------
    # Summary table
    # -------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    col1, col2, col3, col4 = 32, 14, 16, 18
    header = (f"{'Model':<{col1}} {'disc_val (CEU)':>{col2}} "
              f"{'tgt_train (YRI)':>{col3}} {'tgt_held_out (YRI)':>{col4}}")
    print(header)
    print("-" * len(header))
    print(
        f"{'Oracle (causal SNPs)':<{col1}} "
        f"{r2_oracle_disc_val:>{col2}.4f} "
        f"{r2_oracle_target_train:>{col3}.4f} "
        f"{r2_oracle_target_held_out:>{col4}.4f}"
    )
    print(
        f"{'Baseline / gBLUP (all SNPs)':<{col1}} "
        f"{r2_baseline_disc_val:>{col2}.4f} "
        f"{r2_baseline_target_train:>{col3}.4f} "
        f"{r2_baseline_target_held_out:>{col4}.4f}"
    )
    print()
    print("XPopVAE target: beat baseline R² on target_held_out (YRI)")
    print()

    # -------------------------------------------------
    # Scatter plots
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    panels = [
        (axes[0, 0], y_disc_val,        y_pred_oracle_disc_val,        r2_oracle_disc_val,        "Oracle — disc_val (CEU)"),
        (axes[0, 1], y_target_train,    y_pred_oracle_target_train,    r2_oracle_target_train,    "Oracle — target_train (YRI)"),
        (axes[0, 2], y_target_held_out, y_pred_oracle_target_held_out, r2_oracle_target_held_out, "Oracle — target_held_out (YRI)"),
        (axes[1, 0], y_disc_val,        y_pred_baseline_disc_val,        r2_baseline_disc_val,        "gBLUP — disc_val (CEU)"),
        (axes[1, 1], y_target_train,    y_pred_baseline_target_train,    r2_baseline_target_train,    "gBLUP — target_train (YRI)"),
        (axes[1, 2], y_target_held_out, y_pred_baseline_target_held_out, r2_baseline_target_held_out, "gBLUP — target_held_out (YRI)"),
    ]

    for ax, y_true, y_pred, r2, title in panels:
        ax.scatter(y_true, y_pred, alpha=0.4, s=10)
        ax.set_xlabel("True Phenotype")
        ax.set_ylabel("Predicted Phenotype")
        ax.set_title(f"{title}\n$R^2$ = {r2:.4f}")

    plt.tight_layout()
    plt.savefig(output_path / "oracle_baseline_scatter.png", dpi=150)
    plt.close()
    print("Saved scatter plots → oracle_baseline_scatter.png")

    # -------------------------------------------------
    # Phenotype histograms
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(y_disc_train.ravel(),      bins=50, alpha=0.5, label="discovery_train (CEU)")
    plt.hist(y_disc_val.ravel(),        bins=50, alpha=0.5, label="discovery_validation (CEU)")
    plt.hist(y_target_train.ravel(),    bins=50, alpha=0.5, label="target_train (YRI)")
    plt.hist(y_target_held_out.ravel(), bins=50, alpha=0.5, label="target_held_out (YRI)")
    plt.title("Histogram of Simulated Phenotype")
    plt.xlabel("Phenotype Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "simulated_phenotype_histogram.png", dpi=150)
    plt.close()

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------
    print(f"Simulated phenotype: {num_causal} causal SNPs, heritability={heritability}")
    print(f"Actual causal SNPs used: {len(causal_snps)}")
    print()
    for name, y in [
        ("discovery_train",      y_disc_train),
        ("discovery_validation", y_disc_val),
        ("target_train",         y_target_train),
        ("target_held_out",      y_target_held_out),
    ]:
        print(f"  {name:<25} shape={y.shape}  mean={y.mean():.4f}  std={y.std():.4f}")
    print()

    disc_train_h2 = np.var(g_disc_train, ddof=0) / np.var(g_disc_train + e_disc_train, ddof=0)
    disc_val_h2   = np.var(g_disc_val,   ddof=0) / np.var(g_disc_val   + e_disc_val,   ddof=0)
    print(f"Empirical heritability — discovery_train : {disc_train_h2:.4f}")
    print(f"Empirical heritability — discovery_val   : {disc_val_h2:.4f}")
    print(f"Noise SD: {architecture['noise_sd']:.6f}")
    print()

    # -------------------------------------------------
    # Save phenotypes
    # -------------------------------------------------
    np.save(output_path / "simulated_phenotype_disc_train.npy",      y_disc_train)
    np.save(output_path / "simulated_phenotype_disc_val.npy",        y_disc_val)
    np.save(output_path / "simulated_phenotype_target_train.npy",    y_target_train)
    np.save(output_path / "simulated_phenotype_target_held_out.npy", y_target_held_out)

    np.save(output_path / "causal_snps.npy",  causal_snps)
    np.save(output_path / "effect_sizes.npy", betas)

    # genetic components + noise
    np.save(output_path / "disc_train_genetic_component.npy",      g_disc_train)
    np.save(output_path / "disc_val_genetic_component.npy",        g_disc_val)
    np.save(output_path / "target_train_genetic_component.npy",    g_target_train)
    np.save(output_path / "target_held_out_genetic_component.npy", g_target_held_out)

    np.save(output_path / "disc_train_noise.npy",      e_disc_train)
    np.save(output_path / "disc_val_noise.npy",        e_disc_val)
    np.save(output_path / "target_train_noise.npy",    e_target_train)
    np.save(output_path / "target_held_out_noise.npy", e_target_held_out)

    # standardization params (CEU)
    np.save(output_path / "train_standardization_mean.npy", architecture["mean"])
    np.save(output_path / "train_standardization_std.npy",  architecture["std"])

    # oracle predictions
    np.save(output_path / "pred_oracle_disc_val.npy",        y_pred_oracle_disc_val)
    np.save(output_path / "pred_oracle_target_train.npy",    y_pred_oracle_target_train)
    np.save(output_path / "pred_oracle_target_held_out.npy", y_pred_oracle_target_held_out)

    # baseline / gBLUP predictions
    np.save(output_path / "pred_baseline_disc_val.npy",        y_pred_baseline_disc_val)
    np.save(output_path / "pred_baseline_target_train.npy",    y_pred_baseline_target_train)
    np.save(output_path / "pred_baseline_target_held_out.npy", y_pred_baseline_target_held_out)

    print(f"Saved all outputs → {output_path.resolve()}")


if __name__ == "__main__":
    main()