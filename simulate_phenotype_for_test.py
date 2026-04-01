# simulate_phenotype_for_test.py

'''
This script samples causal SNPs from the genomic window using the TRAIN set,
fits a phenotype architecture on TRAIN, and then applies that same architecture
to TRAIN, VAL, and TARGET.

Phenotype model:
    y = G + e
where
    G = X_beta
and noise variance is chosen to approximately match the requested heritability.

Additionally computes:
    - Oracle R² (Ridge on causal SNPs only) -- upper bound
    - Baseline R² (gBLUP / Ridge on all SNPs) -- cross-population baseline to beat
Both evaluated on within-population VAL and cross-population TARGET.
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import os


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
        architecture: dict containing
            causal_snps
            effect_sizes
            mean
            std
            noise_sd
            heritability
            standardize
        genetic_component_train: ndarray of shape (n_individuals,)
    '''
    rng = np.random.default_rng(seed)

    n_individuals, n_snps = genotype_matrix.shape

    # -------------------------------------------------
    # choose causal SNPs from polymorphic SNPs in TRAIN
    # -------------------------------------------------
    maf = genotype_matrix.mean(axis=0) / 2.0
    polymorphic = np.where((maf > 0) & (maf < 1))[0]

    if num_causal > len(polymorphic):
        raise ValueError(
            f"Requested {num_causal} causal SNPs but only {len(polymorphic)} "
            f"polymorphic SNPs are available."
        )

    causal_snps = rng.choice(polymorphic, size=num_causal, replace=False)

    # -------------------------------------------------
    # extract causal genotypes
    # -------------------------------------------------
    X = genotype_matrix[:, causal_snps].astype(float)

    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)

        keep = std > 0
        X = X[:, keep]
        causal_snps = causal_snps[keep]
        mean = mean[keep]
        std = std[keep]

        if X.shape[1] == 0:
            raise ValueError("All selected causal SNPs became non-variable after filtering.")

        X = (X - mean) / std
    else:
        mean = np.zeros(X.shape[1], dtype=float)
        std = np.ones(X.shape[1], dtype=float)

    # -------------------------------------------------
    # simulate effect sizes
    # -------------------------------------------------
    effect_sizes = rng.normal(loc=0.0, scale=1.0, size=X.shape[1])

    # -------------------------------------------------
    # compute genetic component on TRAIN
    # -------------------------------------------------
    genetic_component_train = X @ effect_sizes
    var_g = np.var(genetic_component_train, ddof=0)

    if var_g == 0:
        raise ValueError("Genetic variance is zero on TRAIN.")

    # -------------------------------------------------
    # choose environmental noise to match heritability
    # -------------------------------------------------
    var_e = var_g * (1.0 - heritability) / heritability
    noise_sd = np.sqrt(var_e)

    architecture = {
        "causal_snps": causal_snps,
        "effect_sizes": effect_sizes,
        "mean": mean,
        "std": std,
        "noise_sd": noise_sd,
        "heritability": heritability,
        "standardize": standardize,
    }

    return architecture, genetic_component_train


def apply_phenotype_architecture(genotype_matrix, architecture, seed=0):
    '''
    Apply a pre-fit phenotype architecture to a new genotype matrix.

    Returns:
        phenotype: ndarray of shape (n_individuals,)
        genetic_component: ndarray of shape (n_individuals,)
        environmental_noise: ndarray of shape (n_individuals,)
    '''
    rng = np.random.default_rng(seed)

    causal_snps = architecture["causal_snps"]
    effect_sizes = architecture["effect_sizes"]
    mean = architecture["mean"]
    std = architecture["std"]
    noise_sd = architecture["noise_sd"]
    standardize = architecture["standardize"]

    X = genotype_matrix[:, causal_snps].astype(float)

    if standardize:
        X = (X - mean) / std

    genetic_component = X @ effect_sizes
    environmental_noise = rng.normal(loc=0.0, scale=noise_sd, size=genotype_matrix.shape[0])
    phenotype = genetic_component + environmental_noise

    return phenotype, genetic_component, environmental_noise


def main():
    # -------------------------------------------------
    # paths
    # -------------------------------------------------
    output_path = Path("phenotype_creation")
    os.makedirs(output_path, exist_ok=True)

    train_path = Path(
        "/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_train.npy"
    )
    val_path = Path(
        "/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_val.npy"
    )
    target_path = Path(
        "/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/target.npy"
    )

    # -------------------------------------------------
    # load genotypes
    # -------------------------------------------------
    X_train  = np.load(train_path)
    X_val    = np.load(val_path)
    X_target = np.load(target_path)

    num_causal  = 100
    heritability = 1.0
    seed        = 295
    standardize = True

    print("Loaded genotype matrices")
    print(f"Train genotype shape:  {X_train.shape}")
    print(f"Val genotype shape:    {X_val.shape}")
    print(f"Target genotype shape: {X_target.shape}")
    print()

    # -------------------------------------------------
    # fit architecture on TRAIN only
    # -------------------------------------------------
    architecture, g_train_no_noise = fit_phenotype_architecture(
        X_train,
        num_causal=num_causal,
        heritability=heritability,
        seed=seed,
        standardize=standardize,
    )

    causal_snps = architecture["causal_snps"]
    betas = architecture["effect_sizes"]

    # -------------------------------------------------
    # apply same architecture to TRAIN, VAL, TARGET
    # -------------------------------------------------
    y_train,  g_train,  e_train  = apply_phenotype_architecture(X_train,  architecture, seed=seed)
    y_val,    g_val,    e_val    = apply_phenotype_architecture(X_val,    architecture, seed=seed + 1)
    y_target, g_target, e_target = apply_phenotype_architecture(X_target, architecture, seed=seed + 2)

    # -------------------------------------------------
    # Oracle: Ridge on causal SNPs only
    # Upper bound -- assumes you know exactly which SNPs are causal
    # -------------------------------------------------
    print("=" * 60)
    print("ORACLE (causal SNPs only)")
    print("=" * 60)

    X_train_causal  = X_train[:, causal_snps]
    X_val_causal    = X_val[:, causal_snps]
    X_target_causal = X_target[:, causal_snps]

    oracle_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(X_train_causal, y_train)

    y_pred_oracle_val    = oracle_model.predict(X_val_causal)
    y_pred_oracle_target = oracle_model.predict(X_target_causal)

    r2_oracle_val    = r2_score(y_val,    y_pred_oracle_val)
    r2_oracle_target = r2_score(y_target, y_pred_oracle_target)

    print(f"Oracle R² on val    (within-pop  CEU): {r2_oracle_val:.4f}")
    print(f"Oracle R² on target (cross-pop   YRI): {r2_oracle_target:.4f}")
    print()

    # -------------------------------------------------
    # Baseline: Ridge on all SNPs (gBLUP equivalent)
    # This is the cross-population baseline XPopVAE needs to beat
    # -------------------------------------------------
    print("=" * 60)
    print("BASELINE / gBLUP (all SNPs)")
    print("=" * 60)

    baseline_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(X_train, y_train)

    y_pred_baseline_val    = baseline_model.predict(X_val)
    y_pred_baseline_target = baseline_model.predict(X_target)

    r2_baseline_val    = r2_score(y_val,    y_pred_baseline_val)
    r2_baseline_target = r2_score(y_target, y_pred_baseline_target)

    print(f"Baseline R² on val    (within-pop CEU): {r2_baseline_val:.4f}")
    print(f"Baseline R² on target (cross-pop  YRI): {r2_baseline_target:.4f}")
    print()

    # -------------------------------------------------
    # summary table
    # -------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Val (CEU)':>12} {'Target (YRI)':>14}")
    print(f"{'-'*30} {'-'*12} {'-'*14}")
    print(f"{'Oracle (causal SNPs)':<30} {r2_oracle_val:>12.4f} {r2_oracle_target:>14.4f}")
    print(f"{'Baseline / gBLUP (all SNPs)':<30} {r2_baseline_val:>12.4f} {r2_baseline_target:>14.4f}")
    print()
    print("XPopVAE target: beat baseline R² on target (YRI)")
    print()

    # -------------------------------------------------
    # scatter plots: oracle and baseline, val and target
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        (axes[0, 0], y_val,    y_pred_oracle_val,    r2_oracle_val,    "Oracle — Val (CEU)"),
        (axes[0, 1], y_target, y_pred_oracle_target, r2_oracle_target, "Oracle — Target (YRI)"),
        (axes[1, 0], y_val,    y_pred_baseline_val,    r2_baseline_val,    "gBLUP — Val (CEU)"),
        (axes[1, 1], y_target, y_pred_baseline_target, r2_baseline_target, "gBLUP — Target (YRI)"),
    ]

    for ax, y_true, y_pred, r2, title in panels:
        ax.scatter(y_true, y_pred, alpha=0.4, s=10)
        ax.set_xlabel("True Phenotype")
        ax.set_ylabel("Predicted Phenotype")
        ax.set_title(f"{title}\n$R^2$ = {r2:.4f}")

    plt.tight_layout()
    plt.savefig(output_path / "oracle_baseline_scatter.png", dpi=150)
    plt.close()
    print("Saved scatter plots to oracle_baseline_scatter.png")

    # -------------------------------------------------
    # phenotype histograms
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(y_train.ravel(),  bins=50, alpha=0.5, label="Train (CEU)")
    plt.hist(y_val.ravel(),    bins=50, alpha=0.5, label="Val (CEU)")
    plt.hist(y_target.ravel(), bins=50, alpha=0.5, label="Target (YRI)")
    plt.title("Histogram of Simulated Phenotype")
    plt.xlabel("Phenotype Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "simulated_phenotype_histogram.png", dpi=150)
    plt.close()

    # -------------------------------------------------
    # debugging print statements
    # -------------------------------------------------
    print()
    print(f"Simulated phenotype: {num_causal} causal SNPs, heritability={heritability}")
    print(f"Actual causal SNPs used: {len(causal_snps)}")
    print()
    print(f"Train shape: {y_train.shape}  |  Val shape: {y_val.shape}  |  Target shape: {y_target.shape}")
    print()
    print(f"Train  — mean: {y_train.mean():.4f},  std: {y_train.std():.4f}")
    print(f"Val    — mean: {y_val.mean():.4f},  std: {y_val.std():.4f}")
    print(f"Target — mean: {y_target.mean():.4f},  std: {y_target.std():.4f}")
    print()

    train_h2 = np.var(g_train, ddof=0) / np.var(g_train + e_train, ddof=0)
    val_h2   = np.var(g_val,   ddof=0) / np.var(g_val   + e_val,   ddof=0)
    print(f"Empirical heritability — Train: {train_h2:.4f},  Val: {val_h2:.4f}")
    print(f"Noise SD: {architecture['noise_sd']:.6f}")
    print()

    # -------------------------------------------------
    # save phenotypes
    # -------------------------------------------------
    np.save(output_path / "simulated_phenotype_train.npy",  y_train)
    np.save(output_path / "simulated_phenotype_val.npy",    y_val)
    np.save(output_path / "simulated_phenotype_target.npy", y_target)
    np.save(output_path / "causal_snps.npy",                causal_snps)
    np.save(output_path / "effect_sizes.npy",               betas)

    # genetic components and noise
    np.save(output_path / "train_genetic_component.npy", g_train)
    np.save(output_path / "val_genetic_component.npy",   g_val)
    np.save(output_path / "train_noise.npy",             e_train)
    np.save(output_path / "val_noise.npy",               e_val)

    # standardization params
    np.save(output_path / "train_standardization_mean.npy", architecture["mean"])
    np.save(output_path / "train_standardization_std.npy",  architecture["std"])

    # oracle predictions
    np.save(output_path / "pred_oracle_val.npy",    y_pred_oracle_val)
    np.save(output_path / "pred_oracle_target.npy", y_pred_oracle_target)

    # baseline / gBLUP predictions
    np.save(output_path / "pred_baseline_val.npy",    y_pred_baseline_val)
    np.save(output_path / "pred_baseline_target.npy", y_pred_baseline_target)

    print(f"Saved all outputs to: {output_path.resolve()}")


if __name__ == "__main__":
    main()