# simulate_phenotype_for_test.py

'''
This script samples causal SNPs from the genomic window using the TRAIN set,
fits a phenotype architecture on TRAIN, and then applies that same architecture
to train_discovery, validation_discovery, train_target, and test_target.

Phenotype model:
    y = G + e
where
    G = X_beta
and noise variance is chosen to approximately match the requested heritability.

Architecture is always fit on discovery_train (CEU) only.
The same causal SNPs, effect sizes, and standardization are then applied to
all other splits — including target individuals — so phenotypes are on a
consistent scale.
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
            causal_snps, effect_sizes, mean, std, noise_sd,
            heritability, standardize
        genetic_component_train: ndarray of shape (n_individuals,)
    '''
    rng = np.random.default_rng(seed)

    n_individuals, n_snps = genotype_matrix.shape

    # choose causal SNPs from polymorphic SNPs in discovery_train only
    maf = genotype_matrix.mean(axis=0) / 2.0
    polymorphic = np.where((maf > 0) & (maf < 1))[0]

    if num_causal > len(polymorphic):
        raise ValueError(
            f"Requested {num_causal} causal SNPs but only {len(polymorphic)} "
            f"polymorphic SNPs are available."
        )

    causal_snps = rng.choice(polymorphic, size=num_causal, replace=False)

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

    effect_sizes = rng.normal(loc=0.0, scale=1.0, size=X.shape[1])

    genetic_component_train = X @ effect_sizes
    var_g = np.var(genetic_component_train, ddof=0)

    if var_g == 0:
        raise ValueError("Genetic variance is zero on discovery_train.")

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

    causal_snps  = architecture["causal_snps"]
    effect_sizes = architecture["effect_sizes"]
    mean         = architecture["mean"]
    std          = architecture["std"]
    noise_sd     = architecture["noise_sd"]
    standardize  = architecture["standardize"]

    X = genotype_matrix[:, causal_snps].astype(float)

    if standardize:
        X = (X - mean) / std

    genetic_component   = X @ effect_sizes
    environmental_noise = rng.normal(loc=0.0, scale=noise_sd, size=genotype_matrix.shape[0])
    phenotype           = genetic_component + environmental_noise

    return phenotype, genetic_component, environmental_noise


def main():
    output_path = Path("phenotype_creation")
    os.makedirs(output_path, exist_ok=True)

    base = Path("/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0")

    train_discovery_path    = base / "train_discovery.npy"
    val_discovery_path      = base / "validation_discovery.npy"
    train_target_path       = base / "train_target.npy"
    test_target_path        = base / "test_target.npy"

    X_train_discovery = np.load(train_discovery_path)
    X_val_discovery   = np.load(val_discovery_path)
    X_train_target    = np.load(train_target_path)
    X_test_target     = np.load(test_target_path)

    num_causal   = 100
    heritability = 0.7
    seed         = 295
    standardize  = True

    print("Loaded genotype matrices")
    print(f"train_discovery shape:  {X_train_discovery.shape}")
    print(f"val_discovery shape:    {X_val_discovery.shape}")
    print(f"train_target shape:     {X_train_target.shape}")
    print(f"test_target shape:      {X_test_target.shape}")
    print()

    # fit architecture on discovery_train only
    architecture, g_train_no_noise = fit_phenotype_architecture(
        X_train_discovery,
        num_causal=num_causal,
        heritability=heritability,
        seed=seed,
        standardize=standardize,
    )

    causal_snps = architecture["causal_snps"]
    betas       = architecture["effect_sizes"]

    # apply the same architecture to all splits
    y_train_discovery, g_train, e_train = apply_phenotype_architecture(
        X_train_discovery, architecture, seed=seed,
    )
    y_val_discovery, g_val, e_val = apply_phenotype_architecture(
        X_val_discovery, architecture, seed=seed + 1,
    )
    y_train_target, g_train_target, e_train_target = apply_phenotype_architecture(
        X_train_target, architecture, seed=seed + 2,
    )
    y_test_target, g_test_target, e_test_target = apply_phenotype_architecture(
        X_test_target, architecture, seed=seed + 3,
    )

    # ------------------------------------------------------------------
    # sanity check: linear regression on discovery only
    # Apply the same mean/std standardization used during phenotype
    # simulation so the regression operates on the same scale.
    # ------------------------------------------------------------------
    arch_mean = architecture["mean"]
    arch_std  = architecture["std"]

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X_train_causal       = (X_train_discovery[:, causal_snps] - arch_mean) / arch_std
    X_val_causal         = (X_val_discovery[:,   causal_snps] - arch_mean) / arch_std
    X_test_target_causal = (X_test_target[:,     causal_snps] - arch_mean) / arch_std

    reg = LinearRegression()
    reg.fit(X_train_causal, y_train_discovery)

    y_pred_val = reg.predict(X_val_causal)
    r2_val = r2_score(y_val_discovery, y_pred_val)
    print(f"Discovery val R^2 (linear regression sanity check): {r2_val:.3f}")

    # cross-population sanity check
    y_pred_test_target = reg.predict(X_test_target_causal)
    r2_test_target     = r2_score(y_test_target, y_pred_test_target)
    print(f"Test target R^2 (linear regression sanity check):   {r2_test_target:.3f}")
    print()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val_discovery, y_pred_val, alpha=0.5)
    plt.xlabel("True Discovery Val Phenotype")
    plt.ylabel("Predicted Discovery Val Phenotype")
    plt.title(f"Discovery Val R^2: {r2_val:.3f}")
    plt.tight_layout()
    plt.savefig(output_path / "validation_scatter.png")
    plt.close()

    # ------------------------------------------------------------------
    # debugging print statements
    # ------------------------------------------------------------------
    print(f"Simulated phenotype with {num_causal} causal SNPs, heritability={heritability}")
    print(f"Actual causal SNPs used: {len(causal_snps)}")
    print()
    for name, y, g, e in [
        ("train_discovery",    y_train_discovery, g_train,        e_train),
        ("val_discovery",      y_val_discovery,   g_val,          e_val),
        ("train_target",       y_train_target,    g_train_target, e_train_target),
        ("test_target",        y_test_target,     g_test_target,  e_test_target),
    ]:
        h2_emp = np.var(g, ddof=0) / np.var(g + e, ddof=0)
        print(
            f"{name:25s}  n={len(y):4d}  "
            f"mean={y.mean():.3f}  std={y.std():.3f}  "
            f"empirical_h2={h2_emp:.3f}"
        )

    print(f"\nNoise SD: {architecture['noise_sd']:.4f}")

    # ------------------------------------------------------------------
    # histogram
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(y_train_discovery.ravel(), bins=50, alpha=0.6, label="train_discovery")
    plt.hist(y_val_discovery.ravel(),   bins=50, alpha=0.6, label="val_discovery")
    plt.hist(y_train_target.ravel(),    bins=50, alpha=0.6, label="train_target")
    plt.hist(y_test_target.ravel(),     bins=50, alpha=0.6, label="test_target")
    plt.title("Histogram of Simulated Phenotype")
    plt.xlabel("Phenotype Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "simulated_phenotype_histogram.png")
    plt.close()

    # ------------------------------------------------------------------
    # save outputs
    # ------------------------------------------------------------------
    np.save(output_path / "simulated_phenotype_train_discovery.npy", y_train_discovery)
    np.save(output_path / "simulated_phenotype_val_discovery.npy",   y_val_discovery)
    np.save(output_path / "simulated_phenotype_train_target.npy",    y_train_target)
    np.save(output_path / "simulated_phenotype_test_target.npy",     y_test_target)
    np.save(output_path / "causal_snps.npy",                         causal_snps)
    np.save(output_path / "effect_sizes.npy",                        betas)

    # extras for debugging/reproducibility
    np.save(output_path / "train_discovery_genetic_component.npy", g_train)
    np.save(output_path / "val_discovery_genetic_component.npy",   g_val)
    np.save(output_path / "train_target_genetic_component.npy",    g_train_target)
    np.save(output_path / "test_target_genetic_component.npy",     g_test_target)
    np.save(output_path / "train_discovery_noise.npy",             e_train)
    np.save(output_path / "val_discovery_noise.npy",               e_val)
    np.save(output_path / "train_target_noise.npy",                e_train_target)
    np.save(output_path / "test_target_noise.npy",                 e_test_target)
    np.save(output_path / "train_standardization_mean.npy",        architecture["mean"])
    np.save(output_path / "train_standardization_std.npy",         architecture["std"])

    print(f"\nSaved outputs to: {output_path.resolve()}")


if __name__ == "__main__":
    main()