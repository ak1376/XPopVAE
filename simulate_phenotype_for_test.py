# simulate_phenotype_for_test.py

'''
This script samples causal SNPs from the genomic window using the TRAIN set,
fits a phenotype architecture on TRAIN, and then applies that same architecture
to both TRAIN and VAL.

Phenotype model:
    y = G + e
where
    G = X_beta
and noise variance is chosen to approximately match the requested heritability.
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
    X_train = np.load(train_path)
    X_val = np.load(val_path)
    X_target = np.load(target_path)

    num_causal = 100
    heritability = 0.7
    seed = 295
    standardize = True

    print("Loaded genotype matrices")
    print(f"Train genotype shape: {X_train.shape}")
    print(f"Val genotype shape:   {X_val.shape}")
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
    # apply same architecture to TRAIN and VAL
    # -------------------------------------------------
    y_train, g_train, e_train = apply_phenotype_architecture(
        X_train,
        architecture,
        seed=seed,
    )

    y_val, g_val, e_val = apply_phenotype_architecture(
        X_val,
        architecture,
        seed=seed + 1,
    )

    y_target, g_target, e_target = apply_phenotype_architecture(
        X_target,
        architecture,
        seed=seed + 2,
    )

    # reshape to (N, 1) if you want to save that way
    # y_train = y_train.reshape(-1, 1)
    # y_val = y_val.reshape(-1, 1)
    # y_target = y_target.reshape(-1, 1)


    # Let's do a linear regression to see if everything is okay -- that we can predict the validation phenotype from the causal SNPs
    X_train_causal = X_train[:, causal_snps]
    X_val_causal = X_val[:, causal_snps]

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train_causal, y_train)

    from sklearn.metrics import r2_score
    y_pred_val = model.predict(X_val_causal)

    r2 = r2_score(y_val, y_pred_val)
    print("Validation R^2:", r2)

    # Plot a scatterplot of the validation predictions, with the correlation coefficient in the title
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_pred_val, alpha=0.5)
    plt.xlabel("True Validation Phenotype")
    plt.ylabel("Predicted Validation Phenotype")
    plt.title(f"Validation R^2: {r2:.3f}")
    plt.tight_layout()
    plt.savefig(output_path / "validation_scatter.png")
    plt.close()

    # -------------------------------------------------
    # debugging print statements
    # -------------------------------------------------
    print(f"Simulated phenotype with requested {num_causal} causal SNPs and heritability {heritability}")
    print()
    print(f"Actual number of causal SNPs used: {len(causal_snps)}")
    print()
    print(f"Causal SNP indices: {causal_snps}")
    print()
    print(f"Effect sizes: {betas}")
    print()
    print(f"Train Phenotype Shape: {y_train.shape}, Val Phenotype Shape: {y_val.shape}")
    print()
    print(f"Train Phenotype Mean: {y_train.mean()}, Train Phenotype Std: {y_train.std()}")
    print(f"Val Phenotype Mean: {y_val.mean()}, Val Phenotype Std: {y_val.std()}")
    print()
    print(f"Train Genetic Component Mean: {g_train.mean()}, Train Genetic Component Std: {g_train.std()}")
    print(f"Val Genetic Component Mean: {g_val.mean()}, Val Genetic Component Std: {g_val.std()}")
    print()
    print(f"Train Environmental Noise Mean: {e_train.mean()}, Train Environmental Noise Std: {e_train.std()}")
    print(f"Val Environmental Noise Mean: {e_val.mean()}, Val Environmental Noise Std: {e_val.std()}")
    print()
    print(f"Noise SD used for both TRAIN and VAL: {architecture['noise_sd']}")
    print()

    # empirical heritability estimates
    train_h2_empirical = np.var(g_train, ddof=0) / np.var(g_train + e_train, ddof=0)
    val_h2_empirical = np.var(g_val, ddof=0) / np.var(g_val + e_val, ddof=0)

    print(f"Empirical TRAIN heritability: {train_h2_empirical}")
    print(f"Empirical VAL heritability: {val_h2_empirical}")
    print()

    # -------------------------------------------------
    # histogram
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(y_train.ravel(), bins=50, alpha=0.6, label="Train")
    plt.hist(y_val.ravel(), bins=50, alpha=0.6, label="Val")
    plt.title("Histogram of Simulated Phenotype")
    plt.xlabel("Phenotype Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "simulated_phenotype_histogram.png")
    plt.close()

    # -------------------------------------------------
    # save outputs
    # -------------------------------------------------
    np.save(output_path / "simulated_phenotype_train.npy", y_train)
    np.save(output_path / "simulated_phenotype_val.npy", y_val)
    np.save(output_path / "simulated_phenotype_target.npy", y_target)
    np.save(output_path / "causal_snps.npy", causal_snps)
    np.save(output_path / "effect_sizes.npy", betas)

    # useful extras for debugging/reproducibility
    np.save(output_path / "train_genetic_component.npy", g_train)
    np.save(output_path / "val_genetic_component.npy", g_val)
    np.save(output_path / "train_noise.npy", e_train)
    np.save(output_path / "val_noise.npy", e_val)
    np.save(output_path / "train_standardization_mean.npy", architecture["mean"])
    np.save(output_path / "train_standardization_std.npy", architecture["std"])

    print(f"Saved outputs to: {output_path.resolve()}")


if __name__ == "__main__":
    main()