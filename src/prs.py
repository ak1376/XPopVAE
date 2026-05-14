from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import calculate_normalization_statistics, normalize, plot_prs_scatter
from scipy.optimize import minimize
from scipy.linalg import eigh

'''
This script will do PRS models. Here are the following models
    - Traditional PRS: this will take the beta effect sizes from the gwas step and fit a linear regression on them
    - gBLUP: Operating in individual-level space rather than in SNP space (which is what the traditional PRS does)
'''


def kinship_matrix(G, mean, std):
    G_norm = normalize(G, mean, std)
    p = G_norm.shape[1]
    return (G_norm @ G_norm.T) / p


def estimate_variance_components(K: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    REML estimation of sigma2_g and sigma2_e.
    Model: y = mu + u + e
           u ~ MVN(0, sigma2_g * K)
           e ~ MVN(0, sigma2_e * I)
    """
    n = len(y)
    y = y.astype(float)

    # eigendecomposition of K — done once, reused every likelihood eval
    S, Q = eigh(K)       # S: (n,) eigenvalues, Q: (n,n) eigenvectors
    Qty = Q.T @ y        # project y into eigenspace

    # project out the intercept for REML
    # X = 1_n (intercept only), so QtX = Q.T @ 1_n
    QtX = Q.T @ np.ones(n)

    def neg_reml_loglik(log_params):
        sigma2_g, sigma2_e = np.exp(log_params)  # log-space ensures positivity

        # eigenvalues of V = sigma2_g * K + sigma2_e * I
        d = sigma2_g * S + sigma2_e              # (n,)

        # -0.5 * log|V|
        term1 = 0.5 * np.sum(np.log(d))

        # -0.5 * log|XᵀV⁻¹X|  (scalar since X is intercept only)
        XtVinvX = np.sum(QtX**2 / d)
        term2 = 0.5 * np.log(XtVinvX)

        # -0.5 * yᵀPy  where Py = V⁻¹y - V⁻¹X(XᵀV⁻¹X)⁻¹XᵀV⁻¹y
        Vinv_Qty = Qty / d
        Vinv_QtX = QtX / d
        PyQ      = Vinv_Qty - Vinv_QtX * (np.sum(Vinv_QtX * Qty) / XtVinvX)
        term3    = 0.5 * np.sum(Qty * PyQ)

        return term1 + term2 + term3

    # optimize in log-space — initialize at sigma2_g = sigma2_e = var(y)/2
    init = np.log([y.var() / 2, y.var() / 2])
    res  = minimize(neg_reml_loglik, x0=init, method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000})

    sigma2_g, sigma2_e = np.exp(res.x)
    lam = sigma2_e / sigma2_g
    h2  = sigma2_g / (sigma2_g + sigma2_e)

    print(f"[gBLUP] sigma2_g={sigma2_g:.4f}  sigma2_e={sigma2_e:.4f}  h2={h2:.4f}  lambda={lam:.4f}")
    print(f"[gBLUP] optimizer converged: {res.success}  message: {res.message}")

    return sigma2_g, sigma2_e, lam

def cross_kinship_matrix(G_test, G_train, mean, std):
    """
    Genetic similarity between test and train individuals.
    Shape: (n_test, n_train)
    """
    G_test_norm  = normalize(G_test,  mean, std)
    G_train_norm = normalize(G_train, mean, std)
    p = G_train_norm.shape[1]
    return (G_test_norm @ G_train_norm.T) / p


def gblup_fit(K_train, y_train, lam):
    """
    Fit gBLUP — compute alpha weights over training individuals.
    alpha = (K_train + lambda * I)^{-1} y_train
    """
    n = K_train.shape[0]
    V = K_train + lam * np.eye(n)
    alpha = np.linalg.solve(V, y_train)
    return alpha


def gblup_predict(K_cross, alpha):
    """
    Predict phenotypes for test individuals.
    y_hat = K_cross @ alpha
    """
    return K_cross @ alpha


def run_gblup(
    disc_train_X: np.ndarray,
    disc_train_y: np.ndarray,
    disc_val_X: np.ndarray,
    disc_val_y: np.ndarray,
    target_X: np.ndarray,
    target_y: np.ndarray,
    out_dir: Path | None = None,
) -> dict:
    """
    End-to-end gBLUP pipeline: normalize, build GRMs, REML, fit, predict, evaluate.

    Returns a metrics dict with val_r2 and test_r2.
    """
    from sklearn.metrics import r2_score

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── normalize ─────────────────────────────────────────────────────────────
    geno_mean, geno_std   = calculate_normalization_statistics(disc_train_X)
    pheno_mean, pheno_std = calculate_normalization_statistics(disc_train_y)

    disc_train_y_norm = normalize(disc_train_y, pheno_mean, pheno_std)
    disc_val_y_norm   = normalize(disc_val_y,   pheno_mean, pheno_std)
    target_y_norm     = normalize(target_y,     pheno_mean, pheno_std)

    # ── GRMs ──────────────────────────────────────────────────────────────────
    K_train        = kinship_matrix(disc_train_X, geno_mean, geno_std)
    K_cross_val    = cross_kinship_matrix(disc_val_X, disc_train_X, geno_mean, geno_std)
    K_cross_target = cross_kinship_matrix(target_X,   disc_train_X, geno_mean, geno_std)

    # ── REML ──────────────────────────────────────────────────────────────────
    sigma2_g, sigma2_e, lam = estimate_variance_components(K_train, disc_train_y_norm)

    # ── fit + predict ─────────────────────────────────────────────────────────
    alpha = gblup_fit(K_train, disc_train_y_norm, lam)

    y_hat_val    = gblup_predict(K_cross_val,    alpha)
    y_hat_target = gblup_predict(K_cross_target, alpha)

    val_r2  = float(r2_score(disc_val_y_norm, y_hat_val))
    test_r2 = float(r2_score(target_y_norm,   y_hat_target))

    print(f"[gBLUP] val R²={val_r2:.4f}  test R²={test_r2:.4f}")

    # ── scatter plots ─────────────────────────────────────────────────────────
    if out_dir is not None:
        for split, y_true, y_pred, r2 in [
            ("val",  disc_val_y_norm, y_hat_val,    val_r2),
            ("test", target_y_norm,   y_hat_target, test_r2),
        ]:
            plot_prs_scatter(y_true, y_pred, split, r2, out_dir / f"gblup_{split}_scatter.png")

        np.save(out_dir / "gblup_alpha.npy",          alpha)
        np.save(out_dir / "y_hat_val.npy",             y_hat_val)
        np.save(out_dir / "y_hat_target.npy",          y_hat_target)
        np.save(out_dir / "y_true_val_norm.npy",       disc_val_y_norm)
        np.save(out_dir / "y_true_target_norm.npy",    target_y_norm)
        np.save(out_dir / "K_train.npy",               K_train)
        np.save(out_dir / "K_cross_target_train.npy",  K_cross_target)
        print(f"[gBLUP] Arrays saved to {out_dir}")

    return {"val_r2": val_r2, "test_r2": test_r2, "sigma2_g": float(sigma2_g),
            "sigma2_e": float(sigma2_e), "lambda": float(lam)}


def run_standard_prs(
    disc_train_X: np.ndarray,
    disc_train_y: np.ndarray,
    disc_val_X: np.ndarray,
    disc_val_y: np.ndarray,
    target_X: np.ndarray,
    target_y: np.ndarray,
    beta: np.ndarray,
    out_dir: Path | None = None,
) -> dict:
    """
    Standard PRS: apply pre-computed GWAS beta weights to normalized genotypes.

    Marginal OLS scores (X @ beta) are not calibrated to the phenotype scale, so
    we evaluate with Pearson r² (correlation-based, scale-invariant) and
    standardize the raw scores before plotting/saving.

    Normalization statistics are re-derived from disc_train_X so they match
    exactly what was used when the betas were estimated.
    """
    from scipy.stats import pearsonr

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    geno_mean, geno_std   = calculate_normalization_statistics(disc_train_X)
    pheno_mean, pheno_std = calculate_normalization_statistics(disc_train_y)

    X_val_norm    = normalize(disc_val_X, geno_mean, geno_std)
    X_target_norm = normalize(target_X,   geno_mean, geno_std)

    disc_val_y_norm = normalize(disc_val_y, pheno_mean, pheno_std)
    target_y_norm   = normalize(target_y,   pheno_mean, pheno_std)

    # raw PRS — uncalibrated scale
    prs_val    = X_val_norm    @ beta
    prs_target = X_target_norm @ beta

    # standardize so scatter plots are on the same scale as the phenotype
    prs_val_std    = (prs_val    - prs_val.mean())    / (prs_val.std()    + 1e-8)
    prs_target_std = (prs_target - prs_target.mean()) / (prs_target.std() + 1e-8)

    # correlation-based R² — the standard metric for PRS
    val_r2  = float(pearsonr(prs_val,    disc_val_y_norm)[0] ** 2)
    test_r2 = float(pearsonr(prs_target, target_y_norm)[0]   ** 2)

    print(f"[Standard PRS] val R²={val_r2:.4f}  test R²={test_r2:.4f}")

    if out_dir is not None:
        for split, y_true, y_pred_std, r2 in [
            ("val",  disc_val_y_norm, prs_val_std,    val_r2),
            ("test", target_y_norm,   prs_target_std, test_r2),
        ]:
            plot_prs_scatter(
                y_true, y_pred_std, split, r2,
                out_dir / f"standard_prs_{split}_scatter.png",
                model_name="Standard PRS",
            )

        np.save(out_dir / "prs_val_raw.npy",         prs_val)
        np.save(out_dir / "prs_target_raw.npy",       prs_target)
        np.save(out_dir / "prs_val_std.npy",          prs_val_std)
        np.save(out_dir / "prs_target_std.npy",       prs_target_std)
        np.save(out_dir / "y_true_val_norm.npy",      disc_val_y_norm)
        np.save(out_dir / "y_true_target_norm.npy",   target_y_norm)
        print(f"[Standard PRS] Arrays saved to {out_dir}")

    return {"val_r2": val_r2, "test_r2": test_r2}


if __name__ == "__main__":
    pass