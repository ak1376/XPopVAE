"""
baseline_predictors.py

Runs baseline predictors on genotype data to predict phenotype.

Baselines:
1. Linear Regression
2. Ridge Regression
3. Lasso             — via LassoLarsCV (LARS path, much faster than coord descent for wide data)
4. gBLUP

Usage
-----
python baseline_predictors.py \
    --x_train /sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_train.npy \
    --y_train /sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_train.npy \
    --x_val   /sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_val.npy   \
    --y_val   /sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_val.npy   \
    --x_test  /sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/target.npy  \
    --y_test  /sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_target.npy \
    --out_dir results/baselines    \
    --h2      0.7
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LassoLarsCV, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# gBLUP
# ---------------------------------------------------------------------------

def build_grm(X: np.ndarray) -> np.ndarray:
    """VanRaden (2008) GRM: G = WW' / [2 * sum p(1-p)]"""
    p     = X.mean(axis=0) / 2.0
    W     = X - 2 * p
    denom = 2.0 * np.sum(p * (1.0 - p))
    return (W @ W.T) / denom


def build_cross_grm(X_new, X_train, p):
    W_new   = X_new   - 2 * p
    W_train = X_train - 2 * p
    denom   = 2.0 * np.sum(p * (1.0 - p))
    return (W_new @ W_train.T) / denom


class GBLUPPredictor:
    def __init__(self, h2=None):
        self.h2 = h2
        self.lambda_ = self.alpha_ = self.mu_ = self.p_ = self.X_train_ = None

    def _solve(self, G, y_c, lam):
        return np.linalg.solve(G + lam * np.eye(len(G)), y_c)

    def fit(self, X_train, y_train, X_val=None, y_val=None, lambdas=None):
        self.mu_      = y_train.mean()
        self.p_       = X_train.mean(axis=0) / 2.0
        self.X_train_ = X_train.copy()
        G   = build_grm(X_train)
        y_c = y_train - self.mu_

        if self.h2 is not None:
            self.lambda_ = (1.0 - self.h2) / self.h2
        else:
            if X_val is None or y_val is None:
                raise ValueError("Provide X_val/y_val when h2 is None.")
            lambdas = lambdas if lambdas is not None else np.logspace(-3, 3, 50)
            G_cv    = build_cross_grm(X_val, X_train, self.p_)
            best_r2, best_lam = -np.inf, lambdas[0]
            for lam in lambdas:
                r2 = r2_score(y_val, self.mu_ + G_cv @ self._solve(G, y_c, lam))
                if r2 > best_r2:
                    best_r2, best_lam = r2, lam
            self.lambda_ = best_lam

        self.alpha_ = self._solve(G, y_c, self.lambda_)
        return self

    def predict(self, X_new):
        return self.mu_ + build_cross_grm(X_new, self.X_train_, self.p_) @ self.alpha_


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred):
    return {
        "r2":   r2_score(y_true, y_pred),
        "mse":  mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--x_train",     required=True)
    p.add_argument("--y_train",     required=True)
    p.add_argument("--x_val",       required=True)
    p.add_argument("--y_val",       required=True)
    p.add_argument("--x_test",      required=True)
    p.add_argument("--y_test",      required=True)
    p.add_argument("--out_dir",     default="results/baselines")
    p.add_argument("--h2",          type=float, default=0.7)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--n_jobs",      type=int,   default=-1,
                   help="Parallel jobs for LassoLarsCV (-1 = all cores)")
    p.add_argument("--max_n_alphas",type=int,   default=100,
                   help="Max alphas on the LARS path (default 100; lower = faster)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Loading data...")
    X_train = np.load(args.x_train)
    y_train = np.load(args.y_train)
    X_val   = np.load(args.x_val)
    y_val   = np.load(args.y_val)
    X_test  = np.load(args.x_test)
    y_test  = np.load(args.y_test)
    print(f"  Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")

    if args.standardize:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
        print("  SNPs standardized.")

    # Train+val for CV-based methods
    X_tv = np.concatenate([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "BASELINE PREDICTOR RESULTS",
        "=" * 60,
        f"Train N  : {X_train.shape[0]}",
        f"Val   N  : {X_val.shape[0]}",
        f"Test  N  : {X_test.shape[0]}",
        f"SNPs     : {X_train.shape[1]}",
        f"h2       : {args.h2}",
        f"Seed     : {args.seed}",
        f"Std      : {args.standardize}",
        "",
    ]

    def log(msg=""):
        print(msg)
        lines.append(msg)

    def report(name, val_m, test_m, extra=""):
        log(f"--- {name} ---")
        if extra:
            log(f"  {extra}")
        log(f"  Val  R2={val_m['r2']:.4f}  RMSE={val_m['rmse']:.4f}  MSE={val_m['mse']:.4f}")
        log(f"  Test R2={test_m['r2']:.4f}  RMSE={test_m['rmse']:.4f}  MSE={test_m['mse']:.4f}")
        log()

    # -------------------------------------------------------------------------
    # 1. Linear Regression
    # -------------------------------------------------------------------------
    log("Fitting Linear Regression...")
    lr = LinearRegression().fit(X_train, y_train)
    report("Linear Regression",
           evaluate(y_val,  lr.predict(X_val)),
           evaluate(y_test, lr.predict(X_test)))

    # -------------------------------------------------------------------------
    # 2. Ridge
    # -------------------------------------------------------------------------
    log("Fitting Ridge Regression...")
    best_r2, best_a = -np.inf, 1.0
    for a in np.logspace(-3, 6, 80):
        r2 = r2_score(y_val, Ridge(alpha=a).fit(X_train, y_train).predict(X_val))
        if r2 > best_r2:
            best_r2, best_a = r2, a

    ridge = Ridge(alpha=best_a).fit(X_train, y_train)
    report("Ridge Regression",
           evaluate(y_val,  ridge.predict(X_val)),
           evaluate(y_test, ridge.predict(X_test)),
           extra=f"Best alpha={best_a:.4g}")

    # -------------------------------------------------------------------------
    # 3. Lasso via LassoLarsCV
    #
    # WHY LARS INSTEAD OF COORDINATE DESCENT?
    # ----------------------------------------
    # Coordinate descent (sklearn's Lasso) iterates over all M=9208 SNPs
    # repeatedly until convergence, for each alpha on the grid.
    #
    # LARS (Least Angle Regression) builds the ENTIRE regularization path
    # in one shot by stepping through variables one at a time. For sparse
    # solutions (only 100 causal SNPs), the path terminates after ~100 steps
    # rather than sweeping the full grid. This is O(N * k^2) where k is the
    # number of selected variables, vs O(N * M * n_alphas) for coord descent.
    #
    # LassoLarsCV runs this fast path inside CV folds.
    # max_n_alphas caps the path length (100 is fine given 100 causal SNPs).
    # n_jobs parallelises the CV folds.
    # -------------------------------------------------------------------------
    log("Fitting Lasso via LassoLarsCV...")
    llcv = LassoLarsCV(
        cv           = 5,
        max_n_alphas = args.max_n_alphas,
        n_jobs       = args.n_jobs,
    ).fit(X_tv, y_tv)

    # Refit on train only with CV-chosen alpha for fair val evaluation
    from sklearn.linear_model import LassoLars
    lasso = LassoLars(alpha=llcv.alpha_).fit(X_train, y_train)

    n_nz = int(np.sum(lasso.coef_ != 0))
    report("Lasso (LassoLarsCV)",
           evaluate(y_val,  lasso.predict(X_val)),
           evaluate(y_test, lasso.predict(X_test)),
           extra=f"CV alpha={llcv.alpha_:.4g}, non-zero coefs={n_nz}")

    # -------------------------------------------------------------------------
    # 4. gBLUP
    # -------------------------------------------------------------------------
    log("Fitting gBLUP...")
    gblup = GBLUPPredictor(h2=args.h2 if args.h2 > 0 else None)
    gblup.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    report("gBLUP",
           evaluate(y_val,  gblup.predict(X_val)),
           evaluate(y_test, gblup.predict(X_test)),
           extra=f"lambda={gblup.lambda_:.4g}  (h2={args.h2})")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log("=" * 60)
    log("SUMMARY — generalisation gap (Val R2 - Test R2)")
    log("=" * 60)
    rows = [
        ("Linear Regression",   lr.predict(X_val),    lr.predict(X_test)),
        ("Ridge Regression",    ridge.predict(X_val), ridge.predict(X_test)),
        ("Lasso (LassoLarsCV)", lasso.predict(X_val), lasso.predict(X_test)),
        ("gBLUP",               gblup.predict(X_val), gblup.predict(X_test)),
    ]
    log(f"  {'Model':<24}  {'Val R2':>8}  {'Test R2':>8}  {'Gap':>8}")
    log(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, pv, pt in rows:
        rv = r2_score(y_val,  pv)
        rt = r2_score(y_test, pt)
        log(f"  {name:<24}  {rv:>8.4f}  {rt:>8.4f}  {rv - rt:>8.4f}")
    log()

    out_path = out_dir / "baseline_results.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()