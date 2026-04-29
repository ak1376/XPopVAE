"""
MLP baseline: train on CEU genotypes → phenotype, evaluate transfer on YRI.

For each replicate:
  - Standardize genotypes using CEU training set per-SNP mean/std
  - Apply same transform to CEU validation and YRI held-out sets
  - Early stopping on CEU validation loss
  - Report R² for CEU train and YRI transfer
  - Save scatterplots and summary CSV to experiments/OOA/mlp_baseline/
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path

matplotlib.use("Agg")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path("experiments/OOA")
OUTPUT_DIR = BASE_DIR / "mlp_baseline"
PROCESSED_DIR = BASE_DIR / "processed_data"

NUM_DRAWS = 1
NUM_REPS = 10

# ── hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIMS = [512, 128]
DROPOUT = 0.2
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 20


# ── model ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── per-replicate routine ──────────────────────────────────────────────────────
def train_and_eval(draw: int, rep: int, device: torch.device) -> dict:
    rep_dir = PROCESSED_DIR / str(draw) / f"rep{rep}"

    # load
    X_train = np.load(rep_dir / "genotype_matrices" / "discovery_train.npy")
    y_train = np.load(rep_dir / "phenotypes" / "discovery_train_pheno.npy")
    X_val   = np.load(rep_dir / "genotype_matrices" / "discovery_validation.npy")
    y_val   = np.load(rep_dir / "phenotypes" / "discovery_validation_pheno.npy")
    X_yri   = np.load(rep_dir / "genotype_matrices" / "target_held_out.npy")
    y_yri   = np.load(rep_dir / "phenotypes" / "target_held_out_pheno.npy")

    # standardize genotypes using CEU training statistics
    mu    = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_s = (X_train - mu) / sigma
    X_val_s   = (X_val   - mu) / sigma
    X_yri_s   = (X_yri   - mu) / sigma

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    X_tr = to_tensor(X_train_s); y_tr = to_tensor(y_train)
    X_vl = to_tensor(X_val_s);   y_vl = to_tensor(y_val)
    X_yr = to_tensor(X_yri_s)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train.shape[1]
    model = MLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_vl), y_vl).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_train = model(X_tr).cpu().numpy()
        pred_yri   = model(X_yr).cpu().numpy()

    r2_train = r2_score(y_train, pred_train)
    r2_yri   = r2_score(y_yri,   pred_yri)

    # ── save outputs ─────────────────────────────────────────────────────────
    out_dir = OUTPUT_DIR / str(draw) / f"rep{rep}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"MLP baseline  |  draw={draw}  rep={rep}", fontsize=13)

    for ax, y_true, y_pred, pop, r2 in [
        (axes[0], y_train, pred_train, "CEU train",    r2_train),
        (axes[1], y_yri,   pred_yri,   "YRI transfer", r2_yri),
    ]:
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.scatter(y_true, y_pred, alpha=0.4, s=8, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("True phenotype")
        ax.set_ylabel("Predicted phenotype")
        ax.set_title(f"{pop}  R²={r2:.3f}")

    plt.tight_layout()
    plt.savefig(out_dir / "scatterplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"draw": draw, "rep": rep, "r2_train": r2_train, "r2_yri": r2_yri}


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    results = []
    for draw in range(NUM_DRAWS):
        for rep in range(NUM_REPS):
            print(f"draw={draw}  rep={rep} ...", flush=True)
            result = train_and_eval(draw, rep, device)
            results.append(result)
            print(f"  R² train={result['r2_train']:.4f}  R² YRI={result['r2_yri']:.4f}")

    df = pd.DataFrame(results)

    summary_stats = df[["r2_train", "r2_yri"]].agg(["mean", "std"]).T
    summary_stats.columns = ["mean", "std"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\n── per-replicate results ──────────────────────────")
    print(df.to_string(index=False))
    print("\n── aggregate ──────────────────────────────────────")
    print(summary_stats.to_string())
    print(f"\nResults written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
