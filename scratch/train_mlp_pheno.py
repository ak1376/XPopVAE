#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


# ------------------------------------------------------------------
# utils
# ------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_2d_genotype(x: np.ndarray) -> np.ndarray:
    """
    Accept genotype array shape:
      (N, L) or (N, 1, L)
    Return float32 array shaped (N, L).
    """
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(f"Expected genotype shape (N, 1, L), got {x.shape}")
        x = x[:, 0, :]
    elif x.ndim != 2:
        raise ValueError(f"Expected genotype array with ndim 2 or 3, got shape {x.shape}")
    return x.astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    corr = float(np.corrcoef(y_true.squeeze(), y_pred.squeeze())[0, 1]) if len(y_true) > 1 else float("nan")
    return {
        "mse": float(mse),
        "rmse": rmse,
        "r2": r2,
        "pearson_r": corr,
    }


# ------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------
class GenotypePhenotypeDataset(Dataset):
    def __init__(self, genotype_path: Path, phenotype_path: Path, causal_snps: np.ndarray | None = None):
        x = np.load(genotype_path)
        y = np.load(phenotype_path)

        x = ensure_2d_genotype(x)

        if causal_snps is not None:
            x = x[:, causal_snps]

        x = ensure_2d_genotype(x)

        if y.ndim == 1:
            y = y[:, None]
        elif y.ndim != 2:
            raise ValueError(f"Expected phenotype array ndim 1 or 2, got shape {y.shape}")

        if len(x) != len(y):
            raise ValueError(
                f"Genotype/phenotype length mismatch: {len(x)} vs {len(y)}\n"
                f"genotype={genotype_path}\nphenotype={phenotype_path}"
            )

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ------------------------------------------------------------------
# model
# ------------------------------------------------------------------
class MLPPhenoModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        phenotype_dim: int = 1,
        dropout: float = 0.0,
        activation: str = "elu",
        use_batchnorm: bool = False,
    ):
        super().__init__()

        if activation == "relu":
            act_cls = nn.ReLU
        elif activation == "gelu":
            act_cls = nn.GELU
        elif activation == "elu":
            act_cls = nn.ELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, phenotype_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x: (B, L)
        pred = self.network(x)
        return pred


# ------------------------------------------------------------------
# training
# ------------------------------------------------------------------
@dataclass
class EpochStats:
    loss: float
    mse: float
    rmse: float
    r2: float


def run_epoch(model, loader, optimizer, device, loss_fn):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(x)
        loss = loss_fn(pred, y)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    metrics = compute_metrics(y_true, y_pred)

    return EpochStats(
        loss=total_loss / len(loader),
        mse=metrics["mse"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
    ), y_true, y_pred


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.num_bad_epochs = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
            return True, False
        else:
            self.num_bad_epochs += 1
            should_stop = self.num_bad_epochs >= self.patience
            return False, should_stop


# ------------------------------------------------------------------
# plots
# ------------------------------------------------------------------
def plot_loss_curves(train_losses, val_losses, outpath: Path):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("MLP phenotype prediction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_scatter(y_true, y_pred, title: str, outpath: Path):
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true.squeeze(), y_pred.squeeze(), alpha=0.35, s=14)

    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("True phenotype")
    plt.ylabel("Predicted phenotype")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ------------------------------------------------------------------
# parser
# ------------------------------------------------------------------
def parse_hidden_dims(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an MLP for supervised phenotype prediction from genotype matrices."
    )

    # data
    parser.add_argument("--train-geno", type=Path, required=True)
    parser.add_argument("--train-pheno", type=Path, required=True)
    parser.add_argument("--val-geno", type=Path, required=True)
    parser.add_argument("--val-pheno", type=Path, required=True)
    parser.add_argument("--target-geno", type=Path, required=True)
    parser.add_argument("--target-pheno", type=Path, required=True)

    # output
    parser.add_argument("--output-dir", type=Path, required=True)

    # architecture
    parser.add_argument("--hidden-dims", type=str, default="256,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="elu", choices=["relu", "gelu", "elu"])
    parser.add_argument("--use-batchnorm", action="store_true")

    # optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--causal-snps", type=Path, default=None)

    return parser


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    causal_snps = None
    if args.causal_snps is not None:
        causal_snps = np.load(args.causal_snps)
        print(f"Using {len(causal_snps)} causal SNPs")

    train_ds = GenotypePhenotypeDataset(args.train_geno, args.train_pheno, causal_snps)
    val_ds   = GenotypePhenotypeDataset(args.val_geno, args.val_pheno, causal_snps)
    target_ds= GenotypePhenotypeDataset(args.target_geno, args.target_pheno, causal_snps)

    input_dim = train_ds.x.shape[1]
    phenotype_dim = train_ds.y.shape[1]
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    print(f"Train shape:  {tuple(train_ds.x.shape)}  pheno: {tuple(train_ds.y.shape)}")
    print(f"Val shape:    {tuple(val_ds.x.shape)}  pheno: {tuple(val_ds.y.shape)}")
    print(f"Target shape: {tuple(target_ds.x.shape)}  pheno: {tuple(target_ds.y.shape)}")
    print(f"Input dim: {input_dim}")
    print(f"Phenotype dim: {phenotype_dim}")
    print(f"Hidden dims: {hidden_dims}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = MLPPhenoModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        phenotype_dim=phenotype_dim,
        dropout=args.dropout,
        activation=args.activation,
        use_batchnorm=args.use_batchnorm,
    ).to(device)

    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_stats, y_train_true, y_train_pred = run_epoch(model, train_loader, optimizer, device, loss_fn)
        val_stats, y_val_true, y_val_pred = run_epoch(model, val_loader, None, device, loss_fn)

        train_losses.append(train_stats.loss)
        val_losses.append(val_stats.loss)

        improved, should_stop = early_stopper.step(val_stats.loss)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_stats.loss:.6f} train_r2={train_stats.r2:.4f} "
            f"val_loss={val_stats.loss:.6f} val_r2={val_stats.r2:.4f}"
        )

        if improved:
            safe_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": safe_args,
                "input_dim": input_dim,
                "phenotype_dim": phenotype_dim,
                "hidden_dims": hidden_dims,
            }
            torch.save(best_state, args.output_dir / "best_mlp_pheno_model.pt")

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    ckpt = torch.load(
        args.output_dir / "best_mlp_pheno_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    train_stats, y_train_true, y_train_pred = run_epoch(model, train_loader, None, device, loss_fn)
    val_stats, y_val_true, y_val_pred = run_epoch(model, val_loader, None, device, loss_fn)
    target_stats, y_target_true, y_target_pred = run_epoch(model, target_loader, None, device, loss_fn)

    metrics = {
        "train": compute_metrics(y_train_true, y_train_pred),
        "val": compute_metrics(y_val_true, y_val_pred),
        "target": compute_metrics(y_target_true, y_target_pred),
        "best_epoch": int(ckpt["epoch"]),
        "hidden_dims": hidden_dims,
    }

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(args.output_dir / "train_true.npy", y_train_true)
    np.save(args.output_dir / "train_pred.npy", y_train_pred)
    np.save(args.output_dir / "val_true.npy", y_val_true)
    np.save(args.output_dir / "val_pred.npy", y_val_pred)
    np.save(args.output_dir / "target_true.npy", y_target_true)
    np.save(args.output_dir / "target_pred.npy", y_target_pred)

    plot_loss_curves(train_losses, val_losses, args.output_dir / "loss_curves.png")
    plot_scatter(y_train_true, y_train_pred, "Train: predicted vs true phenotype", args.output_dir / "scatter_train.png")
    plot_scatter(y_val_true, y_val_pred, "Val: predicted vs true phenotype", args.output_dir / "scatter_val.png")
    plot_scatter(y_target_true, y_target_pred, "Target: predicted vs true phenotype", args.output_dir / "scatter_target.png")

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()