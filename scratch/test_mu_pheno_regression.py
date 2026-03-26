#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------------
# project paths & local imports
# ------------------------------------------------------------------
PROJECT_ROOT = Path("/sietch_colab/akapoor/XPopVAE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvVAE


def load_vae_config(vae_config_path: Path) -> dict:
    with open(vae_config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> ConvVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vae_config = checkpoint["vae_config"]
    input_length = int(checkpoint["input_length"])

    model = ConvVAE(
        input_length=input_length,
        in_channels=1,
        hidden_channels=vae_config["model"]["hidden_channels"],
        kernel_size=int(vae_config["model"]["kernel_size"]),
        stride=int(vae_config["model"]["stride"]),
        padding=int(vae_config["model"]["padding"]),
        latent_dim=int(vae_config["model"]["latent_dim"]),
        use_batchnorm=False,
        activation="elu",
        pheno_dim=1,
        pheno_hidden_dim=vae_config["phenotype"].get("pheno_hidden_dim", None),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def extract_mu(model: ConvVAE, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    mu_all = []

    model.eval()
    for (x,) in dataloader:
        x = x.to(device)
        _, mu, _, _, _ = model(x)
        mu_all.append(mu.cpu().numpy())

    return np.concatenate(mu_all, axis=0)


def evaluate_regressor(name, reg, X_train, y_train, X_val, y_val):
    reg.fit(X_train, y_train)

    yhat_train = reg.predict(X_train)
    yhat_val = reg.predict(X_val)

    train_r2 = r2_score(y_train, yhat_train)
    val_r2 = r2_score(y_val, yhat_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, yhat_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, yhat_val))

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Train R^2 : {train_r2:.6f}")
    print(f"Val   R^2 : {val_r2:.6f}")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Val   RMSE: {val_rmse:.6f}")


def main():
    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    checkpoint_path = Path("/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/vae/default/vae_outputs/checkpoints/best_model.pt")

    train_geno_path = Path("/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_train.npy")
    val_geno_path   = Path("/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_val.npy")

    train_pheno_path = Path("/sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_train.npy")
    val_pheno_path   = Path("/sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_val.npy")

    # ------------------------------------------------------------------
    # load data
    # ------------------------------------------------------------------
    X_train = np.load(train_geno_path).astype(np.float32)
    X_val = np.load(val_geno_path).astype(np.float32)

    y_train = np.load(train_pheno_path).astype(np.float32)
    y_val = np.load(val_pheno_path).astype(np.float32)

    # ensure phenotype is 1D for sklearn
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)

    print("Loaded arrays:")
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:  ", y_val.shape)

    # ------------------------------------------------------------------
    # standardize phenotype same way as training script
    # ------------------------------------------------------------------
    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_std = (y_train - y_mean) / y_std
    y_val_std = (y_val - y_mean) / y_std

    print("\nPhenotype standardization:")
    print("train mean/std after standardization:",
          y_train_std.mean(), y_train_std.std())
    print("val mean/std after standardization:",
          y_val_std.mean(), y_val_std.std())

    # ------------------------------------------------------------------
    # torch loaders for mu extraction
    # ------------------------------------------------------------------
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_torch), batch_size=128, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_torch), batch_size=128, shuffle=False)

    # ------------------------------------------------------------------
    # load model + extract mu
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    model = load_model_from_checkpoint(checkpoint_path, device)

    mu_train = extract_mu(model, train_loader, device)
    mu_val = extract_mu(model, val_loader, device)

    print("\nExtracted latent means:")
    print("mu_train shape:", mu_train.shape)
    print("mu_val shape:  ", mu_val.shape)

    # ------------------------------------------------------------------
    # regress phenotype from mu
    # ------------------------------------------------------------------
    evaluate_regressor(
        "LinearRegression on mu",
        LinearRegression(),
        mu_train, y_train_std,
        mu_val, y_val_std,
    )

    for alpha in [0.1, 1.0, 10.0, 100.0]:
        evaluate_regressor(
            f"Ridge(alpha={alpha}) on mu",
            Ridge(alpha=alpha),
            mu_train, y_train_std,
            mu_val, y_val_std,
        )


if __name__ == "__main__":
    main()