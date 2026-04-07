#!/usr/bin/env python3
"""
probe_latent_phenotype.py

Loads a trained ConvVAE checkpoint, extracts mu vectors for train and val,
then fits a Ridge regression probe on top to assess how much phenotype
information is present in the latent space.

If probe R² >> pheno head R²  --> signal is in latent space, head is the problem
If probe R² is also low       --> bottleneck is destroying phenotype signal

Usage
-----
python mu_test.py \
    --checkpoint /sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/vae/default/vae_outputs/checkpoints/best_model.pt \
    --train_geno /sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_train.npy \
    --val_geno   /sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_val.npy \
    --train_pheno /sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_train.npy \
    --val_pheno   /sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_val.npy \
    --out_dir probe_results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvVAE


# ------------------------------------------------------------------
# mu extraction
# ------------------------------------------------------------------

def extract_mu(model, dataloader, device):
    model.eval()
    mu_all = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            _, mu, _, _, _ = model(x)
            mu_all.append(mu.cpu().numpy())
    return np.concatenate(mu_all, axis=0)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # load checkpoint
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    vae_config  = checkpoint["vae_config"]
    input_length = checkpoint["input_length"]

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.6f})")

    # ------------------------------------------------------------------
    # rebuild model
    # ------------------------------------------------------------------
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
    print(f"Model loaded. Latent dim: {vae_config['model']['latent_dim']}")

    # ------------------------------------------------------------------
    # load genotypes and phenotypes
    # ------------------------------------------------------------------
    X_train = np.load(args.train_geno)
    X_val   = np.load(args.val_geno)
    y_train = np.load(args.train_pheno)
    y_val   = np.load(args.val_pheno)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Phenotype — train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"Phenotype — val   mean: {y_val.mean():.4f},   std: {y_val.std():.4f}")

    # standardize phenotype using train stats (same as training)
    train_mean = y_train.mean()
    train_std  = y_train.std()
    y_train_std = (y_train - train_mean) / train_std
    y_val_std   = (y_val   - train_mean) / train_std

    # ------------------------------------------------------------------
    # build dataloaders (simple, no masking)
    # ------------------------------------------------------------------
    batch_size = 64

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size, shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size, shuffle=False,
    )

    # ------------------------------------------------------------------
    # extract mu
    # ------------------------------------------------------------------
    print("\nExtracting mu vectors...")
    train_mu = extract_mu(model, train_loader, device)
    val_mu   = extract_mu(model, val_loader,   device)
    print(f"train_mu shape: {train_mu.shape}")
    print(f"val_mu   shape: {val_mu.shape}")

    # ------------------------------------------------------------------
    # Ridge probe on mu
    # ------------------------------------------------------------------
    print("\nFitting Ridge probe on mu...")
    probe = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]).fit(train_mu, y_train_std)

    y_pred_train = probe.predict(train_mu)
    y_pred_val   = probe.predict(val_mu)

    r2_train = r2_score(y_train_std, y_pred_train)
    r2_val   = r2_score(y_val_std,   y_pred_val)

    # ------------------------------------------------------------------
    # also get the VAE pheno head predictions for comparison
    # ------------------------------------------------------------------
    print("Getting VAE pheno head predictions...")
    model.eval()
    pheno_head_preds_train = []
    pheno_head_preds_val   = []

    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            _, mu, _, _, pheno_pred = model(x)
            pheno_head_preds_train.append(pheno_pred.cpu().numpy().squeeze())

        for batch in val_loader:
            x = batch[0].to(device)
            _, mu, _, _, pheno_pred = model(x)
            pheno_head_preds_val.append(pheno_pred.cpu().numpy().squeeze())

    pheno_head_train = np.concatenate(pheno_head_preds_train)
    pheno_head_val   = np.concatenate(pheno_head_preds_val)

    r2_head_train = r2_score(y_train_std, pheno_head_train)
    r2_head_val   = r2_score(y_val_std,   pheno_head_val)

    # ------------------------------------------------------------------
    # print results
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PROBE RESULTS")
    print("=" * 60)
    print(f"{'Method':<35} {'Train R²':>10} {'Val R²':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10}")
    print(f"{'Ridge probe on mu':<35} {r2_train:>10.4f} {r2_val:>10.4f}")
    print(f"{'VAE pheno head':<35} {r2_head_train:>10.4f} {r2_head_val:>10.4f}")
    print()

    if r2_val > r2_head_val + 0.05:
        print(">> Probe >> pheno head: signal IS in latent space.")
        print("   The head is the bottleneck — try larger pheno_hidden_dim or higher gamma.")
    elif r2_val < 0.3:
        print(">> Both probe and head are low: phenotype signal is NOT in latent space.")
        print("   The bottleneck is destroying signal — try higher gamma or larger latent_dim.")
    else:
        print(">> Probe and head are similar — head capacity is not the limiting factor.")

    # ------------------------------------------------------------------
    # save mu vectors for further analysis
    # ------------------------------------------------------------------
    np.save(out_dir / "train_mu.npy", train_mu)
    np.save(out_dir / "val_mu.npy",   val_mu)
    np.save(out_dir / "probe_preds_val.npy", y_pred_val)
    print(f"\nSaved mu vectors and probe predictions to: {out_dir.resolve()}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=Path, required=True)
    p.add_argument("--train_geno",  type=Path, required=True)
    p.add_argument("--val_geno",    type=Path, required=True)
    p.add_argument("--train_pheno", type=Path, required=True)
    p.add_argument("--val_pheno",   type=Path, required=True)
    p.add_argument("--out_dir",     type=Path, default=Path("probe_results"))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)