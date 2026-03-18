#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

"""
Inputs:
    --vae-config: YAML config for the VAE (model + training hyperparameters)
    --training-data: discovery train dataset (numpy array)
    --validation-data: discovery val dataset (numpy array)
    --target-data: target dataset (numpy array)
    --outputs: directory to save outputs (model checkpoints, loss curves, latent space plots, etc.)
"""

# ------------------------------------------------------------------
# project paths & local imports
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loss import vae_loss
from src.masking import Masker
from src.model import ConvVAE
from src.plotting import (
    plot_example_masked_input_heatmap,
    plot_latent_space,
    plot_loss_curves,
    plot_reconstruction,
    plot_latent_pca_shared_basis,
)
from src.train import evaluate, train_one_epoch


def extract_mu(model, dataloader, device):
    model.eval()
    mu_all = []
    labels_all = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                # training-style loader: (x, label)
                x, y = batch
            elif len(batch) == 4:
                # val/test-style loader: (masked_x, x_true, mask, label)
                x, _, _, y = batch
            else:
                raise ValueError(f"Unexpected batch structure of length {len(batch)}")

            x = x.to(device)
            logits, mu, logvar, z = model(x)
            mu_all.append(mu.cpu().numpy())
            labels_all.append(y.cpu().numpy())

    mu_all = np.concatenate(mu_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return mu_all, labels_all


def load_vae_config(vae_config_path: Path) -> dict:
    with open(vae_config_path, "r") as f:
        vae_config = yaml.safe_load(f)
    return vae_config


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    vae_config: dict,
    input_length: int,
):
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vae_config": vae_config,
        "input_length": input_length,
    }
    torch.save(checkpoint, path)


def main(
    vae_config_path: Path,
    training_data_path: Path,
    validation_data_path: Path,
    target_data_path: Path,
    output_dir: Path,
):
    vae_config = load_vae_config(vae_config_path)

    # ------------------------------------------------------------------
    # output directories
    # ------------------------------------------------------------------
    out = output_dir / "vae_outputs"
    out.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to: {out.resolve()}")

    # ------------------------------------------------------------------
    # hyperparameters
    # ------------------------------------------------------------------
    in_channels = 1
    hidden_channels = vae_config["model"]["hidden_channels"]
    kernel_size = int(vae_config["model"]["kernel_size"])
    stride = int(vae_config["model"]["stride"])
    padding = int(vae_config["model"]["padding"])
    latent_dim = int(vae_config["model"]["latent_dim"])

    beta = float(vae_config["training"]["beta"])
    learning_rate = float(vae_config["training"]["lr"])
    batch_size = int(vae_config["training"]["batch_size"])
    num_epochs = int(vae_config["training"]["max_epochs"])
    patience = int(vae_config["training"].get("patience", 20))
    min_delta = float(vae_config["training"].get("min_delta", 1e-4))

    alpha = float(vae_config["masking"]["alpha_masked"])
    block_length = int(vae_config["masking"]["block_len"])
    mask_frac = float(vae_config["masking"]["mask_frac"])

    # ------------------------------------------------------------------
    # masking
    # ------------------------------------------------------------------
    masker = Masker(block_length=block_length, mask_fraction=mask_frac)

    # ------------------------------------------------------------------
    # load datasets
    # ------------------------------------------------------------------
    training_dataset = np.load(training_data_path)
    validation_dataset = np.load(validation_data_path)
    target_dataset = np.load(target_data_path)

    input_length = training_dataset.shape[-1]

    training_dataset_torch = torch.tensor(training_dataset, dtype=torch.float32).unsqueeze(1)
    validation_dataset_torch = torch.tensor(validation_dataset, dtype=torch.float32).unsqueeze(1)
    target_dataset_torch = torch.tensor(target_dataset, dtype=torch.float32).unsqueeze(1)

    # Precompute masked dataset for validation / target
    masked_val_x, val_mask = masker.mask(validation_dataset_torch)
    masked_target_x, target_mask = masker.mask(target_dataset_torch)

    # Save the pre-masked validation and target datasets for later analysis
    np.save(out / "masked_validation_dataset.npy", masked_val_x.numpy())
    np.save(out / "validation_masks.npy", val_mask.numpy())
    np.save(out / "masked_target_dataset.npy", masked_target_x.numpy())
    np.save(out / "target_masks.npy", target_mask.numpy())

    # Diagnostic mask plot
    plot_example_masked_input_heatmap(
        original_x=validation_dataset_torch,
        masked_x=masked_val_x,
        mask=val_mask,
        output_path=out / "example_masked_input_heatmap_val.png",
        sample_indices=(0, 1, 2, 3, 4),
        snp_start=0,
        snp_count=1000,
    )

    # Dummy labels:
    # 0 = CEU/discovery
    # 1 = YRI/target
    training_dataset_torch = TensorDataset(
        training_dataset_torch,
        torch.zeros(len(training_dataset_torch), dtype=torch.long),
    )

    validation_dataset_torch = TensorDataset(
        masked_val_x,
        validation_dataset_torch,
        val_mask,
        torch.zeros(len(validation_dataset_torch), dtype=torch.long),
    )

    target_dataset_torch = TensorDataset(
        masked_target_x,
        target_dataset_torch,
        target_mask,
        torch.ones(len(target_dataset_torch), dtype=torch.long),
    )

    train_loader = DataLoader(training_dataset_torch, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset_torch, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset_torch, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # model setup
    # ------------------------------------------------------------------
    a = next(iter(train_loader))[0]
    print("Batch shape:", a.shape)
    print("Batch dtype:", a.dtype)
    print("Batch min/max:", a.min().item(), a.max().item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ConvVAE(
        input_length=input_length,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        latent_dim=latent_dim,
        use_batchnorm=False,
        activation="elu",
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # shape tracing
    x_batch = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        logits, mu, logvar, z = model(x_batch, verbose=True)

    print("\nFinal outputs:")
    print("logits shape:", logits.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("z shape:", z.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # training state
    # ------------------------------------------------------------------
    train_loss_list = []
    train_recon_unmasked_list = []
    train_recon_masked_list = []
    train_kl_list = []

    val_loss_list = []
    val_recon_unmasked_list = []
    val_recon_masked_list = []
    val_kl_list = []

    best_val_loss = float("inf")
    best_model_path = checkpoint_dir / "best_model.pt"
    final_model_path = checkpoint_dir / "final_model.pt"
    epochs_without_improvement = 0
    best_epoch = 0

    # fresh masker for train-time dynamic masking
    masker = Masker(block_length=block_length, mask_fraction=mask_frac)

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    for epoch in range(num_epochs):
        train_loss, train_recon_unmasked, train_recon_masked, train_kl = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=vae_loss,
            masker=masker,
            beta=beta,
            alpha=alpha,
        )

        val_loss, val_recon_unmasked, val_recon_masked, val_kl = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=vae_loss,
            beta=beta,
            alpha=alpha,
        )

        train_loss_list.append(train_loss)
        train_recon_unmasked_list.append(train_recon_unmasked)
        train_recon_masked_list.append(train_recon_masked)
        train_kl_list.append(train_kl)

        val_loss_list.append(val_loss)
        val_recon_unmasked_list.append(val_recon_unmasked)
        val_recon_masked_list.append(val_recon_masked)
        val_kl_list.append(val_kl)

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"train_recon_unmasked={train_recon_unmasked:.6f} | "
            f"train_recon_masked={train_recon_masked:.6f} | "
            f"train_kl={train_kl:.6f} | "
            f"val_recon_unmasked={val_recon_unmasked:.6f} | "
            f"val_recon_masked={val_recon_masked:.6f} | "
            f"val_kl={val_kl:.6f}"
        )

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_loss=val_loss,
                vae_config=vae_config,
                input_length=input_length,
            )
            print(f"  Saved new best model at epoch {epoch + 1} (val_loss={val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            print(
                f"  No validation-loss improvement for {epochs_without_improvement} epoch(s) "
                f"(best={best_val_loss:.6f} at epoch {best_epoch})"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    stopped_epoch = len(train_loss_list)
    # Save final model after all training / early stopping
    save_checkpoint(
        path=final_model_path,
        model=model,
        optimizer=optimizer,
        epoch=stopped_epoch,
        val_loss=val_loss_list[-1],
        vae_config=vae_config,
        input_length=input_length,
    )

    # Save training history for later analysis
    history_path = out / "training_history.npz"
    np.savez(
        history_path,
        train_losses=np.array(train_loss_list),
        val_losses=np.array(val_loss_list),
        train_recon_unmasked_losses=np.array(train_recon_unmasked_list),
        train_recon_masked_losses=np.array(train_recon_masked_list),
        val_recon_unmasked_losses=np.array(val_recon_unmasked_list),
        val_recon_masked_losses=np.array(val_recon_masked_list),
        train_kl_losses=np.array(train_kl_list),
        val_kl_losses=np.array(val_kl_list),
    )
    print(f"Saved training history to: {history_path}")

    # Reload best model before downstream plots/evaluation
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(
        f"Reloaded best model from epoch {best_checkpoint['epoch']} "
        f"with val_loss={best_checkpoint['val_loss']:.6f}"
    )

    # ------------------------------------------------------------------
    # plots: validation
    # ------------------------------------------------------------------
    plot_reconstruction(model, val_loader, device, output_dir=out)
    plot_latent_space(model, val_loader, device, output_dir=out)
    plot_loss_curves(
        train_losses=train_loss_list,
        val_losses=val_loss_list,
        train_recon_unmasked_losses=train_recon_unmasked_list,
        train_recon_masked_losses=train_recon_masked_list,
        val_recon_unmasked_losses=val_recon_unmasked_list,
        val_recon_masked_losses=val_recon_masked_list,
        train_kl_losses=train_kl_list,
        val_kl_losses=val_kl_list,
        output_dir=out,
    )

    # ------------------------------------------------------------------
    # target evaluation
    # ------------------------------------------------------------------
    target_loss, target_recon_unmasked, target_recon_masked, target_kl = evaluate(
        model=model,
        dataloader=target_loader,
        device=device,
        loss_fn=vae_loss,
        alpha=alpha,
        beta=beta,
    )
    print("\nTarget set evaluation:")
    print(f"target_loss  = {target_loss:.6f}")
    print(f"target_recon_unmasked = {target_recon_unmasked:.6f}")
    print(f"target_recon_masked   = {target_recon_masked:.6f}")
    print(f"target_kl    = {target_kl:.6f}")

    target_out_path = out / "target"
    target_out_path.mkdir(exist_ok=True)

    plot_reconstruction(model, target_loader, device, output_dir=target_out_path)
    plot_latent_space(model, target_loader, device, output_dir=target_out_path)

    # ------------------------------------------------------------------
    # shared-coordinate latent PCA
    # ------------------------------------------------------------------
    train_mu, _ = extract_mu(model, train_loader, device)
    val_mu, _ = extract_mu(model, val_loader, device)
    target_mu, _ = extract_mu(model, target_loader, device)

    shared_pca_path = out / "latent_pca_ceu_basis_val_vs_target.png"
    plot_latent_pca_shared_basis(
        reference_mu=train_mu,
        ceu_mu=val_mu,
        yri_mu=target_mu,
        output_path=shared_pca_path,
        reference_name="CEU discovery train",
        ceu_name="CEU validation",
        yri_name="YRI target",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train masked ConvVAE on genotype data and save plots, checkpoints, and evaluation outputs."
    )
    parser.add_argument(
        "--vae-config",
        type=Path,
        required=True,
        help="Path to VAE YAML config file.",
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        required=True,
        help="Path to discovery training .npy file.",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        required=True,
        help="Path to discovery validation .npy file.",
    )
    parser.add_argument(
        "--target-data",
        type=Path,
        required=True,
        help="Path to target .npy file.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        required=True,
        help="Directory where output plots/results/checkpoints will be saved.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    main(
        vae_config_path=args.vae_config,
        training_data_path=args.training_data,
        validation_data_path=args.validation_data,
        target_data_path=args.target_data,
        output_dir=args.outputs,
    )