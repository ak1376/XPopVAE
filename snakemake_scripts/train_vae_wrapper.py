#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

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
    extract_mu,
    plot_example_input_heatmap,
    plot_latent_space,
    plot_latent_pca_shared_basis,
    plot_loss_curves,
    plot_reconstruction,
    plot_pheno_predictions,
    plot_pheno_predictions_by_population,
    plot_pheno_residuals,
)
from src.train import evaluate, train_one_epoch


def load_vae_config(vae_config_path: Path) -> dict:
    with open(vae_config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(path, model, optimizer, epoch, val_loss, vae_config, input_length):
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vae_config": vae_config,
            "input_length": input_length,
        },
        path,
    )


def main(
    vae_config_path,
    training_data_path,
    validation_data_path,
    train_target_data_path,
    test_target_data_path,
    training_pheno_path,
    validation_pheno_path,
    test_target_pheno_path,
    output_dir,
):
    vae_config = load_vae_config(vae_config_path)

    out = output_dir / "vae_outputs"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {out.resolve()}")

    # ------------------------------------------------------------------
    # hyperparameters
    # ------------------------------------------------------------------
    in_channels     = 1
    hidden_channels = vae_config["model"]["hidden_channels"]
    kernel_size     = int(vae_config["model"]["kernel_size"])
    stride          = int(vae_config["model"]["stride"])
    padding         = int(vae_config["model"]["padding"])
    latent_dim      = int(vae_config["model"]["latent_dim"])

    beta          = float(vae_config["training"]["beta"])
    learning_rate = float(vae_config["training"]["lr"])
    batch_size    = int(vae_config["training"]["batch_size"])
    num_epochs    = int(vae_config["training"]["max_epochs"])
    patience      = int(vae_config["training"].get("patience", 500))
    min_delta     = float(vae_config["training"].get("min_delta", 1e-4))

    masking      = vae_config["masking"].get("enabled", False)
    alpha        = float(vae_config["masking"]["alpha_masked"])
    block_length = int(vae_config["masking"]["block_len"])
    mask_frac    = float(vae_config["masking"]["mask_frac"])

    pheno_hidden_dim = vae_config["phenotype"].get("pheno_hidden_dim", None)
    pheno_latent_dim = int(vae_config["phenotype"].get("pheno_latent_dim", 16))
    gamma            = float(vae_config["phenotype"].get("gamma", 1.0))
    lambda_ortho     = float(vae_config["phenotype"].get("lambda_ortho", 1e-5))

    print(f"Masking enabled:   {masking}")
    print(f"pheno_latent_dim:  {pheno_latent_dim}")
    print(f"lambda_ortho:      {lambda_ortho}")

    masker = Masker(block_length=block_length, mask_fraction=mask_frac) if masking else None
    if not masking:
        print("Masking is disabled.")

    # ------------------------------------------------------------------
    # load + normalize
    # ------------------------------------------------------------------
    training_dataset     = np.load(training_data_path)
    validation_dataset   = np.load(validation_data_path)
    train_target_dataset = np.load(train_target_data_path)
    test_target_dataset  = np.load(test_target_data_path)

    training_pheno     = np.load(training_pheno_path)
    validation_pheno   = np.load(validation_pheno_path)
    test_target_pheno  = np.load(test_target_pheno_path)

    train_mean, train_std = training_pheno.mean(), training_pheno.std()
    training_pheno    = (training_pheno    - train_mean) / train_std
    validation_pheno  = (validation_pheno  - train_mean) / train_std
    test_target_pheno = (test_target_pheno - train_mean) / train_std

    input_length = training_dataset.shape[-1]

    # ------------------------------------------------------------------
    # tensorify
    # ------------------------------------------------------------------
    training_dataset_torch     = torch.tensor(training_dataset,     dtype=torch.float32).unsqueeze(1)
    validation_dataset_torch   = torch.tensor(validation_dataset,   dtype=torch.float32).unsqueeze(1)
    train_target_dataset_torch = torch.tensor(train_target_dataset, dtype=torch.float32).unsqueeze(1)
    test_target_dataset_torch  = torch.tensor(test_target_dataset,  dtype=torch.float32).unsqueeze(1)

    training_pheno_torch     = torch.tensor(training_pheno,     dtype=torch.float32).unsqueeze(1)
    validation_pheno_torch   = torch.tensor(validation_pheno,   dtype=torch.float32).unsqueeze(1)
    test_target_pheno_torch  = torch.tensor(test_target_pheno,  dtype=torch.float32).unsqueeze(1)
    # YRI train pheno zeroed out — phenotype loss is masked by pop_label in train.py
    train_target_pheno_torch = torch.zeros(len(train_target_dataset_torch), 1, dtype=torch.float32)

    # ------------------------------------------------------------------
    # masking / identity for val + test_target
    # ------------------------------------------------------------------
    if masking:
        val_input_x,         val_mask         = masker.mask(validation_dataset_torch)
        test_target_input_x, test_target_mask = masker.mask(test_target_dataset_torch)
        np.save(out / "masked_validation_dataset.npy",   val_input_x.numpy())
        np.save(out / "validation_masks.npy",            val_mask.numpy())
        np.save(out / "masked_test_target_dataset.npy",  test_target_input_x.numpy())
        np.save(out / "test_target_masks.npy",           test_target_mask.numpy())
    else:
        val_input_x         = validation_dataset_torch
        val_mask            = torch.zeros(validation_dataset_torch.shape[0],
                                          validation_dataset_torch.shape[2], dtype=torch.bool)
        test_target_input_x = test_target_dataset_torch
        test_target_mask    = torch.zeros(test_target_dataset_torch.shape[0],
                                          test_target_dataset_torch.shape[2], dtype=torch.bool)
        np.save(out / "validation_dataset.npy",   val_input_x.numpy())
        np.save(out / "test_target_dataset.npy",  test_target_input_x.numpy())

    plot_example_input_heatmap(
        original_x=validation_dataset_torch,
        masked_x=val_input_x,
        mask=val_mask,
        output_path=out / "example_input_heatmap_val.png",
        sample_indices=(0, 1, 2, 3, 4),
        snp_start=0,
        snp_count=1000,
    )

    # ------------------------------------------------------------------
    # DataLoaders
    # train : (x, pheno, pop_label)          CEU=0, YRI=1
    # val   : (input_x, x, pheno, mask, pop) all CEU (pop=0)
    # test  : (input_x, x, pheno, mask, pop) all YRI (pop=1)
    # ------------------------------------------------------------------
    train_ds = TensorDataset(
        torch.cat([training_dataset_torch, train_target_dataset_torch], dim=0),
        torch.cat([training_pheno_torch,   train_target_pheno_torch],   dim=0),
        torch.cat([
            torch.zeros(len(training_dataset_torch),    dtype=torch.long),  # CEU = 0
            torch.ones(len(train_target_dataset_torch), dtype=torch.long),  # YRI = 1
        ], dim=0),
    )
    val_ds = TensorDataset(
        val_input_x,
        validation_dataset_torch,
        validation_pheno_torch,
        val_mask,
        torch.zeros(len(validation_dataset_torch), dtype=torch.long),
    )
    test_target_ds = TensorDataset(
        test_target_input_x,
        test_target_dataset_torch,
        test_target_pheno_torch,
        test_target_mask,
        torch.ones(len(test_target_dataset_torch), dtype=torch.long),
    )

    train_loader       = DataLoader(train_ds,       batch_size=batch_size, shuffle=True)
    val_loader         = DataLoader(val_ds,         batch_size=batch_size, shuffle=False)
    test_target_loader = DataLoader(test_target_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        pheno_dim=1,
        pheno_hidden_dim=pheno_hidden_dim,
        pheno_latent_dim=pheno_latent_dim,
    ).to(device)

    print(f"Parameters:       {sum(p.numel() for p in model.parameters()):,}")
    print(f"recon_latent_dim: {model.recon_latent_dim}")
    print(f"pheno_latent_dim: {model.pheno_latent_dim}")

    x_batch = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        logits, mu, logvar, z, pheno_pred = model(x_batch, verbose=True)
    print(f"logits={logits.shape}  mu={mu.shape}  z={z.shape}  pheno_pred={pheno_pred.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # training history
    # ------------------------------------------------------------------
    train_loss_list           = []
    train_recon_unmasked_list = []
    train_recon_masked_list   = []
    train_kl_list             = []
    train_phenotype_loss_list = []
    train_ortho_loss_list     = []

    val_loss_list             = []
    val_recon_unmasked_list   = []
    val_recon_masked_list     = []
    val_kl_list               = []
    val_phenotype_loss_list   = []
    val_ortho_loss_list       = []

    best_val_stop_metric       = float("inf")
    best_model_path            = checkpoint_dir / "best_model.pt"
    final_model_path           = checkpoint_dir / "final_model.pt"
    epochs_without_improvement = 0
    best_epoch                 = 0

    if masking:
        masker = Masker(block_length=block_length, mask_fraction=mask_frac)

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    for epoch in range(num_epochs):

        (train_loss, train_recon_unmasked, train_recon_masked,
         train_kl, train_phenotype_loss, train_ortho_loss) = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=vae_loss,
            masker=masker,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
            lambda_ortho=lambda_ortho,
        )

        (val_loss, val_recon_unmasked, val_recon_masked,
         val_kl, val_phenotype_loss, val_ortho_loss) = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=vae_loss,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
            lambda_ortho=lambda_ortho,
        )

        val_stop_metric = (
            val_recon_unmasked
            + alpha * val_recon_masked
            + beta  * val_kl
            + gamma * val_phenotype_loss
        )

        train_loss_list.append(train_loss)
        train_recon_unmasked_list.append(train_recon_unmasked)
        train_recon_masked_list.append(train_recon_masked)
        train_kl_list.append(train_kl)
        train_phenotype_loss_list.append(train_phenotype_loss)
        train_ortho_loss_list.append(train_ortho_loss)

        val_loss_list.append(val_loss)
        val_recon_unmasked_list.append(val_recon_unmasked)
        val_recon_masked_list.append(val_recon_masked)
        val_kl_list.append(val_kl)
        val_phenotype_loss_list.append(val_phenotype_loss)
        val_ortho_loss_list.append(val_ortho_loss)

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"recon={train_recon_unmasked:.4f} val_recon={val_recon_unmasked:.4f} | "
            f"pheno={train_phenotype_loss:.4f} val_pheno={val_phenotype_loss:.4f} | "
            f"ortho={train_ortho_loss:.4f}"
        )

        if val_stop_metric < (best_val_stop_metric - min_delta):
            best_val_stop_metric       = val_stop_metric
            best_epoch                 = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_loss=val_stop_metric,
                vae_config=vae_config,
                input_length=input_length,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for {epochs_without_improvement} epoch(s) "
                f"(best={best_val_stop_metric:.6f} at epoch {best_epoch})"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    save_checkpoint(
        path=final_model_path,
        model=model,
        optimizer=optimizer,
        epoch=len(train_loss_list),
        val_loss=val_stop_metric,
        vae_config=vae_config,
        input_length=input_length,
    )

    # ------------------------------------------------------------------
    # save training history
    # ------------------------------------------------------------------
    np.savez(
        out / "training_history.npz",
        train_losses=np.array(train_loss_list),
        val_losses=np.array(val_loss_list),
        train_recon_unmasked_losses=np.array(train_recon_unmasked_list),
        val_recon_unmasked_losses=np.array(val_recon_unmasked_list),
        train_kl_losses=np.array(train_kl_list),
        val_kl_losses=np.array(val_kl_list),
        train_phenotype_losses=np.array(train_phenotype_loss_list),
        val_phenotype_losses=np.array(val_phenotype_loss_list),
        train_ortho_losses=np.array(train_ortho_loss_list),
        val_ortho_losses=np.array(val_ortho_loss_list),
        train_recon_masked_losses=np.array(train_recon_masked_list),
        val_recon_masked_losses=np.array(val_recon_masked_list),
    )
    print(f"Saved training history to: {out / 'training_history.npz'}")

    # ------------------------------------------------------------------
    # reload best model
    # ------------------------------------------------------------------
    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Reloaded best model from epoch {best_ckpt['epoch']} (stop={best_ckpt['val_loss']:.6f})")

    use_masked = masking

    # ------------------------------------------------------------------
    # loss curves
    # ------------------------------------------------------------------
    plot_loss_curves(
        train_losses=train_loss_list,
        val_losses=val_loss_list,
        train_recon_unmasked_losses=train_recon_unmasked_list,
        val_recon_unmasked_losses=val_recon_unmasked_list,
        train_kl_losses=train_kl_list,
        val_kl_losses=val_kl_list,
        train_pheno_losses=train_phenotype_loss_list,
        val_pheno_losses=val_phenotype_loss_list,
        train_recon_masked_losses=train_recon_masked_list if masking else None,
        val_recon_masked_losses=val_recon_masked_list     if masking else None,
        output_dir=out,
    )

    # ------------------------------------------------------------------
    # validation (CEU) evaluation
    # ------------------------------------------------------------------
    val_metrics = plot_reconstruction(
        model=model, dataloader=val_loader, device=device,
        output_dir=out, use_masked_input=use_masked,
    )
    print(f"Val balanced accuracy: {val_metrics['balanced_accuracy']:.6f}")

    plot_latent_space(
        model=model, dataloader=val_loader, device=device,
        output_dir=out, save_path="latent_space.png", use_masked_input=use_masked,
    )

    val_pheno = plot_pheno_predictions(
        model=model, dataloader=val_loader, device=device,
        output_path=out / "pheno_pred_vs_true_val.png",
        use_masked_input=use_masked,
        title="Validation phenotype prediction",
    )
    print(f"Val RMSE={val_pheno['rmse']:.6f}  R²={val_pheno['r2']:.6f}")

    plot_pheno_predictions_by_population(
        model=model, dataloader=val_loader, device=device,
        output_path=out / "pheno_pred_vs_true_val_by_population.png",
        use_masked_input=use_masked,
        title="Validation phenotype prediction by population",
    )

    plot_pheno_residuals(
        model=model, dataloader=val_loader, device=device,
        output_path=out / "pheno_residuals_val.png",
        use_masked_input=use_masked,
        title="Validation phenotype residuals",
    )

    # ------------------------------------------------------------------
    # test target (YRI) evaluation
    # ------------------------------------------------------------------
    test_target_out = out / "test_target"
    test_target_out.mkdir(exist_ok=True)

    tt_metrics = plot_reconstruction(
        model=model, dataloader=test_target_loader, device=device,
        output_dir=test_target_out, use_masked_input=use_masked,
    )
    print(f"Test target balanced accuracy: {tt_metrics['balanced_accuracy']:.6f}")

    plot_latent_space(
        model=model, dataloader=test_target_loader, device=device,
        output_dir=test_target_out, save_path="latent_space.png", use_masked_input=use_masked,
    )

    tt_pheno = plot_pheno_predictions(
        model=model, dataloader=test_target_loader, device=device,
        output_path=test_target_out / "pheno_pred_vs_true_test_target.png",
        use_masked_input=use_masked,
        title="Test target phenotype prediction",
    )
    print(f"Test target RMSE={tt_pheno['rmse']:.6f}  R²={tt_pheno['r2']:.6f}")

    plot_pheno_predictions_by_population(
        model=model, dataloader=test_target_loader, device=device,
        output_path=test_target_out / "pheno_pred_vs_true_test_target_by_population.png",
        use_masked_input=use_masked,
        title="Test target phenotype prediction by population",
    )

    plot_pheno_residuals(
        model=model, dataloader=test_target_loader, device=device,
        output_path=test_target_out / "pheno_residuals_test_target.png",
        use_masked_input=use_masked,
        title="Test target phenotype residuals",
    )

    # ------------------------------------------------------------------
    # shared-coordinate latent PCA
    # ------------------------------------------------------------------
    train_mu, _ = extract_mu(model=model, dataloader=train_loader,       device=device, use_masked_input=False)
    val_mu,   _ = extract_mu(model=model, dataloader=val_loader,         device=device, use_masked_input=use_masked)
    tt_mu,    _ = extract_mu(model=model, dataloader=test_target_loader, device=device, use_masked_input=use_masked)

    plot_latent_pca_shared_basis(
        reference_mu=train_mu,
        ceu_mu=val_mu,
        yri_mu=tt_mu,
        output_path=out / "latent_pca_ceu_basis_val_vs_test_target.png",
        reference_name="CEU discovery train",
        ceu_name="CEU validation",
        yri_name="YRI test target",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train split-latent ConvVAE.")
    parser.add_argument("--vae-config",          type=Path, required=True)
    parser.add_argument("--training-data",       type=Path, required=True)
    parser.add_argument("--validation-data",     type=Path, required=True)
    parser.add_argument("--train-target-data",   type=Path, required=True)
    parser.add_argument("--test-target-data",    type=Path, required=True)
    parser.add_argument("--training-pheno",      type=Path, required=True)
    parser.add_argument("--validation-pheno",    type=Path, required=True)
    parser.add_argument("--test-target-pheno",   type=Path, required=True)
    parser.add_argument("--outputs",             type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(
        vae_config_path=args.vae_config,
        training_data_path=args.training_data,
        validation_data_path=args.validation_data,
        train_target_data_path=args.train_target_data,
        test_target_data_path=args.test_target_data,
        training_pheno_path=args.training_pheno,
        validation_pheno_path=args.validation_pheno,
        test_target_pheno_path=args.test_target_pheno,
        output_dir=args.outputs,
    )