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
    --vae-config                : YAML config for the VAE
    --training-data             : stacked CEU+YRI genotypes  (training.npy)
    --disc-train-data           : CEU-only training genotypes (discovery_train.npy)  [eval only]
    --target-train-data         : YRI-only training genotypes (target_train.npy)     [eval only]
    --validation-data           : CEU validation genotypes   (discovery_validation.npy)
    --disc-train-pheno          : CEU training phenotypes    (simulated_phenotype_disc_train.npy)
    --target-train-pheno        : YRI training phenotypes    (simulated_phenotype_target_train.npy)
    --validation-pheno          : CEU validation phenotypes  (simulated_phenotype_disc_val.npy)
    --outputs                   : root output directory

Training loader  : training.npy  (CEU + YRI mixed, shuffled)
                   phenotype loss masked to CEU rows only (pop_label == 0)
Evaluation loaders (plots + metrics only, no gradient updates):
    discovery_train/  <- discovery_train.npy
    target_train/     <- target_train.npy
    discovery_validation/ <- discovery_validation.npy
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
    plot_example_input_heatmap,
    plot_latent_space,
    plot_loss_curves,
    plot_reconstruction,
    plot_latent_pca_shared_basis,
    plot_pheno_predictions,
    plot_pheno_predictions_by_population,
    plot_pheno_residuals,
)
from src.train import evaluate, train_one_epoch


# =============================================================================
# Helpers
# =============================================================================

def extract_mu(model, dataloader, device, use_masked_input=False):
    model.eval()
    mu_all, labels_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x = batch[0]
                pop_label = batch[-1]
            elif len(batch) == 5:
                x = batch[0] if use_masked_input else batch[1]
                pop_label = batch[-1]
            else:
                raise ValueError(f"Unexpected batch length {len(batch)}")
            x = x.to(device)
            _, mu, _, _, _ = model(x)
            mu_all.append(mu.cpu().numpy())
            labels_all.append(pop_label.cpu().numpy())
    return np.concatenate(mu_all, axis=0), np.concatenate(labels_all, axis=0)


def load_vae_config(path: Path) -> dict:
    with open(path) as f:
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


def make_eval_loader(
    geno: np.ndarray,
    pheno_norm: np.ndarray,
    pop_label_value: int,
    batch_size: int,
    masker,
    masking: bool,
    out_dir: Path,
    split_name: str,
):
    """
    Build a val-style TensorDataset/DataLoader for one evaluation split.
    Tuple order: (input_x, original_x, pheno, mask, pop_label)
    """
    geno_t  = torch.tensor(geno,       dtype=torch.float32).unsqueeze(1)
    pheno_t = torch.tensor(pheno_norm, dtype=torch.float32).unsqueeze(1)

    if masking:
        input_x, mask = masker.mask(geno_t)
        np.save(out_dir / f"masked_{split_name}.npy", input_x.numpy())
        np.save(out_dir / f"mask_{split_name}.npy",   mask.numpy())
    else:
        input_x = geno_t
        mask    = torch.zeros(geno_t.shape[0], geno_t.shape[2], dtype=torch.bool)

    pop_labels = torch.full((len(geno_t),), pop_label_value, dtype=torch.long)

    ds     = TensorDataset(input_x, geno_t, pheno_t, mask, pop_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def run_eval_plots(
    model,
    loader,
    device,
    out_dir: Path,
    split_name: str,
    use_masked: bool,
    loss_fn,
    alpha: float,
    beta: float,
    gamma: float,
):
    """
    Run a full evaluation pass for one split and save all plots + metrics
    into out_dir/split_name/.
    """
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    loss, recon_unmasked, recon_masked, kl, pheno_loss = evaluate(
        model=model,
        dataloader=loader,
        device=device,
        loss_fn=loss_fn,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    print(f"\n[{split_name}] loss={loss:.6f}  recon={recon_unmasked:.6f}  "
          f"kl={kl:.6f}  pheno_loss={pheno_loss:.6f}")

    recon_metrics = plot_reconstruction(
        model=model,
        dataloader=loader,
        device=device,
        output_dir=split_dir,
        use_masked_input=use_masked,
    )
    print(f"[{split_name}] balanced_accuracy={recon_metrics['balanced_accuracy']:.6f}")

    plot_latent_space(
        model=model,
        dataloader=loader,
        device=device,
        output_dir=split_dir,
        save_path="latent_space.png",
        use_masked_input=use_masked,
    )

    pheno_metrics = plot_pheno_predictions(
        model=model,
        dataloader=loader,
        device=device,
        output_path=split_dir / f"pheno_pred_vs_true_{split_name}.png",
        use_masked_input=use_masked,
        title=f"{split_name} phenotype prediction",
    )
    print(f"[{split_name}] pheno RMSE={pheno_metrics['rmse']:.6f}  R²={pheno_metrics['r2']:.6f}")

    plot_pheno_predictions_by_population(
        model=model,
        dataloader=loader,
        device=device,
        output_path=split_dir / f"pheno_pred_vs_true_{split_name}_by_pop.png",
        use_masked_input=use_masked,
        title=f"{split_name} phenotype prediction by population",
    )

    plot_pheno_residuals(
        model=model,
        dataloader=loader,
        device=device,
        output_path=split_dir / f"pheno_residuals_{split_name}.png",
        use_masked_input=use_masked,
        title=f"{split_name} phenotype residuals",
    )

    return recon_metrics, pheno_metrics


# =============================================================================
# Main
# =============================================================================

def main(
    vae_config_path: Path,
    training_data_path: Path,       # training.npy  (CEU + YRI stacked)
    disc_train_data_path: Path,     # discovery_train.npy  (eval only)
    target_train_data_path: Path,   # target_train.npy     (eval only)
    validation_data_path: Path,     # discovery_validation.npy
    disc_train_pheno_path: Path,    # simulated_phenotype_disc_train.npy
    target_train_pheno_path: Path,  # simulated_phenotype_target_train.npy
    validation_pheno_path: Path,    # simulated_phenotype_disc_val.npy
    output_dir: Path,
):
    vae_config = load_vae_config(vae_config_path)

    # ------------------------------------------------------------------
    # output directories
    # ------------------------------------------------------------------
    out            = output_dir / "vae_outputs"
    checkpoint_dir = out / "checkpoints"
    plots_dir      = out / "plots"
    for d in (out, checkpoint_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to: {out.resolve()}")

    # ------------------------------------------------------------------
    # hyperparameters
    # ------------------------------------------------------------------
    in_channels      = 1
    hidden_channels  = vae_config["model"]["hidden_channels"]
    kernel_size      = int(vae_config["model"]["kernel_size"])
    stride           = int(vae_config["model"]["stride"])
    padding          = int(vae_config["model"]["padding"])
    latent_dim       = int(vae_config["model"]["latent_dim"])

    beta             = float(vae_config["training"]["beta"])
    learning_rate    = float(vae_config["training"]["lr"])
    batch_size       = int(vae_config["training"]["batch_size"])
    num_epochs       = int(vae_config["training"]["max_epochs"])
    patience         = int(vae_config["training"].get("patience", 500))
    min_delta        = float(vae_config["training"].get("min_delta", 1e-4))

    masking          = vae_config["masking"].get("enabled", False)
    alpha            = float(vae_config["masking"]["alpha_masked"])
    block_length     = int(vae_config["masking"]["block_len"])
    mask_frac        = float(vae_config["masking"]["mask_frac"])

    pheno_hidden_dim  = vae_config["phenotype"].get("pheno_hidden_dim", None)
    gamma             = float(vae_config["phenotype"].get("gamma", 1.0))
    pheno_weight_decay = float(vae_config["phenotype"].get("pheno_weight_decay", 0.0))

    print(f"Masking enabled: {masking}")

    # ------------------------------------------------------------------
    # load genotypes
    # ------------------------------------------------------------------
    training_dataset   = np.load(training_data_path)    # CEU + YRI stacked
    disc_train_dataset = np.load(disc_train_data_path)  # CEU only  (eval)
    target_train_dataset = np.load(target_train_data_path)  # YRI only (eval)
    validation_dataset = np.load(validation_data_path)  # CEU val

    n_disc_train   = disc_train_dataset.shape[0]
    n_target_train = target_train_dataset.shape[0]
    assert n_disc_train + n_target_train == training_dataset.shape[0], (
        f"training.npy rows ({training_dataset.shape[0]}) != "
        f"discovery_train ({n_disc_train}) + target_train ({n_target_train})"
    )

    input_length = training_dataset.shape[-1]

    # ------------------------------------------------------------------
    # load + normalise phenotypes
    # Standardise using CEU discovery_train mean/std ONLY
    # ------------------------------------------------------------------
    disc_train_pheno   = np.load(disc_train_pheno_path).astype(np.float32)
    target_train_pheno = np.load(target_train_pheno_path).astype(np.float32)
    validation_pheno   = np.load(validation_pheno_path).astype(np.float32)

    train_mean = disc_train_pheno.mean()
    train_std  = disc_train_pheno.std()
    print(f"Phenotype normalisation — mean={train_mean:.4f}  std={train_std:.4f}  (CEU disc_train only)")

    disc_train_pheno_norm   = (disc_train_pheno   - train_mean) / train_std
    target_train_pheno_norm = (target_train_pheno - train_mean) / train_std
    validation_pheno_norm   = (validation_pheno   - train_mean) / train_std

    # stacked training pheno: CEU rows real, YRI rows zeroed (masked in loss via pop_label)
    training_pheno_norm = np.concatenate(
        [disc_train_pheno_norm, np.zeros(n_target_train, dtype=np.float32)], axis=0
    )

    # ------------------------------------------------------------------
    # masker
    # ------------------------------------------------------------------
    masker = Masker(block_length=block_length, mask_fraction=mask_frac) if masking else None

    # ------------------------------------------------------------------
    # build TRAINING tensors + loader
    # pop_label: 0 = CEU (phenotype loss active), 1 = YRI (phenotype loss masked)
    # Tuple order: (x, pheno, pop_label)
    # ------------------------------------------------------------------
    training_geno_t  = torch.tensor(training_dataset,       dtype=torch.float32).unsqueeze(1)
    training_pheno_t = torch.tensor(training_pheno_norm,    dtype=torch.float32).unsqueeze(1)
    training_pop_t   = torch.cat([
        torch.zeros(n_disc_train,   dtype=torch.long),
        torch.ones(n_target_train,  dtype=torch.long),
    ])

    train_ds     = TensorDataset(training_geno_t, training_pheno_t, training_pop_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # build EVALUATION loaders (no gradient updates, plots only)
    # Tuple order: (input_x, original_x, pheno, mask, pop_label)
    # ------------------------------------------------------------------
    disc_train_loader = make_eval_loader(
        geno=disc_train_dataset,
        pheno_norm=disc_train_pheno_norm,
        pop_label_value=0,
        batch_size=batch_size,
        masker=masker,
        masking=masking,
        out_dir=out,
        split_name="discovery_train",
    )

    target_train_loader = make_eval_loader(
        geno=target_train_dataset,
        pheno_norm=target_train_pheno_norm,
        pop_label_value=1,
        batch_size=batch_size,
        masker=masker,
        masking=masking,
        out_dir=out,
        split_name="target_train",
    )

    val_loader = make_eval_loader(
        geno=validation_dataset,
        pheno_norm=validation_pheno_norm,
        pop_label_value=0,
        batch_size=batch_size,
        masker=masker,
        masking=masking,
        out_dir=out,
        split_name="discovery_validation",
    )

    # ------------------------------------------------------------------
    # diagnostic heatmap (validation)
    # ------------------------------------------------------------------
    val_geno_t  = torch.tensor(validation_dataset, dtype=torch.float32).unsqueeze(1)
    val_input_x = next(iter(val_loader))[0]  # masked or unmasked input
    val_mask    = next(iter(val_loader))[3]

    plot_example_input_heatmap(
        original_x=val_geno_t,
        masked_x=val_input_x,
        mask=val_mask,
        output_path=out / "example_input_heatmap_val.png",
        sample_indices=(0, 1, 2, 3, 4),
        snp_start=0,
        snp_count=1000,
    )

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
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # shape trace
    x_batch = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        logits, mu, logvar, z, pheno_pred = model(x_batch, verbose=True)

    # optimizer
    pheno_head_params = set(model.pheno_head.parameters())
    other_params      = [p for p in model.parameters() if p not in pheno_head_params]
    optimizer = torch.optim.Adam([
        {"params": other_params,            "weight_decay": 0.0},
        {"params": list(pheno_head_params), "weight_decay": pheno_weight_decay},
    ], lr=learning_rate)

    # ------------------------------------------------------------------
    # training state
    # ------------------------------------------------------------------
    train_loss_list, train_recon_unmasked_list = [], []
    train_recon_masked_list, train_kl_list     = [], []
    train_phenotype_loss_list                  = []

    val_loss_list, val_recon_unmasked_list     = [], []
    val_recon_masked_list, val_kl_list         = [], []
    val_phenotype_loss_list                    = []

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
        train_loss, train_recon_unmasked, train_recon_masked, train_kl, train_pheno = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=vae_loss,
            masker=masker,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
        )

        val_loss, val_recon_unmasked, val_recon_masked, val_kl, val_pheno = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=vae_loss,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
        )

        val_stop_metric = (
            val_recon_unmasked
            + alpha * val_recon_masked
            + beta  * val_kl
            + gamma * val_pheno
        )

        train_loss_list.append(train_loss)
        train_recon_unmasked_list.append(train_recon_unmasked)
        train_recon_masked_list.append(train_recon_masked)
        train_kl_list.append(train_kl)
        train_phenotype_loss_list.append(train_pheno)

        val_loss_list.append(val_loss)
        val_recon_unmasked_list.append(val_recon_unmasked)
        val_recon_masked_list.append(val_recon_masked)
        val_kl_list.append(val_kl)
        val_phenotype_loss_list.append(val_pheno)

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train_recon={train_recon_unmasked:.6f} | train_pheno={train_pheno:.6f} | "
            f"val_recon={val_recon_unmasked:.6f} | val_pheno={val_pheno:.6f}"
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
            print(f"Early stopping at epoch {epoch + 1}")
            break

    stopped_epoch = len(train_loss_list)
    save_checkpoint(
        path=final_model_path,
        model=model,
        optimizer=optimizer,
        epoch=stopped_epoch,
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
        train_recon_masked_losses=np.array(train_recon_masked_list),
        val_recon_masked_losses=np.array(val_recon_masked_list),
    )

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
    # reload best model
    # ------------------------------------------------------------------
    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(
        f"Reloaded best model from epoch {best_ckpt['epoch']} "
        f"with stop_metric={best_ckpt['val_loss']:.6f}"
    )

    use_masked = masking

    # ------------------------------------------------------------------
    # per-split evaluation plots
    # ------------------------------------------------------------------
    run_eval_plots(
        model=model, loader=disc_train_loader, device=device,
        out_dir=plots_dir, split_name="discovery_train",
        use_masked=use_masked, loss_fn=vae_loss,
        alpha=alpha, beta=beta, gamma=gamma,
    )

    run_eval_plots(
        model=model, loader=target_train_loader, device=device,
        out_dir=plots_dir, split_name="target_train",
        use_masked=use_masked, loss_fn=vae_loss,
        alpha=alpha, beta=beta, gamma=gamma,
    )

    run_eval_plots(
        model=model, loader=val_loader, device=device,
        out_dir=plots_dir, split_name="discovery_validation",
        use_masked=use_masked, loss_fn=vae_loss,
        alpha=alpha, beta=beta, gamma=gamma,
    )

    # ------------------------------------------------------------------
    # shared-coordinate latent PCA across all three splits
    # ------------------------------------------------------------------
    disc_train_mu, _ = extract_mu(model, disc_train_loader,   device, use_masked_input=use_masked)
    target_train_mu, _ = extract_mu(model, target_train_loader, device, use_masked_input=use_masked)
    val_mu, _          = extract_mu(model, val_loader,          device, use_masked_input=use_masked)

    plot_latent_pca_shared_basis(
        reference_mu=disc_train_mu,
        ceu_mu=val_mu,
        yri_mu=target_train_mu,
        output_path=plots_dir / "latent_pca_shared_basis.png",
        reference_name="CEU discovery train",
        ceu_name="CEU validation",
        yri_name="YRI target train",
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train ConvVAE on mixed CEU+YRI genotypes; evaluate per population."
    )
    p.add_argument("--vae-config",           type=Path, required=True)
    p.add_argument("--training-data",        type=Path, required=True,
                   help="Stacked CEU+YRI genotypes: training.npy")
    p.add_argument("--disc-train-data",      type=Path, required=True,
                   help="CEU-only training genotypes: discovery_train.npy (eval only)")
    p.add_argument("--target-train-data",    type=Path, required=True,
                   help="YRI-only training genotypes: target_train.npy (eval only)")
    p.add_argument("--validation-data",      type=Path, required=True,
                   help="CEU validation genotypes: discovery_validation.npy")
    p.add_argument("--disc-train-pheno",     type=Path, required=True,
                   help="CEU training phenotypes: simulated_phenotype_disc_train.npy")
    p.add_argument("--target-train-pheno",   type=Path, required=True,
                   help="YRI training phenotypes: simulated_phenotype_target_train.npy")
    p.add_argument("--validation-pheno",     type=Path, required=True,
                   help="CEU validation phenotypes: simulated_phenotype_disc_val.npy")
    p.add_argument("--outputs",              type=Path, required=True)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(
        vae_config_path=args.vae_config,
        training_data_path=args.training_data,
        disc_train_data_path=args.disc_train_data,
        target_train_data_path=args.target_train_data,
        validation_data_path=args.validation_data,
        disc_train_pheno_path=args.disc_train_pheno,
        target_train_pheno_path=args.target_train_pheno,
        validation_pheno_path=args.validation_pheno,
        output_dir=args.outputs,
    )