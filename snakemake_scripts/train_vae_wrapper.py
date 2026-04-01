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


def generate_plots(
    phase: str,
    out: Path,
    model,
    val_loader,
    test_target_loader,
    train_loader,
    device,
    use_masked: bool,
    train_loss_list: list,
    val_loss_list: list,
    train_recon_unmasked_list: list,
    val_recon_unmasked_list: list,
    train_kl_list: list,
    val_kl_list: list,
    train_phenotype_loss_list: list,
    val_phenotype_loss_list: list,
    train_recon_masked_list: list | None,
    val_recon_masked_list: list | None,
    masking: bool,
    alpha: float,
    beta: float,
    gamma: float,
):
    """
    Generate the full suite of diagnostic plots for a given phase
    ('pretrain' or 'finetune'). All outputs land in out/plots/{phase}/.
    """
    phase_dir = out / "plots" / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    tt_dir = phase_dir / "test_target"
    tt_dir.mkdir(exist_ok=True)

    print(f"\n--- Generating plots for phase: {phase} ---")

    # ---- loss curves -------------------------------------------------------
    plot_loss_curves(
        train_losses=train_loss_list,
        val_losses=val_loss_list,
        train_recon_unmasked_losses=train_recon_unmasked_list,
        val_recon_unmasked_losses=val_recon_unmasked_list,
        train_kl_losses=train_kl_list,
        val_kl_losses=val_kl_list,
        train_pheno_losses=train_phenotype_loss_list,
        val_pheno_losses=val_phenotype_loss_list,
        output_dir=phase_dir,
        train_recon_masked_losses=train_recon_masked_list if masking else None,
        val_recon_masked_losses=val_recon_masked_list if masking else None,
    )

    # ---- validation --------------------------------------------------------
    val_metrics = plot_reconstruction(
        model=model,
        dataloader=val_loader,
        device=device,
        output_dir=phase_dir,
        use_masked_input=use_masked,
    )
    print(f"[{phase}] Val balanced accuracy: {val_metrics['balanced_accuracy']:.6f}")

    plot_latent_space(
        model=model,
        dataloader=val_loader,
        device=device,
        output_dir=phase_dir,
        save_path="latent_space_val.png",
        use_masked_input=use_masked,
    )

    val_pheno = plot_pheno_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        output_path=phase_dir / "pheno_pred_vs_true_val.png",
        use_masked_input=use_masked,
        title=f"[{phase}] Validation phenotype prediction",
    )
    print(f"[{phase}] Val RMSE={val_pheno['rmse']:.6f}  R²={val_pheno['r2']:.6f}")

    plot_pheno_predictions_by_population(
        model=model,
        dataloader=val_loader,
        device=device,
        output_path=phase_dir / "pheno_pred_vs_true_val_by_population.png",
        use_masked_input=use_masked,
        title=f"[{phase}] Validation phenotype prediction by population",
    )

    plot_pheno_residuals(
        model=model,
        dataloader=val_loader,
        device=device,
        output_path=phase_dir / "pheno_residuals_val.png",
        use_masked_input=use_masked,
        title=f"[{phase}] Validation phenotype residuals",
    )

    # ---- test target (YRI) -------------------------------------------------
    tt_loss, tt_recon, tt_recon_m, tt_kl, tt_pheno_loss = evaluate(
        model=model,
        dataloader=test_target_loader,
        device=device,
        loss_fn=vae_loss,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    print(
        f"[{phase}] Test target: loss={tt_loss:.6f}  recon={tt_recon:.6f}"
        f"  kl={tt_kl:.6f}  pheno={tt_pheno_loss:.6f}"
    )

    tt_metrics = plot_reconstruction(
        model=model,
        dataloader=test_target_loader,
        device=device,
        output_dir=tt_dir,
        use_masked_input=use_masked,
    )
    print(f"[{phase}] Test target balanced accuracy: {tt_metrics['balanced_accuracy']:.6f}")

    plot_latent_space(
        model=model,
        dataloader=test_target_loader,
        device=device,
        output_dir=tt_dir,
        save_path="latent_space_test_target.png",
        use_masked_input=use_masked,
    )

    tt_pheno_metrics = plot_pheno_predictions(
        model=model,
        dataloader=test_target_loader,
        device=device,
        output_path=tt_dir / "pheno_pred_vs_true_test_target.png",
        use_masked_input=use_masked,
        title=f"[{phase}] Test target phenotype prediction",
    )
    print(
        f"[{phase}] Test target RMSE={tt_pheno_metrics['rmse']:.6f}"
        f"  R²={tt_pheno_metrics['r2']:.6f}"
    )

    plot_pheno_predictions_by_population(
        model=model,
        dataloader=test_target_loader,
        device=device,
        output_path=tt_dir / "pheno_pred_vs_true_test_target_by_population.png",
        use_masked_input=use_masked,
        title=f"[{phase}] Test target phenotype prediction by population",
    )

    plot_pheno_residuals(
        model=model,
        dataloader=test_target_loader,
        device=device,
        output_path=tt_dir / "pheno_residuals_test_target.png",
        use_masked_input=use_masked,
        title=f"[{phase}] Test target phenotype residuals",
    )

    # ---- shared-coordinate latent PCA across all three splits --------------
    # train_loader is mixed (CEU + YRI); extract_mu returns mu for all rows.
    # The pop_label tensor in train_loader is at index 2, so we need a small
    # helper to split CEU vs YRI after extraction.
    train_mu, train_pop = extract_mu(
        model=model,
        dataloader=train_loader,
        device=device,
        use_masked_input=False,
    )
    val_mu, _ = extract_mu(
        model=model,
        dataloader=val_loader,
        device=device,
        use_masked_input=use_masked,
    )
    tt_mu, _ = extract_mu(
        model=model,
        dataloader=test_target_loader,
        device=device,
        use_masked_input=use_masked,
    )

    # Split train_mu into CEU (pop=0) and YRI (pop=1) components.
    # train_pop is expected to be a 1-D int array of population labels.
    if train_pop is not None:
        ceu_train_mask = train_pop == 0
        yri_train_mask = train_pop == 1
        ceu_train_mu = train_mu[ceu_train_mask]
        yri_train_mu = train_mu[yri_train_mask]
    else:
        ceu_train_mu = train_mu
        yri_train_mu = None

    # PCA plot: CEU train as reference basis
    plot_latent_pca_shared_basis(
        reference_mu=ceu_train_mu,
        ceu_mu=val_mu,
        yri_mu=tt_mu,
        output_path=phase_dir / "latent_pca_ceu_basis_val_vs_test_target.png",
        reference_name="CEU discovery train",
        ceu_name="CEU validation",
        yri_name="YRI test target",
    )

    # Optional: PCA plot also overlaying YRI training individuals so we can
    # see where unlabelled YRI training samples sit in the latent space.
    if yri_train_mu is not None:
        try:
            plot_latent_pca_shared_basis(
                reference_mu=ceu_train_mu,
                ceu_mu=val_mu,
                yri_mu=yri_train_mu,
                output_path=phase_dir / "latent_pca_ceu_basis_val_vs_yri_train.png",
                reference_name="CEU discovery train",
                ceu_name="CEU validation",
                yri_name="YRI train (unlabelled)",
            )
        except Exception as e:
            print(f"[{phase}] Could not generate YRI-train PCA plot: {e}")


def run_phase(
    phase: str,
    model,
    optimizer,
    train_loader,
    val_loader,
    device,
    masker,
    loss_fn,
    num_epochs: int,
    patience: int,
    min_delta: float,
    beta: float,
    alpha: float,
    gamma: float,
    checkpoint_dir: Path,
    vae_config: dict,
    input_length: int,
) -> dict:
    """
    Run one training phase (pretrain or finetune).

    Returns a dict with keys:
        train_losses, val_losses,
        train_recon_unmasked, val_recon_unmasked,
        train_recon_masked, val_recon_masked,
        train_kl, val_kl,
        train_pheno, val_pheno,
        best_epoch, best_val_stop_metric
    """
    print(f"\n{'='*60}")
    print(f"  Phase: {phase}  |  epochs: {num_epochs}  |  gamma: {gamma}")
    print(f"{'='*60}")

    train_loss_list, val_loss_list = [], []
    train_recon_unmasked_list, val_recon_unmasked_list = [], []
    train_recon_masked_list, val_recon_masked_list = [], []
    train_kl_list, val_kl_list = [], []
    train_phenotype_loss_list, val_phenotype_loss_list = [], []

    best_val_stop_metric = float("inf")
    best_model_path = checkpoint_dir / f"best_model_{phase}.pt"
    final_model_path = checkpoint_dir / f"final_model_{phase}.pt"
    epochs_without_improvement = 0
    best_epoch = 0
    val_stop_metric = float("inf")

    for epoch in range(num_epochs):
        (
            train_loss,
            train_recon_unmasked,
            train_recon_masked,
            train_kl,
            train_phenotype_loss,
        ) = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            masker=masker,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
        )

        val_loss, val_recon_unmasked, val_recon_masked, val_kl, val_phenotype_loss = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
        )

        val_stop_metric = (
            val_recon_unmasked
            + alpha * val_recon_masked
            + beta * val_kl
            + gamma * val_phenotype_loss
        )

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_recon_unmasked_list.append(train_recon_unmasked)
        val_recon_unmasked_list.append(val_recon_unmasked)
        train_recon_masked_list.append(train_recon_masked)
        val_recon_masked_list.append(val_recon_masked)
        train_kl_list.append(train_kl)
        val_kl_list.append(val_kl)
        train_phenotype_loss_list.append(train_phenotype_loss)
        val_phenotype_loss_list.append(val_phenotype_loss)

        print(
            f"[{phase}] Epoch {epoch+1:04d}/{num_epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"recon={train_recon_unmasked:.4f} pheno={train_phenotype_loss:.4f} | "
            f"val_recon={val_recon_unmasked:.4f} val_pheno={val_phenotype_loss:.4f}"
        )

        if val_stop_metric < (best_val_stop_metric - min_delta):
            best_val_stop_metric = val_stop_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                epoch + 1,
                val_stop_metric,
                vae_config,
                input_length,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for {epochs_without_improvement} epoch(s) "
                f"(best={best_val_stop_metric:.6f} at epoch {best_epoch})"
            )

        if epochs_without_improvement >= patience:
            print(f"[{phase}] Early stopping triggered at epoch {epoch + 1}")
            break

    save_checkpoint(
        final_model_path,
        model,
        optimizer,
        len(train_loss_list),
        val_stop_metric,
        vae_config,
        input_length,
    )
    print(f"[{phase}] Saved final checkpoint → {final_model_path}")

    return dict(
        train_losses=train_loss_list,
        val_losses=val_loss_list,
        train_recon_unmasked=train_recon_unmasked_list,
        val_recon_unmasked=val_recon_unmasked_list,
        train_recon_masked=train_recon_masked_list,
        val_recon_masked=val_recon_masked_list,
        train_kl=train_kl_list,
        val_kl=val_kl_list,
        train_pheno=train_phenotype_loss_list,
        val_pheno=val_phenotype_loss_list,
        best_epoch=best_epoch,
        best_val_stop_metric=best_val_stop_metric,
        best_model_path=best_model_path,
        final_model_path=final_model_path,
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
    patience = int(vae_config["training"].get("patience", 500))
    min_delta = float(vae_config["training"].get("min_delta", 1e-4))
    pretrain_epochs = int(vae_config["training"].get("pretrain_epochs", 0))

    masking = vae_config["masking"].get("enabled", False)
    alpha = float(vae_config["masking"]["alpha_masked"])
    block_length = int(vae_config["masking"]["block_len"])
    mask_frac = float(vae_config["masking"]["mask_frac"])

    pheno_hidden_dim = vae_config["phenotype"].get("pheno_hidden_dim", None)
    gamma = float(vae_config["phenotype"].get("gamma", 1.0))

    print(f"Masking enabled:  {masking}")
    print(f"pretrain_epochs:  {pretrain_epochs}")
    print(f"gamma:            {gamma}")

    masker = (
        Masker(block_length=block_length, mask_fraction=mask_frac) if masking else None
    )
    if not masking:
        print("Masking is disabled.")

    # ------------------------------------------------------------------
    # load + normalize
    # ------------------------------------------------------------------
    training_dataset = np.load(training_data_path)
    validation_dataset = np.load(validation_data_path)
    train_target_dataset = np.load(train_target_data_path)
    test_target_dataset = np.load(test_target_data_path)

    training_pheno = np.load(training_pheno_path)
    validation_pheno = np.load(validation_pheno_path)
    test_target_pheno = np.load(test_target_pheno_path)

    train_mean, train_std = training_pheno.mean(), training_pheno.std()
    training_pheno = (training_pheno - train_mean) / train_std
    validation_pheno = (validation_pheno - train_mean) / train_std
    test_target_pheno = (test_target_pheno - train_mean) / train_std

    input_length = training_dataset.shape[-1]

    # ------------------------------------------------------------------
    # tensorify
    # ------------------------------------------------------------------
    training_dataset_torch = torch.tensor(training_dataset, dtype=torch.float32).unsqueeze(1)
    validation_dataset_torch = torch.tensor(validation_dataset, dtype=torch.float32).unsqueeze(1)
    train_target_dataset_torch = torch.tensor(train_target_dataset, dtype=torch.float32).unsqueeze(1)
    test_target_dataset_torch = torch.tensor(test_target_dataset, dtype=torch.float32).unsqueeze(1)

    training_pheno_torch = torch.tensor(training_pheno, dtype=torch.float32).unsqueeze(1)
    validation_pheno_torch = torch.tensor(validation_pheno, dtype=torch.float32).unsqueeze(1)
    test_target_pheno_torch = torch.tensor(test_target_pheno, dtype=torch.float32).unsqueeze(1)
    # YRI phenotypes are masked from the loss — use zeros as placeholder
    train_target_pheno_torch = torch.zeros(len(train_target_dataset_torch), 1, dtype=torch.float32)

    # ------------------------------------------------------------------
    # masking / identity for val + test_target
    # ------------------------------------------------------------------
    if masking:
        val_input_x, val_mask = masker.mask(validation_dataset_torch)
        test_target_input_x, test_target_mask = masker.mask(test_target_dataset_torch)
        np.save(out / "masked_validation_dataset.npy", val_input_x.numpy())
        np.save(out / "validation_masks.npy", val_mask.numpy())
        np.save(out / "masked_test_target_dataset.npy", test_target_input_x.numpy())
        np.save(out / "test_target_masks.npy", test_target_mask.numpy())
    else:
        val_input_x = validation_dataset_torch
        val_mask = torch.zeros(
            validation_dataset_torch.shape[0],
            validation_dataset_torch.shape[2],
            dtype=torch.bool,
        )
        test_target_input_x = test_target_dataset_torch
        test_target_mask = torch.zeros(
            test_target_dataset_torch.shape[0],
            test_target_dataset_torch.shape[2],
            dtype=torch.bool,
        )
        np.save(out / "validation_dataset.npy", val_input_x.numpy())
        np.save(out / "test_target_dataset.npy", test_target_input_x.numpy())

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
    # train_loader : (x, pheno, pop_label)   CEU=0, YRI=1  (mixed)
    # val_loader   : (input_x, x, pheno, mask, pop_label)  all pop=0
    # test_target_loader : (input_x, x, pheno, mask, pop_label)  all pop=1
    #
    # We also build separate CEU-only and YRI-only train loaders so that
    # per-population plots on the training set are feasible.
    # ------------------------------------------------------------------
    train_ds = TensorDataset(
        torch.cat([training_dataset_torch, train_target_dataset_torch], dim=0),
        torch.cat([training_pheno_torch, train_target_pheno_torch], dim=0),
        torch.cat(
            [
                torch.zeros(len(training_dataset_torch), dtype=torch.long),
                torch.ones(len(train_target_dataset_torch), dtype=torch.long),
            ],
            dim=0,
        ),
    )

    # CEU-only training loader — same format as val/test loaders so that
    # plot functions that expect (input_x, x, pheno, mask, pop) work directly.
    ceu_train_mask_t = torch.zeros(
        training_dataset_torch.shape[0],
        training_dataset_torch.shape[2],
        dtype=torch.bool,
    )
    ceu_train_ds = TensorDataset(
        training_dataset_torch,                                     # input_x (unmasked)
        training_dataset_torch,                                     # x (target)
        training_pheno_torch,                                       # pheno
        ceu_train_mask_t,                                           # mask (all False)
        torch.zeros(len(training_dataset_torch), dtype=torch.long), # pop_label = 0
    )

    # YRI-only training loader (unlabelled — pheno is zeros placeholder)
    yri_train_mask_t = torch.zeros(
        train_target_dataset_torch.shape[0],
        train_target_dataset_torch.shape[2],
        dtype=torch.bool,
    )
    yri_train_ds = TensorDataset(
        train_target_dataset_torch,
        train_target_dataset_torch,
        train_target_pheno_torch,
        yri_train_mask_t,
        torch.ones(len(train_target_dataset_torch), dtype=torch.long),
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    ceu_train_loader = DataLoader(ceu_train_ds, batch_size=batch_size, shuffle=False)
    yri_train_loader = DataLoader(yri_train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_target_loader = DataLoader(test_target_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
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
        pheno_dim=1,
        pheno_hidden_dim=pheno_hidden_dim,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    x_batch = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        logits, mu, logvar, z, pheno_pred = model(x_batch, verbose=True)
    print(f"logits={logits.shape}  mu={mu.shape}  pheno={pheno_pred.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    use_masked = masking

    # ==================================================================
    # Phase 1: Pretraining (reconstruction only, gamma=0)
    # ==================================================================
    all_history = {}

    if pretrain_epochs > 0:
        pretrain_history = run_phase(
            phase="pretrain",
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            masker=masker,
            loss_fn=vae_loss,
            num_epochs=pretrain_epochs,
            patience=patience,
            min_delta=min_delta,
            beta=beta,
            alpha=alpha,
            gamma=0.0,           # reconstruction only
            checkpoint_dir=checkpoint_dir,
            vae_config=vae_config,
            input_length=input_length,
        )
        all_history["pretrain"] = pretrain_history

        # reload best pretrain model before generating plots
        best_ckpt = torch.load(pretrain_history["best_model_path"], map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(
            f"[pretrain] Reloaded best model from epoch {best_ckpt['epoch']}"
            f" (stop={best_ckpt['val_loss']:.6f})"
        )

        generate_plots(
            phase="pretrain",
            out=out,
            model=model,
            val_loader=val_loader,
            test_target_loader=test_target_loader,
            train_loader=ceu_train_loader,   # pass CEU-only loader for PCA split
            device=device,
            use_masked=use_masked,
            train_loss_list=pretrain_history["train_losses"],
            val_loss_list=pretrain_history["val_losses"],
            train_recon_unmasked_list=pretrain_history["train_recon_unmasked"],
            val_recon_unmasked_list=pretrain_history["val_recon_unmasked"],
            train_kl_list=pretrain_history["train_kl"],
            val_kl_list=pretrain_history["val_kl"],
            train_phenotype_loss_list=pretrain_history["train_pheno"],
            val_phenotype_loss_list=pretrain_history["val_pheno"],
            train_recon_masked_list=pretrain_history["train_recon_masked"],
            val_recon_masked_list=pretrain_history["val_recon_masked"],
            masking=masking,
            alpha=alpha,
            beta=beta,
            gamma=0.0,
        )

        np.savez(
            out / "training_history_pretrain.npz",
            train_losses=np.array(pretrain_history["train_losses"]),
            val_losses=np.array(pretrain_history["val_losses"]),
            train_recon_unmasked_losses=np.array(pretrain_history["train_recon_unmasked"]),
            val_recon_unmasked_losses=np.array(pretrain_history["val_recon_unmasked"]),
            train_kl_losses=np.array(pretrain_history["train_kl"]),
            val_kl_losses=np.array(pretrain_history["val_kl"]),
            train_phenotype_losses=np.array(pretrain_history["train_pheno"]),
            val_phenotype_losses=np.array(pretrain_history["val_pheno"]),
            train_recon_masked_losses=np.array(pretrain_history["train_recon_masked"]),
            val_recon_masked_losses=np.array(pretrain_history["val_recon_masked"]),
        )
        print(f"Saved pretrain history → {out / 'training_history_pretrain.npz'}")

    # ==================================================================
    # Phase 2: Fine-tuning (reconstruction + phenotype supervision)
    # ==================================================================
    finetune_epochs = num_epochs if pretrain_epochs == 0 else num_epochs - pretrain_epochs
    if finetune_epochs <= 0:
        print("Warning: no fine-tuning epochs remaining after pretrain. Skipping finetune phase.")
    else:
        # Reset early stopping state — the loss jumps when gamma switches on,
        # so we don't want the pretrain best metric to trigger immediate stopping.
        finetune_history = run_phase(
            phase="finetune",
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            masker=masker,
            loss_fn=vae_loss,
            num_epochs=finetune_epochs,
            patience=patience,
            min_delta=min_delta,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
            checkpoint_dir=checkpoint_dir,
            vae_config=vae_config,
            input_length=input_length,
        )
        all_history["finetune"] = finetune_history

        # reload best finetune model before generating plots
        best_ckpt = torch.load(finetune_history["best_model_path"], map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(
            f"[finetune] Reloaded best model from epoch {best_ckpt['epoch']}"
            f" (stop={best_ckpt['val_loss']:.6f})"
        )

        generate_plots(
            phase="finetune",
            out=out,
            model=model,
            val_loader=val_loader,
            test_target_loader=test_target_loader,
            train_loader=ceu_train_loader,
            device=device,
            use_masked=use_masked,
            train_loss_list=finetune_history["train_losses"],
            val_loss_list=finetune_history["val_losses"],
            train_recon_unmasked_list=finetune_history["train_recon_unmasked"],
            val_recon_unmasked_list=finetune_history["val_recon_unmasked"],
            train_kl_list=finetune_history["train_kl"],
            val_kl_list=finetune_history["val_kl"],
            train_phenotype_loss_list=finetune_history["train_pheno"],
            val_phenotype_loss_list=finetune_history["val_pheno"],
            train_recon_masked_list=finetune_history["train_recon_masked"],
            val_recon_masked_list=finetune_history["val_recon_masked"],
            masking=masking,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        np.savez(
            out / "training_history_finetune.npz",
            train_losses=np.array(finetune_history["train_losses"]),
            val_losses=np.array(finetune_history["val_losses"]),
            train_recon_unmasked_losses=np.array(finetune_history["train_recon_unmasked"]),
            val_recon_unmasked_losses=np.array(finetune_history["val_recon_unmasked"]),
            train_kl_losses=np.array(finetune_history["train_kl"]),
            val_kl_losses=np.array(finetune_history["val_kl"]),
            train_phenotype_losses=np.array(finetune_history["train_pheno"]),
            val_phenotype_losses=np.array(finetune_history["val_pheno"]),
            train_recon_masked_losses=np.array(finetune_history["train_recon_masked"]),
            val_recon_masked_losses=np.array(finetune_history["val_recon_masked"]),
        )
        print(f"Saved finetune history → {out / 'training_history_finetune.npz'}")

    # ------------------------------------------------------------------
    # Legacy combined history (for backward compat with downstream rules)
    # ------------------------------------------------------------------
    combined_train = []
    combined_val = []
    for phase_key in ("pretrain", "finetune"):
        if phase_key in all_history:
            combined_train.extend(all_history[phase_key]["train_losses"])
            combined_val.extend(all_history[phase_key]["val_losses"])

    np.savez(
        out / "training_history.npz",
        train_losses=np.array(combined_train),
        val_losses=np.array(combined_val),
    )
    print(f"Saved combined history → {out / 'training_history.npz'}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-config", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--train-target-data", type=Path, required=True)
    parser.add_argument("--test-target-data", type=Path, required=True)
    parser.add_argument("--training-pheno", type=Path, required=True)
    parser.add_argument("--validation-pheno", type=Path, required=True)
    parser.add_argument("--test-target-pheno", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
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