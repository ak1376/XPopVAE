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
    --disc-train-pheno          : CEU training phenotypes
    --target-train-pheno        : YRI training phenotypes
    --validation-pheno          : CEU validation phenotypes
    --outputs                   : root output directory

Training loader  : training.npy  (CEU + YRI mixed, or CEU-only if target_held_out_frac=1.0)
                   phenotype loss masked to CEU rows only (pop_label == 0)
Evaluation loaders (plots + metrics only, no gradient updates):
    discovery_train/       <- discovery_train.npy
    target_train/          <- target_train.npy         (skipped if empty)
    target_held_out/       <- target_held_out.npy      (always evaluated if file exists)
    discovery_validation/  <- discovery_validation.npy
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
    plot_loss_curves,
    plot_latent_pca_shared_basis,
)
from src.train import evaluate, train_one_epoch, compute_grl_lambda

from src.utils import (
    extract_latent,
    load_vae_config,
    save_checkpoint,
    make_eval_loader,
    run_eval_plots,
    extract_latent_with_pheno,
)

# =============================================================================
# Main
# =============================================================================


def main(
    vae_config_path: Path,
    training_data_path: Path,
    disc_train_data_path: Path,
    target_train_data_path: Path,
    validation_data_path: Path,
    disc_train_pheno_path: Path,
    target_train_pheno_path: Path,
    validation_pheno_path: Path,
    output_dir: Path,
):
    vae_config = load_vae_config(vae_config_path)

    # ------------------------------------------------------------------
    # output directories
    # ------------------------------------------------------------------
    out = output_dir / "vae_outputs"
    checkpoint_dir = out / "checkpoints"
    plots_dir = out / "plots"
    for d in (out, checkpoint_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to: {out.resolve()}")

    # ------------------------------------------------------------------
    # hyperparameters — model
    # ------------------------------------------------------------------
    in_channels = 1
    hidden_channels = vae_config["model"]["hidden_channels"]
    kernel_size = int(vae_config["model"]["kernel_size"])
    stride = int(vae_config["model"]["stride"])
    padding = int(vae_config["model"]["padding"])
    latent_dim = int(vae_config["model"]["latent_dim"])
    shared_dim = vae_config["model"].get(
        "shared_dim", None
    )  # defaults to latent_dim // 2

    # ------------------------------------------------------------------
    # hyperparameters — training
    # ------------------------------------------------------------------
    beta = float(vae_config["training"]["beta"])
    learning_rate = float(vae_config["training"]["lr"])
    batch_size = int(vae_config["training"]["batch_size"])
    num_epochs = int(vae_config["training"]["max_epochs"])
    patience = int(vae_config["training"].get("patience", 500))
    min_delta = float(vae_config["training"].get("min_delta", 1e-4))

    # ------------------------------------------------------------------
    # hyperparameters — masking
    # ------------------------------------------------------------------
    masking = vae_config["masking"].get("enabled", False)
    alpha = float(vae_config["masking"]["alpha_masked"])
    block_length = int(vae_config["masking"]["block_len"])
    mask_frac = float(vae_config["masking"]["mask_frac"])

    # ------------------------------------------------------------------
    # hyperparameters — phenotype head
    # ------------------------------------------------------------------
    pheno_hidden_dim = vae_config["phenotype"].get("pheno_hidden_dim", None)
    gamma = float(vae_config["phenotype"].get("gamma", 1.0))
    pheno_weight_decay = float(vae_config["phenotype"].get("pheno_weight_decay", 0.0))

    # ------------------------------------------------------------------
    # hyperparameters — domain adaptation (GRL)
    # ------------------------------------------------------------------
    da_cfg = vae_config.get("domain_adaptation", {})
    use_grl = bool(da_cfg.get("use_grl", False))
    raw = da_cfg.get("grl_hidden_dim", None)
    grl_hidden_dim = int(raw) if raw is not None else None
    grl_lambda_max = float(da_cfg.get("lambda_max", 1.0))
    delta = float(da_cfg.get("delta", 1.0))

    print(f"Masking enabled: {masking}")
    print(
        f"GRL enabled: {use_grl}"
        + (
            f"  lambda_max={grl_lambda_max}  grl_hidden_dim={grl_hidden_dim}  delta={delta}"
            if use_grl
            else ""
        )
    )
    print(
        f"Latent dim: {latent_dim}  shared_dim: {shared_dim if shared_dim is not None else latent_dim // 2} (default)"
    )

    # ------------------------------------------------------------------
    # load genotypes
    # ------------------------------------------------------------------
    training_dataset = np.load(training_data_path)
    disc_train_dataset = np.load(disc_train_data_path)
    target_train_dataset = np.load(target_train_data_path)
    validation_dataset = np.load(validation_data_path)

    n_disc_train = disc_train_dataset.shape[0]
    n_target_train = target_train_dataset.shape[0]
    assert n_disc_train + n_target_train == training_dataset.shape[0], (
        f"training.npy rows ({training_dataset.shape[0]}) != "
        f"discovery_train ({n_disc_train}) + target_train ({n_target_train})"
    )

    input_length = training_dataset.shape[-1]

    # ------------------------------------------------------------------
    # load + normalise phenotypes
    # ------------------------------------------------------------------
    disc_train_pheno = np.load(disc_train_pheno_path).astype(np.float32)
    target_train_pheno = np.load(target_train_pheno_path).astype(np.float32)
    validation_pheno = np.load(validation_pheno_path).astype(np.float32)

    train_mean = disc_train_pheno.mean()
    train_std = disc_train_pheno.std()
    print(
        f"Phenotype normalisation — mean={train_mean:.4f}  std={train_std:.4f}  (CEU disc_train only)"
    )

    disc_train_pheno_norm = (disc_train_pheno - train_mean) / train_std
    target_train_pheno_norm = (target_train_pheno - train_mean) / train_std
    validation_pheno_norm = (validation_pheno - train_mean) / train_std

    training_pheno_norm = np.concatenate(
        [disc_train_pheno_norm, np.zeros(n_target_train, dtype=np.float32)], axis=0
    )

    # ------------------------------------------------------------------
    # masker
    # ------------------------------------------------------------------
    masker = (
        Masker(block_length=block_length, mask_fraction=mask_frac) if masking else None
    )

    # ------------------------------------------------------------------
    # training loader
    # ------------------------------------------------------------------
    training_geno_t = torch.tensor(training_dataset, dtype=torch.float32).unsqueeze(1)
    training_pheno_t = torch.tensor(training_pheno_norm, dtype=torch.float32).unsqueeze(
        1
    )
    training_pop_t = torch.cat(
        [
            torch.zeros(n_disc_train, dtype=torch.long),
            torch.ones(n_target_train, dtype=torch.long),
        ]
    )

    train_ds = TensorDataset(training_geno_t, training_pheno_t, training_pop_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # evaluation loaders
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

    target_train_loader = (
        make_eval_loader(
            geno=target_train_dataset,
            pheno_norm=target_train_pheno_norm,
            pop_label_value=1,
            batch_size=batch_size,
            masker=masker,
            masking=masking,
            out_dir=out,
            split_name="target_train",
        )
        if n_target_train > 0
        else None
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

    # target_held_out: always build if file exists
    held_out_geno_path = target_train_data_path.parent / "target_held_out.npy"
    held_out_pheno_path = (
        target_train_data_path.parent.parent
        / "phenotypes"
        / "target_held_out_pheno.npy"
    )

    target_held_out_loader = None
    if held_out_geno_path.exists() and held_out_pheno_path.exists():
        held_out_geno = np.load(held_out_geno_path)
        held_out_pheno = np.load(held_out_pheno_path).astype(np.float32)
        held_out_pheno_norm = (held_out_pheno - train_mean) / train_std
        if len(held_out_geno) > 0:
            target_held_out_loader = make_eval_loader(
                geno=held_out_geno,
                pheno_norm=held_out_pheno_norm,
                pop_label_value=1,
                batch_size=batch_size,
                masker=masker,
                masking=masking,
                out_dir=out,
                split_name="target_held_out",
            )
            print(f"target_held_out loader built: {held_out_geno.shape}")
    else:
        print("target_held_out.npy not found — skipping held-out YRI eval")

    # ------------------------------------------------------------------
    # diagnostic heatmap
    # ------------------------------------------------------------------
    val_geno_t = torch.tensor(validation_dataset, dtype=torch.float32).unsqueeze(1)
    val_input_x = next(iter(val_loader))[0]
    val_mask = next(iter(val_loader))[3]

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
        use_grl=use_grl,
        grl_hidden_dim=grl_hidden_dim,
        num_domains=2,
        shared_dim=shared_dim,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Latent split: z_shared={model.shared_dim}  z_pop={latent_dim - model.shared_dim}"
    )

    x_batch = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        logits, mu, logvar, z, pheno_pred, domain_logits = model(x_batch, verbose=True)
    if domain_logits is not None:
        print(f"domain_logits: {domain_logits.shape}")

    # ------------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------------
    pheno_head_params = set(model.pheno_head.parameters())
    other_params = [p for p in model.parameters() if p not in pheno_head_params]
    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "weight_decay": 0.0},
            {"params": list(pheno_head_params), "weight_decay": pheno_weight_decay},
        ],
        lr=learning_rate,
    )

    # ------------------------------------------------------------------
    # training-history accumulators
    # ------------------------------------------------------------------
    train_loss_list, train_recon_unmasked_list = [], []
    train_recon_masked_list, train_kl_list = [], []
    train_phenotype_loss_list = []
    train_domain_loss_list = []
    train_domain_acc_list = []
    train_z_shared_var_list = []
    train_z_pop_var_list = []

    val_loss_list, val_recon_unmasked_list = [], []
    val_recon_masked_list, val_kl_list = [], []
    val_phenotype_loss_list = []

    best_val_stop_metric = float("inf")
    best_model_path = checkpoint_dir / "best_model.pt"
    final_model_path = checkpoint_dir / "final_model.pt"
    epochs_without_improvement = 0
    best_epoch = 0

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    for epoch in range(num_epochs):

        if use_grl:
            grl_lam = compute_grl_lambda(epoch, num_epochs, lambda_max=grl_lambda_max)
            model.set_grl_lambda(grl_lam)
        else:
            grl_lam = 0.0

        (
            train_loss,
            train_recon_unmasked,
            train_recon_masked,
            train_kl,
            train_pheno,
            train_domain,
            train_d_acc,
            train_z_shared_var,
            train_z_pop_var,
        ) = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=vae_loss,
            masker=masker,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
            use_grl=use_grl,
            delta=delta,
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
            + beta * val_kl
            + gamma * val_pheno
        )

        train_loss_list.append(train_loss)
        train_recon_unmasked_list.append(train_recon_unmasked)
        train_recon_masked_list.append(train_recon_masked)
        train_kl_list.append(train_kl)
        train_phenotype_loss_list.append(train_pheno)
        train_domain_loss_list.append(train_domain)
        train_domain_acc_list.append(train_d_acc)
        train_z_shared_var_list.append(train_z_shared_var)
        train_z_pop_var_list.append(train_z_pop_var)

        val_loss_list.append(val_loss)
        val_recon_unmasked_list.append(val_recon_unmasked)
        val_recon_masked_list.append(val_recon_masked)
        val_kl_list.append(val_kl)
        val_phenotype_loss_list.append(val_pheno)

        grl_str = (
            (
                f" | grl_lam={grl_lam:.3f}  domain_ce={train_domain:.4f}"
                f"  domain_acc={train_d_acc:.3f}"
            )
            if use_grl
            else ""
        )

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train_recon={train_recon_unmasked:.6f} | train_pheno={train_pheno:.6f} | "
            f"val_recon={val_recon_unmasked:.6f} | val_pheno={val_pheno:.6f}"
            + grl_str
            + f" | z_shared_var={train_z_shared_var:.4f}"
            + (
                f"  z_pop_var={train_z_pop_var:.4f}"
                if train_z_pop_var is not None
                else ""
            )
        )

        if val_stop_metric < (best_val_stop_metric - min_delta):
            best_val_stop_metric = val_stop_metric
            best_epoch = epoch + 1
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
    history_dict = dict(
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
        train_z_shared_var=np.array(train_z_shared_var_list),
        **(
            {"train_z_pop_var": np.array(train_z_pop_var_list)}
            if train_z_pop_var_list[0] is not None
            else {}
        ),
    )
    if use_grl:
        history_dict["train_domain_losses"] = np.array(train_domain_loss_list)
        history_dict["train_domain_acc"] = np.array(train_domain_acc_list)

    np.savez(out / "training_history.npz", **history_dict)

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
        val_recon_masked_losses=val_recon_masked_list if masking else None,
        train_domain_losses=train_domain_loss_list if use_grl else None,
        train_domain_accs=train_domain_acc_list if use_grl else None,
        train_z_shared_vars=train_z_shared_var_list,
        train_z_pop_vars=(
            train_z_pop_var_list if train_z_pop_var_list[0] is not None else None
        ),
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
        model=model,
        loader=disc_train_loader,
        device=device,
        out_dir=plots_dir,
        split_name="discovery_train",
        use_masked=use_masked,
        loss_fn=vae_loss,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    if target_train_loader is not None:
        run_eval_plots(
            model=model,
            loader=target_train_loader,
            device=device,
            out_dir=plots_dir,
            split_name="target_train",
            use_masked=use_masked,
            loss_fn=vae_loss,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    if target_held_out_loader is not None:
        run_eval_plots(
            model=model,
            loader=target_held_out_loader,
            device=device,
            out_dir=plots_dir,
            split_name="target_held_out",
            use_masked=use_masked,
            loss_fn=vae_loss,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    run_eval_plots(
        model=model,
        loader=val_loader,
        device=device,
        out_dir=plots_dir,
        split_name="discovery_validation",
        use_masked=use_masked,
        loss_fn=vae_loss,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    # ------------------------------------------------------------------
    # fit shared PCA basis on disc_train — used for ALL latent plots
    # ------------------------------------------------------------------
    from src.plotting import fit_latent_pca

    disc_train_latent, _, _, _ = extract_latent_with_pheno(
        model, disc_train_loader, device, use_masked_input=use_masked
    )
    shared_scaler, shared_pca = fit_latent_pca(disc_train_latent)

    # ------------------------------------------------------------------
    # per-split evaluation plots — all in shared PCA coordinate system
    # ------------------------------------------------------------------
    run_eval_plots(
        model=model,
        loader=disc_train_loader,
        device=device,
        out_dir=plots_dir,
        split_name="discovery_train",
        use_masked=use_masked,
        loss_fn=vae_loss,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        shared_scaler=shared_scaler,
        shared_pca=shared_pca,
    )

    if target_train_loader is not None:
        run_eval_plots(
            model=model,
            loader=target_train_loader,
            device=device,
            out_dir=plots_dir,
            split_name="target_train",
            use_masked=use_masked,
            loss_fn=vae_loss,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            shared_scaler=shared_scaler,
            shared_pca=shared_pca,
        )

    if target_held_out_loader is not None:
        run_eval_plots(
            model=model,
            loader=target_held_out_loader,
            device=device,
            out_dir=plots_dir,
            split_name="target_held_out",
            use_masked=use_masked,
            loss_fn=vae_loss,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            shared_scaler=shared_scaler,
            shared_pca=shared_pca,
        )

    run_eval_plots(
        model=model,
        loader=val_loader,
        device=device,
        out_dir=plots_dir,
        split_name="discovery_validation",
        use_masked=use_masked,
        loss_fn=vae_loss,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        shared_scaler=shared_scaler,
        shared_pca=shared_pca,
    )

    # ------------------------------------------------------------------
    # shared-coordinate latent PCA — population + phenotype coloring
    # ------------------------------------------------------------------
    val_latent, _, val_true, val_pred = extract_latent_with_pheno(
        model, val_loader, device, use_masked_input=use_masked
    )

    yri_latent = yri_true = yri_pred = None
    if target_held_out_loader is not None:
        yri_latent, _, yri_true, yri_pred = extract_latent_with_pheno(
            model, target_held_out_loader, device, use_masked_input=use_masked
        )
    elif target_train_loader is not None:
        yri_latent, _, yri_true, yri_pred = extract_latent_with_pheno(
            model, target_train_loader, device, use_masked_input=use_masked
        )

    if yri_latent is not None:
        plot_latent_pca_shared_basis(
            reference_vecs=disc_train_latent,
            ceu_vecs=val_latent,
            yri_vecs=yri_latent,
            output_path=plots_dir / "latent_pca_by_population.png",
            reference_name="CEU discovery train",
            ceu_name="CEU validation",
            yri_name="YRI target",
            scaler=shared_scaler,
            pca=shared_pca,
        )

        plot_latent_pca_shared_basis(
            reference_vecs=disc_train_latent,
            ceu_vecs=val_latent,
            yri_vecs=yri_latent,
            output_path=plots_dir / "latent_pca_by_true_pheno.png",
            reference_name="CEU discovery train",
            ceu_name="CEU validation",
            yri_name="YRI target",
            ceu_color_vec=val_true,
            yri_color_vec=yri_true,
            color_label="true phenotype",
            scaler=shared_scaler,
            pca=shared_pca,
        )

        plot_latent_pca_shared_basis(
            reference_vecs=disc_train_latent,
            ceu_vecs=val_latent,
            yri_vecs=yri_latent,
            output_path=plots_dir / "latent_pca_by_pred_pheno.png",
            reference_name="CEU discovery train",
            ceu_name="CEU validation",
            yri_name="YRI target",
            ceu_color_vec=val_pred,
            yri_color_vec=yri_pred,
            color_label="predicted phenotype",
            scaler=shared_scaler,
            pca=shared_pca,
        )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train ConvVAE on mixed CEU+YRI genotypes; evaluate per population."
    )
    p.add_argument("--vae-config", type=Path, required=True)
    p.add_argument("--training-data", type=Path, required=True)
    p.add_argument("--disc-train-data", type=Path, required=True)
    p.add_argument("--target-train-data", type=Path, required=True)
    p.add_argument("--validation-data", type=Path, required=True)
    p.add_argument("--disc-train-pheno", type=Path, required=True)
    p.add_argument("--target-train-pheno", type=Path, required=True)
    p.add_argument("--validation-pheno", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
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
