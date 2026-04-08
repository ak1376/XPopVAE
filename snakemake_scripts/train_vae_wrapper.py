#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

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
    extract_mu_shared,
)
from src.train import evaluate, train_one_epoch, compute_grl_lambda


# =============================================================================
# Helpers
# =============================================================================

def extract_mu_for_pca(model, dataloader, device, use_masked_input=False):
    """Return concatenated [mu_shared | mu_private] for shared-basis PCA."""
    from src.plotting import extract_mu
    return extract_mu(model, dataloader, device, use_masked_input)


def load_vae_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(path, model, optimizer, epoch, val_loss, vae_config, input_length):
    torch.save(
        {
            "epoch":              epoch,
            "val_loss":           val_loss,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vae_config":         vae_config,
            "input_length":       input_length,
        },
        path,
    )


def make_eval_loader(geno, pheno_norm, pop_label_value, batch_size,
                     masker, masking, out_dir, split_name):
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
    ds         = TensorDataset(input_x, geno_t, pheno_t, mask, pop_labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def run_eval_plots(model, loader, device, out_dir, split_name,
                   use_masked, loss_fn, alpha, beta, gamma):
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    loss, recon_u, recon_m, kl, pheno_l = evaluate(
        model=model, dataloader=loader, device=device,
        loss_fn=loss_fn, alpha=alpha, beta=beta, gamma=gamma,
    )
    print(f"\n[{split_name}] loss={loss:.6f}  recon={recon_u:.6f}  "
          f"kl={kl:.6f}  pheno={pheno_l:.6f}")

    recon_metrics = plot_reconstruction(
        model=model, dataloader=loader, device=device,
        output_dir=split_dir, use_masked_input=use_masked,
    )
    print(f"[{split_name}] balanced_accuracy={recon_metrics['balanced_accuracy']:.6f}")

    plot_latent_space(
        model=model, dataloader=loader, device=device,
        output_dir=split_dir, save_path="latent_space.png",
        use_masked_input=use_masked,
    )

    pheno_metrics = plot_pheno_predictions(
        model=model, dataloader=loader, device=device,
        output_path=split_dir / f"pheno_pred_vs_true_{split_name}.png",
        use_masked_input=use_masked,
        title=f"{split_name} phenotype prediction",
    )
    print(f"[{split_name}] pheno RMSE={pheno_metrics['rmse']:.6f}  R²={pheno_metrics['r2']:.6f}")

    plot_pheno_predictions_by_population(
        model=model, dataloader=loader, device=device,
        output_path=split_dir / f"pheno_pred_vs_true_{split_name}_by_pop.png",
        use_masked_input=use_masked,
        title=f"{split_name} phenotype prediction by population",
    )

    plot_pheno_residuals(
        model=model, dataloader=loader, device=device,
        output_path=split_dir / f"pheno_residuals_{split_name}.png",
        use_masked_input=use_masked,
        title=f"{split_name} phenotype residuals",
    )

    return recon_metrics, pheno_metrics


# =============================================================================
# Main
# =============================================================================

def main(
    vae_config_path, training_data_path, disc_train_data_path,
    target_train_data_path, validation_data_path,
    disc_train_pheno_path, target_train_pheno_path, validation_pheno_path,
    output_dir,
):
    vae_config = load_vae_config(vae_config_path)

    out            = output_dir / "vae_outputs"
    checkpoint_dir = out / "checkpoints"
    plots_dir      = out / "plots"
    for d in (out, checkpoint_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to: {out.resolve()}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    in_channels     = 1
    hidden_channels = vae_config["model"]["hidden_channels"]
    kernel_size     = int(vae_config["model"]["kernel_size"])
    stride          = int(vae_config["model"]["stride"])
    padding         = int(vae_config["model"]["padding"])

    # Split latent dims
    latent_dim_shared  = int(vae_config["model"]["latent_dim_shared"])
    latent_dim_private = int(vae_config["model"]["latent_dim_private"])

    beta             = float(vae_config["training"]["beta"])
    learning_rate    = float(vae_config["training"]["lr"])
    batch_size       = int(vae_config["training"]["batch_size"])
    num_epochs       = int(vae_config["training"]["max_epochs"])
    patience         = int(vae_config["training"].get("patience", 500))
    min_delta        = float(vae_config["training"].get("min_delta", 1e-4))

    masking      = vae_config["masking"].get("enabled", False)
    alpha        = float(vae_config["masking"]["alpha_masked"])
    block_length = int(vae_config["masking"]["block_len"])
    mask_frac    = float(vae_config["masking"]["mask_frac"])

    pheno_hidden_dim   = vae_config["phenotype"].get("pheno_hidden_dim", None)
    gamma              = float(vae_config["phenotype"].get("gamma", 1.0))
    pheno_weight_decay = float(vae_config["phenotype"].get("pheno_weight_decay", 0.0))

    da_cfg         = vae_config.get("domain_adaptation", {})
    use_grl        = bool(da_cfg.get("use_grl", False))
    raw_hidden     = da_cfg.get("grl_hidden_dim", None)
    grl_hidden_dim = int(raw_hidden) if raw_hidden is not None else None
    grl_lambda_max = float(da_cfg.get("lambda_max", 1.0))
    delta          = float(da_cfg.get("delta", 1.0))

    print(f"Masking enabled: {masking}")
    print(f"Latent split: shared={latent_dim_shared}  private={latent_dim_private}")
    print(f"GRL enabled: {use_grl}"
          + (f"  lambda_max={grl_lambda_max}  hidden={grl_hidden_dim}  delta={delta}"
             if use_grl else ""))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    training_dataset     = np.load(training_data_path)
    disc_train_dataset   = np.load(disc_train_data_path)
    target_train_dataset = np.load(target_train_data_path)
    validation_dataset   = np.load(validation_data_path)

    n_disc_train   = disc_train_dataset.shape[0]
    n_target_train = target_train_dataset.shape[0]
    assert n_disc_train + n_target_train == training_dataset.shape[0]

    input_length = training_dataset.shape[-1]

    disc_train_pheno   = np.load(disc_train_pheno_path).astype(np.float32)
    target_train_pheno = np.load(target_train_pheno_path).astype(np.float32)
    validation_pheno   = np.load(validation_pheno_path).astype(np.float32)

    train_mean = disc_train_pheno.mean()
    train_std  = disc_train_pheno.std()
    print(f"Phenotype normalisation — mean={train_mean:.4f}  std={train_std:.4f}")

    disc_train_pheno_norm   = (disc_train_pheno   - train_mean) / train_std
    target_train_pheno_norm = (target_train_pheno - train_mean) / train_std
    validation_pheno_norm   = (validation_pheno   - train_mean) / train_std
    training_pheno_norm     = np.concatenate(
        [disc_train_pheno_norm, np.zeros(n_target_train, dtype=np.float32)], axis=0
    )

    masker = Masker(block_length=block_length, mask_fraction=mask_frac) if masking else None

    # Training loader
    train_ds = TensorDataset(
        torch.tensor(training_dataset,    dtype=torch.float32).unsqueeze(1),
        torch.tensor(training_pheno_norm, dtype=torch.float32).unsqueeze(1),
        torch.cat([torch.zeros(n_disc_train, dtype=torch.long),
                   torch.ones(n_target_train, dtype=torch.long)]),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Eval loaders
    disc_train_loader = make_eval_loader(
        disc_train_dataset, disc_train_pheno_norm, 0,
        batch_size, masker, masking, out, "discovery_train",
    )
    target_train_loader = make_eval_loader(
        target_train_dataset, target_train_pheno_norm, 1,
        batch_size, masker, masking, out, "target_train",
    )
    val_loader = make_eval_loader(
        validation_dataset, validation_pheno_norm, 0,
        batch_size, masker, masking, out, "discovery_validation",
    )

    # Diagnostic heatmap
    val_geno_t  = torch.tensor(validation_dataset, dtype=torch.float32).unsqueeze(1)
    val_input_x = next(iter(val_loader))[0]
    val_mask    = next(iter(val_loader))[3]
    plot_example_input_heatmap(
        original_x=val_geno_t, masked_x=val_input_x, mask=val_mask,
        output_path=out / "example_input_heatmap_val.png",
        sample_indices=(0, 1, 2, 3, 4), snp_start=0, snp_count=1000,
    )

    # ------------------------------------------------------------------
    # Model
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
        latent_dim_shared=latent_dim_shared,
        latent_dim_private=latent_dim_private,
        use_batchnorm=False,
        activation="elu",
        pheno_dim=1,
        pheno_hidden_dim=pheno_hidden_dim,
        use_grl=use_grl,
        grl_hidden_dim=grl_hidden_dim,
        num_domains=2,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Shape trace
    x_batch = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        out_tuple = model(x_batch, verbose=True)
    _, mu_s, _, mu_p, _, _, _, _, d_logits = out_tuple
    print(f"mu_shared: {mu_s.shape}  mu_private: {mu_p.shape}")
    if d_logits is not None:
        print(f"domain_logits: {d_logits.shape}")

    # Optimizer: separate weight-decay for pheno head
    pheno_head_params = set(model.pheno_head.parameters())
    other_params      = [p for p in model.parameters() if p not in pheno_head_params]
    optimizer = torch.optim.Adam([
        {"params": other_params,            "weight_decay": 0.0},
        {"params": list(pheno_head_params), "weight_decay": pheno_weight_decay},
    ], lr=learning_rate)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_loss_list, train_recon_u_list = [], []
    train_recon_m_list, train_kl_list   = [], []
    train_pheno_list                    = []
    train_domain_list, train_dacc_list  = [], []

    val_loss_list, val_recon_u_list     = [], []
    val_recon_m_list, val_kl_list       = [], []
    val_pheno_list                      = []

    best_val_stop   = float("inf")
    best_model_path = checkpoint_dir / "best_model.pt"
    final_model_path = checkpoint_dir / "final_model.pt"
    no_improve      = 0
    best_epoch      = 0

    for epoch in range(num_epochs):
        if use_grl:
            grl_lam = compute_grl_lambda(epoch, num_epochs, lambda_max=grl_lambda_max)
            model.set_grl_lambda(grl_lam)
        else:
            grl_lam = 0.0

        (tr_loss, tr_recon_u, tr_recon_m, tr_kl,
         tr_pheno, tr_domain, tr_dacc) = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer,
            device=device, loss_fn=vae_loss, masker=masker,
            beta=beta, alpha=alpha, gamma=gamma,
            use_grl=use_grl, delta=delta,
        )

        (v_loss, v_recon_u, v_recon_m, v_kl, v_pheno) = evaluate(
            model=model, dataloader=val_loader, device=device,
            loss_fn=vae_loss, beta=beta, alpha=alpha, gamma=gamma,
        )

        val_stop = v_recon_u + alpha * v_recon_m + beta * v_kl + gamma * v_pheno

        train_loss_list.append(tr_loss);  train_recon_u_list.append(tr_recon_u)
        train_recon_m_list.append(tr_recon_m); train_kl_list.append(tr_kl)
        train_pheno_list.append(tr_pheno)
        train_domain_list.append(tr_domain); train_dacc_list.append(tr_dacc)

        val_loss_list.append(v_loss);   val_recon_u_list.append(v_recon_u)
        val_recon_m_list.append(v_recon_m); val_kl_list.append(v_kl)
        val_pheno_list.append(v_pheno)

        grl_str = (f" | grl_lam={grl_lam:.3f}  d_ce={tr_domain:.4f}  d_acc={tr_dacc:.3f}"
                   if use_grl else "")
        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"train={tr_loss:.6f} val={v_loss:.6f} | "
            f"recon={tr_recon_u:.6f} pheno={tr_pheno:.6f} | "
            f"val_recon={v_recon_u:.6f} val_pheno={v_pheno:.6f}" + grl_str
        )

        if val_stop < (best_val_stop - min_delta):
            best_val_stop = val_stop
            best_epoch    = epoch + 1
            no_improve    = 0
            save_checkpoint(best_model_path, model, optimizer,
                            epoch + 1, val_stop, vae_config, input_length)
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epoch(s) "
                  f"(best={best_val_stop:.6f} at epoch {best_epoch})")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    stopped_epoch = len(train_loss_list)
    save_checkpoint(final_model_path, model, optimizer,
                    stopped_epoch, val_stop, vae_config, input_length)

    # Training history
    history = dict(
        train_losses=np.array(train_loss_list),
        val_losses=np.array(val_loss_list),
        train_recon_unmasked_losses=np.array(train_recon_u_list),
        val_recon_unmasked_losses=np.array(val_recon_u_list),
        train_kl_losses=np.array(train_kl_list),
        val_kl_losses=np.array(val_kl_list),
        train_phenotype_losses=np.array(train_pheno_list),
        val_phenotype_losses=np.array(val_pheno_list),
        train_recon_masked_losses=np.array(train_recon_m_list),
        val_recon_masked_losses=np.array(val_recon_m_list),
    )
    if use_grl:
        history["train_domain_losses"] = np.array(train_domain_list)
        history["train_domain_acc"]    = np.array(train_dacc_list)
    np.savez(out / "training_history.npz", **history)

    plot_loss_curves(
        train_losses=train_loss_list,          val_losses=val_loss_list,
        train_recon_unmasked_losses=train_recon_u_list,
        val_recon_unmasked_losses=val_recon_u_list,
        train_kl_losses=train_kl_list,         val_kl_losses=val_kl_list,
        train_pheno_losses=train_pheno_list,   val_pheno_losses=val_pheno_list,
        train_recon_masked_losses=train_recon_m_list if masking else None,
        val_recon_masked_losses=val_recon_m_list     if masking else None,
        train_domain_losses=train_domain_list if use_grl else None,
        train_domain_accs=train_dacc_list     if use_grl else None,
        output_dir=out,
    )

    # Reload best
    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Reloaded best model from epoch {best_ckpt['epoch']} "
          f"(stop_metric={best_ckpt['val_loss']:.6f})")

    use_masked = masking

    # Per-split plots
    for loader, name in [
        (disc_train_loader,   "discovery_train"),
        (target_train_loader, "target_train"),
        (val_loader,          "discovery_validation"),
    ]:
        run_eval_plots(model=model, loader=loader, device=device,
                       out_dir=plots_dir, split_name=name,
                       use_masked=use_masked, loss_fn=vae_loss,
                       alpha=alpha, beta=beta, gamma=gamma)

    # Shared-subspace PCA (mu_shared only — the domain-invariant part)
    disc_mu,   _ = extract_mu_shared(model, disc_train_loader,   device, use_masked)
    target_mu, _ = extract_mu_shared(model, target_train_loader, device, use_masked)
    val_mu,    _ = extract_mu_shared(model, val_loader,          device, use_masked)

    plot_latent_pca_shared_basis(
        reference_mu=disc_mu,
        ceu_mu=val_mu,
        yri_mu=target_mu,
        output_path=plots_dir / "latent_pca_shared_basis.png",
        reference_name="CEU discovery train",
        ceu_name="CEU validation",
        yri_name="YRI target train",
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-config",         type=Path, required=True)
    p.add_argument("--training-data",      type=Path, required=True)
    p.add_argument("--disc-train-data",    type=Path, required=True)
    p.add_argument("--target-train-data",  type=Path, required=True)
    p.add_argument("--validation-data",    type=Path, required=True)
    p.add_argument("--disc-train-pheno",   type=Path, required=True)
    p.add_argument("--target-train-pheno", type=Path, required=True)
    p.add_argument("--validation-pheno",   type=Path, required=True)
    p.add_argument("--outputs",            type=Path, required=True)
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