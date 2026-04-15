# src/utils.py
from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.plotting import (
    plot_latent_space,
    plot_reconstruction,
    plot_pheno_predictions,
    plot_pheno_predictions_by_population,
    plot_pheno_residuals,
)
from src.train import evaluate


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
            _, mu, _, _, _, _ = model(x)
            mu_all.append(mu.cpu().numpy())
            labels_all.append(pop_label.cpu().numpy())
    return np.concatenate(mu_all, axis=0), np.concatenate(labels_all, axis=0)


def extract_std(model, dataloader, device, use_masked_input=False):
    model.eval()
    logvar_all, labels_all = [], []
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
            _, _, logvar, _, _, _ = model(x)
            logvar_all.append(logvar.cpu().numpy())
            labels_all.append(pop_label.cpu().numpy())
    return np.concatenate(logvar_all, axis=0), np.concatenate(labels_all, axis=0)


def extract_latent(model, dataloader, device, use_masked_input=False):
    """Extract mu+std concatenated latent vector and labels."""
    mu,  labels = extract_mu(model,  dataloader, device, use_masked_input=use_masked_input)
    std, _      = extract_std(model, dataloader, device, use_masked_input=use_masked_input)
    latent = np.concatenate([mu, np.exp(0.5 * std)], axis=1)
    return latent, labels


# =============================================================================
# Helpers
# =============================================================================

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

    latent_vecs, labels = extract_latent(model, loader, device, use_masked_input=use_masked)
    plot_latent_space(
        latent_vectors=latent_vecs,
        labels=labels,
        output_dir=split_dir,
        save_path="latent_space.png",
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

def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> "ConvVAE":
    from src.model import ConvVAE

    checkpoint = torch.load(checkpoint_path, map_location=device)

    vae_config   = checkpoint["vae_config"]
    input_length = int(checkpoint["input_length"])

    pheno_hidden_dim = vae_config.get("phenotype", {}).get("pheno_hidden_dim", None)

    da_cfg         = vae_config.get("domain_adaptation", {})
    use_grl        = bool(da_cfg.get("use_grl", False))
    raw            = da_cfg.get("grl_hidden_dim", None)
    grl_hidden_dim = int(raw) if raw is not None else None

    model = ConvVAE(
        input_length=input_length,
        in_channels=1,
        hidden_channels=vae_config["model"]["hidden_channels"],
        kernel_size=int(vae_config["model"]["kernel_size"]),
        stride=int(vae_config["model"]["stride"]),
        padding=int(vae_config["model"]["padding"]),
        latent_dim=int(vae_config["model"]["latent_dim"]),
        use_batchnorm=bool(vae_config["model"].get("use_batchnorm", False)),
        activation=vae_config["model"].get("activation", "elu"),
        pheno_dim=1,
        pheno_hidden_dim=pheno_hidden_dim,
        use_grl=use_grl,
        grl_hidden_dim=grl_hidden_dim,
        num_domains=2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def reconstruct_argmax_genotypes(
    model: torch.nn.Module,
    G: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(G, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    recon_batches = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            logits, _, _, _, _, _ = model(x)
            pred = torch.argmax(logits, dim=1)
            recon_batches.append(pred.cpu().numpy())

    return np.concatenate(recon_batches, axis=0)