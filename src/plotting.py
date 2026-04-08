import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unpack_model_output(model_output):
    """
    Unpack the 9-tuple returned by ConvVAE.forward():
        out, mu_shared, logvar_shared, mu_private, logvar_private,
        z_shared, z_private, pheno_pred, domain_logits
    """
    (out,
     mu_shared, logvar_shared,
     mu_private, logvar_private,
     z_shared, z_private,
     pheno_pred, domain_logits) = model_output
    return (out, mu_shared, logvar_shared, mu_private, logvar_private,
            z_shared, z_private, pheno_pred, domain_logits)


def _get_input_and_metadata_from_batch(batch, use_masked_input=False):
    if len(batch) == 3:
        x, pheno, pop_label = batch
        x_input = x
        x_true  = x
        mask    = None
    elif len(batch) == 5:
        masked_x, x_true, pheno, mask, pop_label = batch
        x_input = masked_x if use_masked_input else x_true
    else:
        raise ValueError(f"Unexpected batch structure of length {len(batch)}")
    return x_input, x_true, pheno, pop_label, mask


# ------------------------------------------------------------------
# phenotype prediction
# ------------------------------------------------------------------
@torch.no_grad()
def extract_pheno_predictions(model, dataloader, device, use_masked_input=False):
    model.eval()
    all_true, all_pred, all_pop = [], [], []

    for batch in dataloader:
        x_input, _, pheno, pop_label, _ = _get_input_and_metadata_from_batch(
            batch, use_masked_input=use_masked_input
        )
        x_input = x_input.to(device)
        out     = model(x_input)
        _, _, _, _, _, _, _, pheno_pred, _ = _unpack_model_output(out)

        all_true.append(_to_numpy(pheno))
        all_pred.append(_to_numpy(pheno_pred))
        all_pop.append(_to_numpy(pop_label))

    return (
        np.concatenate(all_true, axis=0).squeeze(),
        np.concatenate(all_pred, axis=0).squeeze(),
        np.concatenate(all_pop,  axis=0).squeeze(),
    )


@torch.no_grad()
def plot_pheno_predictions(model, dataloader, device, output_path,
                           use_masked_input=False, title="Phenotype prediction"):
    model.eval()
    y_true, y_pred, pop = extract_pheno_predictions(model, dataloader, device, use_masked_input)

    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True phenotype")
    plt.ylabel("Predicted phenotype")
    plt.title(f"{title}\nRMSE = {rmse:.4f}, R² = {r2:.4f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved phenotype prediction plot to: {output_path}")
    print(f"Phenotype RMSE: {rmse:.6f}  R²: {r2:.6f}")
    return {"rmse": rmse, "r2": r2, "y_true": y_true, "y_pred": y_pred, "pop": pop}


@torch.no_grad()
def plot_pheno_predictions_by_population(model, dataloader, device, output_path,
                                         use_masked_input=False,
                                         title="Phenotype prediction by population"):
    model.eval()
    y_true, y_pred, pop = extract_pheno_predictions(model, dataloader, device, use_masked_input)

    plt.figure(figsize=(6, 6))
    for pop_value, label in [(0, "CEU/discovery"), (1, "YRI/target")]:
        idx = pop == pop_value
        if np.any(idx):
            plt.scatter(y_true[idx], y_pred[idx], alpha=0.6, s=20, label=label)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True phenotype")
    plt.ylabel("Predicted phenotype")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved phenotype-by-population plot to: {output_path}")


@torch.no_grad()
def plot_pheno_residuals(model, dataloader, device, output_path,
                         use_masked_input=False, title="Phenotype residuals"):
    model.eval()
    y_true, y_pred, _ = extract_pheno_predictions(model, dataloader, device, use_masked_input)
    plt.figure(figsize=(7, 5))
    plt.hist(y_pred - y_true, bins=40)
    plt.xlabel("Prediction error (pred - true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved phenotype residual histogram to: {output_path}")


# ------------------------------------------------------------------
# reconstruction / confusion matrix
# ------------------------------------------------------------------
@torch.no_grad()
def plot_reconstruction(model, dataloader, device, output_dir, use_masked_input=False):
    model.eval()
    _ensure_dir(output_dir)

    all_y_true, all_y_pred = [], []

    for batch in dataloader:
        x_input, x_true, _, _, _ = _get_input_and_metadata_from_batch(
            batch, use_masked_input=use_masked_input
        )
        x_input = x_input.to(device)
        x_true  = x_true.to(device)

        out_tuple = model(x_input)
        logits, _, _, _, _, _, _, _, _ = _unpack_model_output(out_tuple)

        all_y_true.append(x_true.long().squeeze(1).cpu().numpy())
        all_y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())

    y_true_flat = np.concatenate(all_y_true).reshape(-1)
    y_pred_flat = np.concatenate(all_y_pred).reshape(-1)

    classes = np.array([0, 1, 2])
    recalls = recall_score(y_true_flat, y_pred_flat, labels=classes, average=None, zero_division=0)
    bal_acc = recalls.mean()

    cm            = confusion_matrix(y_true_flat, y_pred_flat, labels=classes)
    cm_row_sums   = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm.astype(float), cm_row_sums,
                              out=np.zeros_like(cm, dtype=float), where=cm_row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(3), yticks=np.arange(3),
           xticklabels=classes, yticklabels=classes,
           xlabel="Predicted genotype", ylabel="True genotype",
           title=f"Normalized Confusion Matrix\nBalanced Acc = {bal_acc:.3f}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_normalized[i, j]:.3f}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close(fig)

    np.save(f"{output_dir}/confusion_matrix_raw.npy",        cm)
    np.save(f"{output_dir}/confusion_matrix_normalized.npy", cm_normalized)
    print(f"Global balanced accuracy: {bal_acc:.6f}")

    return {
        "balanced_accuracy":           float(bal_acc),
        "recalls":                     {int(c): float(r) for c, r in zip(classes, recalls)},
        "confusion_matrix_raw":        cm,
        "confusion_matrix_normalized": cm_normalized,
    }


# ------------------------------------------------------------------
# latent extraction / plots
# ------------------------------------------------------------------
@torch.no_grad()
def extract_mu(model, dataloader, device, use_masked_input=False):
    """
    Returns concatenated [mu_shared | mu_private] for PCA/visualisation,
    plus population labels.
    """
    model.eval()
    all_mu, all_labels = [], []

    for batch in dataloader:
        x_input, _, _, pop_label, _ = _get_input_and_metadata_from_batch(
            batch, use_masked_input=use_masked_input
        )
        x_input = x_input.to(device)
        out_tuple = model(x_input)
        _, mu_shared, _, mu_private, _, _, _, _, _ = _unpack_model_output(out_tuple)

        mu_full = torch.cat([mu_shared, mu_private], dim=1)
        all_mu.append(mu_full.cpu())
        all_labels.append(pop_label.cpu())

    return (
        torch.cat(all_mu,     dim=0).numpy(),
        torch.cat(all_labels, dim=0).numpy(),
    )


@torch.no_grad()
def extract_mu_shared(model, dataloader, device, use_masked_input=False):
    """
    Returns mu_shared only — used for shared-basis PCA.
    """
    model.eval()
    all_mu, all_labels = [], []

    for batch in dataloader:
        x_input, _, _, pop_label, _ = _get_input_and_metadata_from_batch(
            batch, use_masked_input=use_masked_input
        )
        x_input = x_input.to(device)
        out_tuple = model(x_input)
        _, mu_shared, _, _, _, _, _, _, _ = _unpack_model_output(out_tuple)

        all_mu.append(mu_shared.cpu())
        all_labels.append(pop_label.cpu())

    return (
        torch.cat(all_mu,     dim=0).numpy(),
        torch.cat(all_labels, dim=0).numpy(),
    )


@torch.no_grad()
def plot_latent_space(model, dataloader, device, output_dir,
                      save_path="latent_space.png", use_masked_input=False):
    model.eval()
    _ensure_dir(output_dir)

    all_mu, all_labels = extract_mu(model, dataloader, device, use_masked_input)
    mu_2d = PCA(n_components=2).fit_transform(all_mu)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(mu_2d[:, 0], mu_2d[:, 1],
                          c=all_labels, cmap="coolwarm", alpha=0.7, s=20)
    plt.xlabel("latent PC1")
    plt.ylabel("latent PC2")
    plt.title("Latent representation — PCA of [mu_shared | mu_private]")
    plt.colorbar(scatter, label="population")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{save_path}", dpi=300)
    plt.close()


def plot_latent_pca_shared_basis(reference_mu, ceu_mu, yri_mu, output_path,
                                 reference_name="CEU discovery train",
                                 ceu_name="CEU validation",
                                 yri_name="YRI target"):
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(reference_mu)
    ceu_scaled = scaler.transform(ceu_mu)
    yri_scaled = scaler.transform(yri_mu)

    pca = PCA(n_components=2)
    pca.fit(ref_scaled)
    ceu_pca = pca.transform(ceu_scaled)
    yri_pca = pca.transform(yri_scaled)
    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(7, 6))
    plt.scatter(ceu_pca[:, 0], ceu_pca[:, 1], alpha=0.7, s=20, label=ceu_name)
    plt.scatter(yri_pca[:, 0], yri_pca[:, 1], alpha=0.7, s=20, label=yri_name)
    plt.xlabel(f"PC1 ({explained[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({explained[1]*100:.2f}% var)")
    plt.title(f"Latent PCA (shared subspace)\nfit on {reference_name}, both projected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved shared-basis PCA plot to: {output_path}")


# ------------------------------------------------------------------
# loss curves
# ------------------------------------------------------------------
def plot_loss_curves(train_losses, val_losses,
                     train_recon_unmasked_losses, val_recon_unmasked_losses,
                     train_kl_losses, val_kl_losses,
                     train_pheno_losses, val_pheno_losses,
                     output_dir,
                     train_recon_masked_losses=None, val_recon_masked_losses=None,
                     train_domain_losses=None, train_domain_accs=None):
    _ensure_dir(output_dir)
    epochs = range(1, len(train_losses) + 1)

    def _save(fname, title, pairs, ylabel="loss"):
        plt.figure(figsize=(8, 5))
        for label, vals in pairs:
            plt.plot(epochs, vals, label=label)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname}", dpi=300)
        plt.close()

    _save("loss_total.png", "Total loss",
          [("train total (VAE+domain)", train_losses),
           ("val total (VAE only)",     val_losses)])

    _save("loss_recon_unmasked.png", "Reconstruction loss (unmasked)",
          [("train recon unmasked", train_recon_unmasked_losses),
           ("val recon unmasked",   val_recon_unmasked_losses)])

    if train_recon_masked_losses is not None and val_recon_masked_losses is not None:
        _save("loss_recon_masked.png", "Reconstruction loss (masked)",
              [("train recon masked", train_recon_masked_losses),
               ("val recon masked",   val_recon_masked_losses)])

    _save("loss_kl.png", "KL loss (shared + private)",
          [("train kl", train_kl_losses),
           ("val kl",   val_kl_losses)])

    _save("loss_pheno.png", "Phenotype prediction loss",
          [("train phenotype", train_pheno_losses),
           ("val phenotype",   val_pheno_losses)])

    if train_domain_losses is not None:
        _save("loss_domain.png", "Domain classification loss (training only)",
              [("train domain CE", train_domain_losses)],
              ylabel="domain cross-entropy")

    if train_domain_accs is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_domain_accs, label="train domain accuracy")
        plt.axhline(0.5, linestyle="--", color="gray", label="chance (balanced)")
        plt.xlabel("epoch")
        plt.ylabel("domain classifier accuracy")
        plt.title("Domain classifier accuracy (training)\n"
                  "Closer to 0.5 = more domain-agnostic shared subspace")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/domain_classifier_accuracy.png", dpi=300)
        plt.close()


# ------------------------------------------------------------------
# masking diagnostic
# ------------------------------------------------------------------
def plot_example_input_heatmap(original_x, masked_x, mask, output_path,
                                sample_indices=(0, 1, 2, 3, 4),
                                snp_start=0, snp_count=1000):
    snp_end        = min(snp_start + snp_count, original_x.shape[-1])
    sample_indices = list(sample_indices)

    def _sl(x):
        if x.ndim == 3:
            return x[sample_indices, 0, snp_start:snp_end].detach().cpu().numpy()
        return x[sample_indices, snp_start:snp_end].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for ax, data, title in zip(axes,
                                [_sl(original_x), _sl(masked_x), _sl(mask)],
                                ["Original input", "Masked input", "Mask (1 = masked)"]):
        im = ax.imshow(data, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("Individual")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    axes[-1].set_xlabel("SNP index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved masked-input heatmap plot to: {output_path}")