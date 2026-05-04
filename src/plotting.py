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


def _get_input_and_metadata_from_batch(batch, use_masked_input=False):
    """
    Supported batch formats:

    train:
        (x, pheno, pop_label)

    val/target:
        (masked_x, x_true, pheno, mask, pop_label)

    Returns
    -------
    x_input : torch.Tensor
    x_true  : torch.Tensor
    pheno   : torch.Tensor
    pop_label : torch.Tensor
    mask    : torch.Tensor or None
    """
    if len(batch) == 3:
        x, pheno, pop_label = batch
        x_input = x
        x_true = x
        mask = None

    elif len(batch) == 5:
        masked_x, x_true, pheno, mask, pop_label = batch
        x_input = masked_x if use_masked_input else x_true

    else:
        raise ValueError(f"Unexpected batch structure of length {len(batch)}")

    return x_input, x_true, pheno, pop_label, mask


# ------------------------------------------------------------------
# phenotype prediction helpers / plots
# ------------------------------------------------------------------
@torch.no_grad()
def extract_pheno_predictions(model, dataloader, device, use_masked_input=False):
    model.eval()

    all_true = []
    all_pred = []
    all_pop = []

    for batch in dataloader:
        x_input, _, pheno, pop_label, _ = _get_input_and_metadata_from_batch(
            batch, use_masked_input=use_masked_input
        )

        x_input = x_input.to(device)

        logits, mu, logvar, z, pheno_pred, _domain_logits = model(x_input)

        all_true.append(_to_numpy(pheno))
        all_pred.append(_to_numpy(pheno_pred))
        all_pop.append(_to_numpy(pop_label))

    y_true = np.concatenate(all_true, axis=0).squeeze()
    y_pred = np.concatenate(all_pred, axis=0).squeeze()
    pop = np.concatenate(all_pop, axis=0).squeeze()

    return y_true, y_pred, pop


@torch.no_grad()
def plot_pheno_predictions(
    model,
    dataloader,
    device,
    output_path,
    use_masked_input=False,
    title="Phenotype prediction",
):
    model.eval()

    y_true, y_pred, pop = extract_pheno_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        use_masked_input=use_masked_input,
    )

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("True phenotype")
    plt.ylabel("Predicted phenotype")
    plt.title(f"{title}\nRMSE = {rmse:.4f}, R² = {r2:.4f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved phenotype prediction plot to: {output_path}")
    print(f"Phenotype RMSE: {rmse:.6f}")
    print(f"Phenotype R^2:  {r2:.6f}")

    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "y_true": y_true,
        "y_pred": y_pred,
        "pop": pop,
    }


@torch.no_grad()
def plot_pheno_predictions_by_population(
    model,
    dataloader,
    device,
    output_path,
    use_masked_input=False,
    title="Phenotype prediction by population",
):
    model.eval()

    y_true, y_pred, pop = extract_pheno_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        use_masked_input=use_masked_input,
    )

    plt.figure(figsize=(6, 6))

    for pop_value, label in [(0, "CEU/discovery"), (1, "YRI/target")]:
        idx = pop == pop_value
        if np.any(idx):
            plt.scatter(y_true[idx], y_pred[idx], alpha=0.6, s=20, label=label)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("True phenotype")
    plt.ylabel("Predicted phenotype")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved phenotype-by-population plot to: {output_path}")


@torch.no_grad()
def plot_pheno_residuals(
    model,
    dataloader,
    device,
    output_path,
    use_masked_input=False,
    title="Phenotype residuals",
):
    model.eval()

    y_true, y_pred, pop = extract_pheno_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        use_masked_input=use_masked_input,
    )

    residuals = y_pred - y_true

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=40)
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
def plot_reconstruction(
    model,
    dataloader,
    device,
    output_dir,
    use_masked_input=False,
):
    model.eval()
    _ensure_dir(output_dir)

    all_y_true = []
    all_y_pred = []

    for batch in dataloader:
        x_input, x_true, _, _, _ = _get_input_and_metadata_from_batch(
            batch, use_masked_input=use_masked_input
        )

        x_input = x_input.to(device)
        x_true = x_true.to(device)

        logits, mu, logvar, z, pheno_pred, _domain_logits = model(x_input)

        y_true = x_true.long().squeeze(1).cpu().numpy()
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    y_true_flat = y_true_all.reshape(-1)
    y_pred_flat = y_pred_all.reshape(-1)

    classes = np.array([0, 1, 2])

    recalls = recall_score(
        y_true_flat,
        y_pred_flat,
        labels=classes,
        average=None,
        zero_division=0,
    )
    bal_acc = recalls.mean()

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=classes)

    cm_row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        cm_row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=cm_row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted genotype",
        ylabel="True genotype",
        title=f"Normalized Confusion Matrix\nBalanced Acc = {bal_acc:.3f}",
    )

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_normalized[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close(fig)

    np.save(f"{output_dir}/confusion_matrix_raw.npy", cm)
    np.save(f"{output_dir}/confusion_matrix_normalized.npy", cm_normalized)

    print(f"Global balanced accuracy: {bal_acc:.6f}")
    for cls, rec in zip(classes, recalls):
        print(f"Recall for class {cls}: {rec:.6f}")

    return {
        "balanced_accuracy": float(bal_acc),
        "recalls": {int(cls): float(rec) for cls, rec in zip(classes, recalls)},
        "confusion_matrix_raw": cm,
        "confusion_matrix_normalized": cm_normalized,
    }


# ------------------------------------------------------------------
# latent extraction / plots
# ------------------------------------------------------------------
@torch.no_grad()
def extract_mu(model, dataloader, device, use_masked_input=False):
    model.eval()

    all_mu = []
    all_labels = []

    for batch in dataloader:
        x_input, _, _, pop_label, _ = _get_input_and_metadata_from_batch(
            batch, use_masked_input=use_masked_input
        )

        x_input = x_input.to(device)

        _, mu, _, _, _, _domain_logits = model(x_input)

        all_mu.append(mu.cpu())
        all_labels.append(pop_label.cpu())

    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_mu, all_labels


def fit_latent_pca(reference_vecs: np.ndarray):
    """
    Fit StandardScaler + PCA on reference vectors.
    Returns (scaler, pca) — pass these into plot_latent_space
    and plot_latent_pca_shared_basis to keep all plots in the
    same coordinate system.
    """
    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(reference_vecs)
    pca = PCA(n_components=2)
    pca.fit(reference_scaled)
    return scaler, pca


def plot_latent_space(
    latent_vectors,
    labels,
    output_dir,
    save_path="latent_space.png",
    title="Latent representation (PCA)",
    scaler=None,
    pca=None,
):
    _ensure_dir(output_dir)

    if scaler is not None and pca is not None:
        coords = pca.transform(scaler.transform(latent_vectors))
        explained = pca.explained_variance_ratio_
        xlabel = f"PC1 ({explained[0] * 100:.2f}% var)"
        ylabel = f"PC2 ({explained[1] * 100:.2f}% var)"
    else:
        coords = PCA(n_components=2).fit_transform(latent_vectors)
        xlabel = "latent PC1"
        ylabel = "latent PC2"

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.7,
        s=20,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(scatter, label="population")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{save_path}", dpi=300)
    plt.close()


def plot_latent_pca_shared_basis(
    reference_vecs,
    ceu_vecs,
    yri_vecs,
    output_path,
    reference_name="CEU discovery train",
    ceu_name="CEU validation",
    yri_name="YRI target",
    ceu_color_vec=None,
    yri_color_vec=None,
    color_label="",
    scaler=None,
    pca=None,
):
    if scaler is None or pca is None:
        scaler, pca = fit_latent_pca(reference_vecs)

    ceu_pca = pca.transform(scaler.transform(ceu_vecs))
    yri_pca = pca.transform(scaler.transform(yri_vecs))
    explained = pca.explained_variance_ratio_

    use_color = ceu_color_vec is not None and yri_color_vec is not None
    vmin = min(ceu_color_vec.min(), yri_color_vec.min()) if use_color else None
    vmax = max(ceu_color_vec.max(), yri_color_vec.max()) if use_color else None

    fig, ax = plt.subplots(figsize=(7, 6))

    sc1 = ax.scatter(
        ceu_pca[:, 0],
        ceu_pca[:, 1],
        c=ceu_color_vec if use_color else None,
        alpha=0.7,
        s=20,
        label=ceu_name,
        cmap="viridis" if use_color else None,
        vmin=vmin,
        vmax=vmax,
        marker="o",
    )
    ax.scatter(
        yri_pca[:, 0],
        yri_pca[:, 1],
        c=yri_color_vec if use_color else None,
        alpha=0.7,
        s=20,
        label=yri_name,
        cmap="viridis" if use_color else None,
        vmin=vmin,
        vmax=vmax,
        marker="^",
    )

    if use_color:
        fig.colorbar(sc1, ax=ax, label=color_label)

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.2f}% var)")
    ax.set_title(f"Latent PCA\nfit on {reference_name}, both projected")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved shared-basis PCA plot to: {output_path}")


# ------------------------------------------------------------------
# loss curves
# ------------------------------------------------------------------
def plot_loss_curves(
    train_losses,
    val_losses,
    train_recon_unmasked_losses,
    val_recon_unmasked_losses,
    train_kl_losses,
    val_kl_losses,
    train_pheno_losses,
    val_pheno_losses,
    output_dir,
    train_recon_masked_losses=None,
    val_recon_masked_losses=None,
    train_domain_losses=None,
    train_domain_accs=None,
    train_z_shared_vars=None,  # new: per-epoch mean variance of z_shared
    train_z_pop_vars=None,  # new: per-epoch mean variance of z_pop
):
    _ensure_dir(output_dir)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train total (incl. domain loss if GRL)")
    plt.plot(epochs, val_losses, label="val VAE loss (recon + KL + pheno)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(
        "Total loss\n(train includes domain loss; val is VAE-only for early stopping)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_total.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_recon_unmasked_losses, label="train recon unmasked")
    plt.plot(epochs, val_recon_unmasked_losses, label="val recon unmasked")
    plt.xlabel("epoch")
    plt.ylabel("recon unmasked loss")
    plt.title("Reconstruction loss (unmasked positions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_recon_unmasked.png", dpi=300)
    plt.close()

    if train_recon_masked_losses is not None and val_recon_masked_losses is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_recon_masked_losses, label="train recon masked")
        plt.plot(epochs, val_recon_masked_losses, label="val recon masked")
        plt.xlabel("epoch")
        plt.ylabel("recon masked loss")
        plt.title("Reconstruction loss (masked positions)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_recon_masked.png", dpi=300)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_kl_losses, label="train kl")
    plt.plot(epochs, val_kl_losses, label="val kl")
    plt.xlabel("epoch")
    plt.ylabel("KL loss")
    plt.title("KL loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_kl.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_pheno_losses, label="train phenotype")
    plt.plot(epochs, val_pheno_losses, label="val phenotype")
    plt.xlabel("epoch")
    plt.ylabel("phenotype loss")
    plt.title("Phenotype prediction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_pheno.png", dpi=300)
    plt.close()

    if train_domain_losses is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_domain_losses, label="train domain CE")
        plt.xlabel("epoch")
        plt.ylabel("domain cross-entropy")
        plt.title("Domain classification loss (training only)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_domain.png", dpi=300)
        plt.close()

    if train_domain_accs is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_domain_accs, label="train domain accuracy")
        plt.axhline(0.5, linestyle="--", color="gray", label="chance (balanced)")
        plt.xlabel("epoch")
        plt.ylabel("domain classifier accuracy")
        plt.title(
            "Domain classifier accuracy (training)\n"
            "Closer to 0.5 = more domain-agnostic latent space"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/domain_classifier_accuracy.png", dpi=300)
        plt.close()

    # latent subspace variance — always plot if provided
    if train_z_shared_vars is not None and train_z_pop_vars is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_z_shared_vars, label="z_shared variance")
        plt.plot(epochs, train_z_pop_vars, label="z_pop variance")
        plt.xlabel("epoch")
        plt.ylabel("mean per-dim variance")
        plt.title(
            "Latent subspace variance\n"
            "z_pop collapsing toward 0 = split not working as intended"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latent_subspace_variance.png", dpi=300)
        plt.close()


# ------------------------------------------------------------------
# lambda vs loss (GRL schedule diagnostic)
# ------------------------------------------------------------------
def plot_lambda_vs_loss(
    lambda_values,
    loss_dict,
    output_dir,
):
    """Plot each loss in loss_dict against GRL lambda value (one file per loss).

    loss_dict: {label: (train_values, val_values_or_None, filename_stem)}
    """
    _ensure_dir(output_dir)

    for label, (train_values, val_values, fname) in loss_dict.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lambda_values, train_values, linewidth=1.5, label="train")
        if val_values is not None:
            ax.plot(lambda_values, val_values, linewidth=1.5, label="val", linestyle="--")
            ax.legend()
        ax.set_xlabel("GRL λ")
        ax.set_ylabel(label)
        ax.set_title(
            f"{label} vs GRL λ\n"
            "(λ ramps from 0 → λ_max via Ganin et al. schedule)"
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname}.png", dpi=300)
        plt.close()


# ------------------------------------------------------------------
# masking diagnostic
# ------------------------------------------------------------------
def plot_example_input_heatmap(
    original_x,
    masked_x,
    mask,
    output_path,
    sample_indices=(0, 1, 2, 3, 4),
    snp_start=0,
    snp_count=1000,
):
    snp_end = min(snp_start + snp_count, original_x.shape[-1])
    sample_indices = list(sample_indices)

    def _slice_tensor(x, sample_indices, snp_start, snp_end):
        if x.ndim == 3:
            return x[sample_indices, 0, snp_start:snp_end].detach().cpu().numpy()
        elif x.ndim == 2:
            return x[sample_indices, snp_start:snp_end].detach().cpu().numpy()
        else:
            raise ValueError(
                f"Expected tensor with 2 or 3 dims, got shape {tuple(x.shape)}"
            )

    orig = _slice_tensor(original_x, sample_indices, snp_start, snp_end)
    masked = _slice_tensor(masked_x, sample_indices, snp_start, snp_end)
    mask_arr = _slice_tensor(mask, sample_indices, snp_start, snp_end)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    im0 = axes[0].imshow(orig, aspect="auto", interpolation="nearest")
    axes[0].set_title("Original input")
    axes[0].set_ylabel("Individual")
    plt.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02)

    im1 = axes[1].imshow(masked, aspect="auto", interpolation="nearest")
    axes[1].set_title("Masked input")
    axes[1].set_ylabel("Individual")
    plt.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

    im2 = axes[2].imshow(mask_arr, aspect="auto", interpolation="nearest")
    axes[2].set_title("Mask (1 = masked)")
    axes[2].set_ylabel("Individual")
    axes[2].set_xlabel("SNP index")
    plt.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved masked-input heatmap plot to: {output_path}")


# ------------------------------------------------------------------
# latent activation movie (GRL degeneracy diagnostic)
# ------------------------------------------------------------------
def plot_mu_vs_domain_loss(snapshots, output_dir):
    """Static PNG: mean |μ| vs domain loss across all epochs."""
    if not snapshots:
        return

    _ensure_dir(output_dir)

    epochs      = [s["epoch"]      for s in snapshots]
    domain_loss = [s.get("domain_loss", float("nan")) for s in snapshots]
    mean_abs_mu = [float(np.mean(np.abs(
                       np.concatenate([s["mu_ceu"]] + ([s["mu_yri"]] if s.get("mu_yri") is not None else []))
                   ))) for s in snapshots]

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(domain_loss, mean_abs_mu, c=epochs, cmap="viridis", s=30, zorder=2)
    ax.plot(domain_loss, mean_abs_mu, color="gray", linewidth=0.8, zorder=1)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("epoch", fontsize=9)
    ax.set_xlabel("domain loss", fontsize=10)
    ax.set_ylabel("mean |μ|", fontsize=10)
    ax.set_title("Mean |μ| vs domain loss (trajectory)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "mu_vs_domain_loss.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved mean |μ| vs domain loss plot → {out_path}")


def plot_latent_activation_movie(
    snapshots,
    output_dir,
    fps=10,
    max_per_pop=200,
    save_frames=False,
):
    """Save a GIF with two stacked panels evolving epoch by epoch.

    Panel 1 (top):    domain loss vs GRL lambda — full curve plotted once,
                      red dot tracks the current epoch.
    Panel 2 (bottom): μ activation heatmap — CEU (top) / YRI (bottom).

    Also calls plot_mu_vs_domain_loss to save a companion static PNG.

    snapshots: list of dicts with keys
        epoch (int), grl_lam (float), domain_loss (float),
        mu_ceu (np.ndarray), mu_yri (np.ndarray or None)
    """
    import matplotlib.animation as animation

    if not snapshots:
        return

    _ensure_dir(output_dir)

    # companion static plot
    plot_mu_vs_domain_loss(snapshots, output_dir)

    # ── per-frame data ──────────────────────────────────────────────────────
    frames_data = []
    for snap in snapshots:
        mu_ceu = snap["mu_ceu"][:max_per_pop]
        n_ceu  = len(mu_ceu)

        if snap.get("mu_yri") is not None:
            mu_yri = snap["mu_yri"][:max_per_pop]
            mat    = np.concatenate([mu_ceu, mu_yri], axis=0)
            n_yri  = len(mu_yri)
        else:
            mat, n_yri = mu_ceu, 0

        frames_data.append(
            dict(epoch=snap["epoch"],
                 grl_lam=snap["grl_lam"],
                 domain_loss=snap.get("domain_loss", float("nan")),
                 mat=mat, n_ceu=n_ceu, n_yri=n_yri)
        )

    all_lam   = np.array([f["grl_lam"]    for f in frames_data])
    all_dloss = np.array([f["domain_loss"] for f in frames_data])

    # global symmetric colour scale for heatmap
    all_vals = np.concatenate([f["mat"].ravel() for f in frames_data])
    vmax = float(np.percentile(np.abs(all_vals), 99))
    vmin = -vmax

    n_ceu   = frames_data[0]["n_ceu"]
    n_yri   = frames_data[0]["n_yri"]
    n_total = n_ceu + n_yri

    all_epochs = np.array([f["epoch"] for f in frames_data])

    # ── figure: 2 rows ──────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [2, 3]}
    )
    fig.subplots_adjust(top=0.93, hspace=0.35)

    # panel 1 — domain loss + lambda vs epoch (dual y-axis)
    ax_top.plot(all_epochs, all_dloss, color="steelblue", linewidth=1.5,
                label="domain loss", zorder=1)
    dot_top, = ax_top.plot(all_epochs[0], all_dloss[0], "o", color="red",
                           markersize=8, zorder=3)
    ax_top.set_xlabel("Epoch", fontsize=9)
    ax_top.set_ylabel("domain loss", color="steelblue", fontsize=9)
    ax_top.tick_params(axis="y", labelcolor="steelblue", labelsize=8)
    ax_top.tick_params(axis="x", labelsize=8)

    ax_lam = ax_top.twinx()
    ax_lam.plot(all_epochs, all_lam, color="darkorange", linewidth=1.2,
                linestyle="--", label="GRL λ", zorder=1)
    dot_lam, = ax_lam.plot(all_epochs[0], all_lam[0], "o", color="red",
                           markersize=8, zorder=3)
    ax_lam.set_ylabel("GRL λ", color="darkorange", fontsize=9)
    ax_lam.tick_params(axis="y", labelcolor="darkorange", labelsize=8)

    lines = [
        plt.Line2D([0], [0], color="steelblue", linewidth=1.5),
        plt.Line2D([0], [0], color="darkorange", linewidth=1.2, linestyle="--"),
    ]
    ax_top.legend(lines, ["domain loss", "GRL λ"], fontsize=8, loc="upper left")
    ax_top.set_title("Domain loss & GRL λ vs epoch", fontsize=10)

    # panel 2 — μ heatmap
    im = ax_bot.imshow(frames_data[0]["mat"], aspect="auto", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_bot.axhline(n_ceu - 0.5, color="black", linestyle="--", linewidth=1.0)
    cbar = fig.colorbar(im, ax=ax_bot, fraction=0.02, pad=0.01)
    cbar.set_label("activation", fontsize=9)
    ax_bot.set_xlabel("Latent dimension (μ)", fontsize=9)
    ax_bot.set_ylabel("Individual", fontsize=9)
    ax_bot.tick_params(labelsize=8)

    if n_total > 0:
        ax_bot.text(-0.03, 1.0 - n_ceu / (2.0 * n_total),
                    "CEU", transform=ax_bot.transAxes, va="center", ha="right",
                    color="#1f77b4", fontsize=9, fontweight="bold")
    if n_yri > 0:
        ax_bot.text(-0.03, 1.0 - (n_ceu + n_yri * 0.5) / n_total,
                    "YRI", transform=ax_bot.transAxes, va="center", ha="right",
                    color="#ff7f0e", fontsize=9, fontweight="bold")

    fd0 = frames_data[0]
    sup = fig.suptitle(f"Epoch {fd0['epoch']}  λ={fd0['grl_lam']:.3f}", fontsize=11)

    def update(i):
        fd = frames_data[i]
        im.set_data(fd["mat"])
        dot_top.set_data([fd["epoch"]], [fd["domain_loss"]])
        dot_lam.set_data([fd["epoch"]], [fd["grl_lam"]])
        sup.set_text(f"Epoch {fd['epoch']}  λ={fd['grl_lam']:.3f}")
        return im, dot_top, dot_lam, sup

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_data), interval=1000 / fps, blit=False
    )

    gif_path = os.path.join(output_dir, "latent_activation_movie.gif")
    print(f"Saving latent activation movie ({len(frames_data)} frames @ {fps} fps) → {gif_path}")
    ani.save(gif_path, writer=animation.PillowWriter(fps=fps))
    plt.close()
    print(f"  Done — {os.path.getsize(gif_path) / 1e6:.1f} MB")

    if save_frames:
        frames_dir = os.path.join(output_dir, "latent_movie_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for fd in frames_data:
            fig2, (ax2_top, ax2_bot) = plt.subplots(
                2, 1, figsize=(12, 7),
                gridspec_kw={"height_ratios": [2, 3]}
            )
            fig2.subplots_adjust(top=0.93, hspace=0.35)

            ax2_top.plot(all_epochs, all_dloss, color="steelblue", linewidth=1.5)
            ax2_top.plot(fd["epoch"], fd["domain_loss"], "o", color="red", markersize=8)
            ax2_top.set_xlabel("Epoch", fontsize=9)
            ax2_top.set_ylabel("domain loss", color="steelblue", fontsize=9)
            ax2_top.tick_params(axis="y", labelcolor="steelblue", labelsize=8)
            ax2_lam = ax2_top.twinx()
            ax2_lam.plot(all_epochs, all_lam, color="darkorange", linewidth=1.2, linestyle="--")
            ax2_lam.plot(fd["epoch"], fd["grl_lam"], "o", color="red", markersize=8)
            ax2_lam.set_ylabel("GRL λ", color="darkorange", fontsize=9)
            ax2_lam.tick_params(axis="y", labelcolor="darkorange", labelsize=8)
            ax2_top.set_title("Domain loss & GRL λ vs epoch", fontsize=10)

            im2 = ax2_bot.imshow(fd["mat"], aspect="auto", cmap="RdBu_r",
                                 vmin=vmin, vmax=vmax, interpolation="nearest")
            ax2_bot.axhline(n_ceu - 0.5, color="black", linestyle="--", linewidth=1.0)
            fig2.colorbar(im2, ax=ax2_bot, fraction=0.02, pad=0.01).set_label("activation", fontsize=9)
            ax2_bot.set_xlabel("Latent dimension (μ)", fontsize=9)
            ax2_bot.set_ylabel("Individual", fontsize=9)
            fig2.suptitle(f"Epoch {fd['epoch']}  λ={fd['grl_lam']:.3f}", fontsize=11)
            plt.savefig(os.path.join(frames_dir, f"epoch_{fd['epoch']:04d}.png"), dpi=100)
            plt.close()
        print(f"  Saved {len(frames_data)} PNG frames → {frames_dir}")
