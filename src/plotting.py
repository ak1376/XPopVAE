import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score


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
        _, _, _, _, pheno_pred = model(x_input)

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

    y_true, y_pred, _ = extract_pheno_predictions(
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

        logits, mu, logvar, z, pheno_pred = model(x_input)

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
        _, mu, _, _, _ = model(x_input)

        all_mu.append(mu.cpu())
        all_labels.append(pop_label.cpu())

    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_mu, all_labels


@torch.no_grad()
def plot_latent_space(
    model,
    dataloader,
    device,
    output_dir,
    save_path="latent_space.png",
    use_masked_input=False,
):
    model.eval()
    _ensure_dir(output_dir)

    all_mu, all_labels = extract_mu(
        model=model,
        dataloader=dataloader,
        device=device,
        use_masked_input=use_masked_input,
    )

    mu_2d = PCA(n_components=2).fit_transform(all_mu)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        mu_2d[:, 0], mu_2d[:, 1], c=all_labels, cmap="coolwarm", alpha=0.7, s=20
    )
    plt.xlabel("latent PC1")
    plt.ylabel("latent PC2")
    plt.title("Latent representation (PCA of mu)")
    plt.colorbar(scatter, label="population")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{save_path}", dpi=300)
    plt.close()


def plot_latent_pca_shared_basis(
    reference_mu,
    ceu_mu,
    yri_mu,
    output_path,
    reference_name="CEU discovery train",
    ceu_name="CEU validation",
    yri_name="YRI target",
):
    scaler = StandardScaler()
    reference_mu_scaled = scaler.fit_transform(reference_mu)
    ceu_mu_scaled = scaler.transform(ceu_mu)
    yri_mu_scaled = scaler.transform(yri_mu)

    pca = PCA(n_components=2)
    pca.fit(reference_mu_scaled)

    ceu_pca = pca.transform(ceu_mu_scaled)
    yri_pca = pca.transform(yri_mu_scaled)
    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(7, 6))
    plt.scatter(ceu_pca[:, 0], ceu_pca[:, 1], alpha=0.7, s=20, label=ceu_name)
    plt.scatter(yri_pca[:, 0], yri_pca[:, 1], alpha=0.7, s=20, label=yri_name)
    plt.xlabel(f"PC1 ({explained[0] * 100:.2f}% var)")
    plt.ylabel(f"PC2 ({explained[1] * 100:.2f}% var)")
    plt.title(f"Latent PCA\nfit on {reference_name}, both projected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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
):
    _ensure_dir(output_dir)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train total")
    plt.plot(epochs, val_losses, label="val total")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total loss")
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
        plt.plot(epochs, train_domain_losses, label="train domain (GRL)")
        plt.xlabel("epoch")
        plt.ylabel("domain loss")
        plt.title("Domain adversarial loss (train only)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_domain.png", dpi=300)
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
# domain separation plots
# ------------------------------------------------------------------
@torch.no_grad()
def extract_domain_probs(model, dataloader, device, use_masked_input=False):
    """
    Run all individuals through the encoder + domain_head and return
    predicted p(YRI) (softmax probability of class 1) alongside true
    population labels.

    Returns
    -------
    p_yri      : np.ndarray, shape (N,)  — predicted prob of being YRI
    true_labels: np.ndarray, shape (N,)  — 0=CEU, 1=YRI
    """
    model.eval()

    all_probs = []
    all_labels = []

    for batch in dataloader:
        x_input, _, _, pop_label, _ = _get_input_and_metadata_from_batch(
            batch, use_masked_input=use_masked_input
        )
        x_input = x_input.to(device)

        _, mu, _, _, _, domain_logits = model(x_input)
        probs = torch.softmax(domain_logits, dim=1)[:, 1]  # p(YRI)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(_to_numpy(pop_label))

    p_yri = np.concatenate(all_probs, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)

    return p_yri, true_labels


def plot_domain_probability_histogram(
    p_yri,
    true_labels,
    output_path,
    title="Domain classifier: predicted p(YRI)",
):
    """
    Overlapping histogram of predicted p(YRI) for CEU and YRI individuals.

    If the GRL is working well the two distributions should overlap heavily.
    If the model retains population structure they will separate toward 0 and 1.
    """
    ceu_probs = p_yri[true_labels == 0]
    yri_probs = p_yri[true_labels == 1]

    # domain balanced accuracy from hard predictions
    hard_preds = (p_yri >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(true_labels.astype(int), hard_preds)

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 25)

    ax.hist(ceu_probs, bins=bins, alpha=0.6, label="CEU (discovery)", density=True)
    ax.hist(yri_probs, bins=bins, alpha=0.6, label="YRI (target)", density=True)

    ax.axvline(
        0.5, color="black", linestyle="--", linewidth=0.8, label="decision boundary"
    )
    ax.set_xlabel("Predicted p(YRI)")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nDomain balanced acc = {bal_acc:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved domain probability histogram to: {output_path}")
    print(f"Domain balanced accuracy: {bal_acc:.4f}")

    return {"balanced_accuracy": float(bal_acc)}


def plot_domain_decile_enrichment(
    p_yri,
    true_labels,
    output_path,
    title="Domain classifier: YRI enrichment by decile",
):
    """
    Sort all individuals by predicted p(YRI), bin into deciles, and plot
    the fraction of true YRI individuals in each decile.

    If the domain classifier has no discriminative power (GRL working perfectly)
    every decile should contain ~50% YRI.
    If the classifier separates populations well, low deciles will be enriched
    for CEU and high deciles for YRI.
    """
    n = len(p_yri)
    order = np.argsort(p_yri)
    sorted_labels = true_labels[order]

    n_deciles = 10
    decile_size = n // n_deciles
    # handle remainder by extending the last bin
    bin_edges = [i * decile_size for i in range(n_deciles)] + [n]

    decile_frac_yri = []
    decile_mean_prob = []

    for i in range(n_deciles):
        sl = slice(bin_edges[i], bin_edges[i + 1])
        decile_frac_yri.append(sorted_labels[sl].mean())
        decile_mean_prob.append(p_yri[order[sl]].mean())

    decile_labels = [f"D{i+1}" for i in range(n_deciles)]
    baseline = true_labels.mean()  # overall fraction of YRI

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: fraction YRI per decile
    ax = axes[0]
    bars = ax.bar(decile_labels, decile_frac_yri, color="steelblue", alpha=0.8)
    ax.axhline(
        baseline,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"overall YRI fraction ({baseline:.2f})",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Decile (sorted by predicted p(YRI))")
    ax.set_ylabel("Fraction true YRI")
    ax.set_title("YRI enrichment per decile")
    ax.legend(fontsize=8)

    # right: mean predicted p(YRI) per decile
    ax2 = axes[1]
    ax2.bar(decile_labels, decile_mean_prob, color="darkorange", alpha=0.8)
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="chance (0.5)")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Decile (sorted by predicted p(YRI))")
    ax2.set_ylabel("Mean predicted p(YRI)")
    ax2.set_title("Mean predicted probability per decile")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved domain decile enrichment plot to: {output_path}")

    return {
        "decile_frac_yri": decile_frac_yri,
        "decile_mean_prob": decile_mean_prob,
    }
