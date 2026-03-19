import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score


@torch.no_grad()
def plot_reconstruction(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_y_true = []
    all_y_pred = []

    for batch in dataloader:
        if len(batch) == 2:
            # loader: (x, label)
            x, _ = batch

        elif len(batch) == 4:
            # loader: (masked_x, x_true, mask, label)
            # since you are no longer doing masked reconstruction,
            # evaluate on the true unmasked genotype matrix
            _, x, _, _ = batch

        else:
            raise ValueError(f"Unexpected batch structure of length {len(batch)}")

        x = x.to(device)

        logits, mu, logvar, z = model(x)  # logits: (B, 3, L)

        # Ground truth and predictions: (B, L)
        y_true = x.long().squeeze(1).cpu().numpy()
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    # Concatenate across all batches
    y_true_all = np.concatenate(all_y_true, axis=0)   # (N, L)
    y_pred_all = np.concatenate(all_y_pred, axis=0)   # (N, L)

    # Flatten across the full genotype matrix
    y_true_flat = y_true_all.reshape(-1)
    y_pred_flat = y_pred_all.reshape(-1)

    classes = np.array([0, 1, 2])

    # Per-class recall and balanced accuracy
    recalls = recall_score(
        y_true_flat,
        y_pred_flat,
        labels=classes,
        average=None,
        zero_division=0,
    )
    bal_acc = recalls.mean()

    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=classes)

    # Row-normalized confusion matrix
    cm_row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        cm_row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=cm_row_sums != 0,
    )

    # Plot
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
        "balanced_accuracy": bal_acc,
        "recalls": {int(cls): float(rec) for cls, rec in zip(classes, recalls)},
        "confusion_matrix_raw": cm,
        "confusion_matrix_normalized": cm_normalized,
    }

@torch.no_grad()
def plot_latent_space(model, dataloader, device, output_dir, save_path="latent_space.png"):
    model.eval()

    all_mu = []
    all_labels = []

    for batch in dataloader:
        if len(batch) == 2:
            # training-style loader: (x, label)
            x_batch, y_batch = batch
        elif len(batch) == 4:
            # validation/test-style loader: (masked_x, x_true, mask, label)
            _, x_batch, _, y_batch = batch
        else:
            raise ValueError(f"Unexpected batch structure of length {len(batch)}")

        x_batch = x_batch.to(device)
        _, mu, _, _ = model(x_batch)

        all_mu.append(mu.cpu())
        all_labels.append(y_batch.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    mu_2d = PCA(n_components=2).fit_transform(all_mu.numpy())

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        mu_2d[:, 0],
        mu_2d[:, 1],
        c=all_labels.numpy(),
        cmap="coolwarm",
        alpha=0.7,
        s=20,
    )
    plt.xlabel("latent PC1")
    plt.ylabel("latent PC2")
    plt.title("Latent representation (PCA of mu)")
    plt.colorbar(scatter, label="population")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{save_path}")
    plt.close()


def plot_loss_curves(
    train_losses,
    val_losses,
    train_recon_unmasked_losses,
    train_recon_masked_losses,
    val_recon_unmasked_losses,
    val_recon_masked_losses,
    train_kl_losses,
    val_kl_losses,
    output_dir,
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train total")
    plt.plot(epochs, val_losses, label="val total")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_total.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_recon_unmasked_losses, label="train recon unmasked")
    plt.plot(epochs, val_recon_unmasked_losses, label="val recon unmasked")
    plt.xlabel("epoch")
    plt.ylabel("recon unmasked loss")
    plt.title("Reconstruction loss (unmasked positions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_recon_unmasked.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_recon_masked_losses, label="train recon masked")
    plt.plot(epochs, val_recon_masked_losses, label="val recon masked")
    plt.xlabel("epoch")
    plt.ylabel("recon masked loss")
    plt.title("Reconstruction loss (masked positions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_recon_masked.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_kl_losses, label="train kl")
    plt.plot(epochs, val_kl_losses, label="val kl")
    plt.xlabel("epoch")
    plt.ylabel("KL loss")
    plt.title("KL loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_kl.png")
    plt.close()

def plot_example_masked_input_heatmap(
    original_x,
    masked_x,
    mask,
    output_path,
    sample_indices=(0, 1, 2, 3, 4),
    snp_start=0,
    snp_count=1000,
):
    """
    Plot heatmaps for:
      1) original genotype input
      2) masked genotype input
      3) binary mask

    Supports:
      - original_x, masked_x of shape (N, 1, L) or (N, L)
      - mask of shape (N, L) or (N, 1, L)

    Assumes mask = 1 at masked positions.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    snp_end = min(snp_start + snp_count, original_x.shape[-1])
    sample_indices = list(sample_indices)

    def _slice_tensor(x, sample_indices, snp_start, snp_end):
        if x.ndim == 3:
            # (N, C, L) -> take first channel
            return x[sample_indices, 0, snp_start:snp_end].detach().cpu().numpy()
        elif x.ndim == 2:
            # (N, L)
            return x[sample_indices, snp_start:snp_end].detach().cpu().numpy()
        else:
            raise ValueError(f"Expected tensor with 2 or 3 dims, got shape {tuple(x.shape)}")

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

def plot_latent_pca_shared_basis(
    reference_mu,
    ceu_mu,
    yri_mu,
    output_path,
    reference_name="CEU discovery train",
    ceu_name="CEU validation",
    yri_name="YRI target",
):
    # Fit scaler on reference only
    scaler = StandardScaler()
    reference_mu_scaled = scaler.fit_transform(reference_mu)
    ceu_mu_scaled = scaler.transform(ceu_mu)
    yri_mu_scaled = scaler.transform(yri_mu)

    # Fit PCA on reference only
    pca = PCA(n_components=2)
    pca.fit(reference_mu_scaled)

    ceu_pca = pca.transform(ceu_mu_scaled)
    yri_pca = pca.transform(yri_mu_scaled)

    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(7, 6))
    plt.scatter(
        ceu_pca[:, 0],
        ceu_pca[:, 1],
        alpha=0.7,
        s=20,
        label=ceu_name,
    )
    plt.scatter(
        yri_pca[:, 0],
        yri_pca[:, 1],
        alpha=0.7,
        s=20,
        label=yri_name,
    )

    plt.xlabel(f"PC1 ({explained[0] * 100:.2f}% var)")
    plt.ylabel(f"PC2 ({explained[1] * 100:.2f}% var)")
    plt.title(f"Latent PCA\nfit on {reference_name}, both projected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved shared-basis PCA plot to: {output_path}")