import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.preprocessing import label_binarize


@torch.no_grad()
def plot_reconstruction(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    x_batch, _ = next(iter(dataloader))
    x_batch = x_batch.to(device)

    logits, mu, logvar, z = model(x_batch)  # logits shape: (B, 3, L)

    # True labels: (B, L)
    y_true = x_batch.long().squeeze(1).cpu().numpy()

    # Predicted probabilities: (B, 3, L) -> (B, L, 3)
    probs = F.softmax(logits, dim=1).permute(0, 2, 1).cpu().numpy()

    # Hard predictions for balanced accuracy
    y_pred = np.argmax(probs, axis=-1)

    # Flatten across batch and sequence positions
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    probs_flat = probs.reshape(-1, probs.shape[-1])  # (N_positions_total, 3)

    # Balanced accuracy on hard predictions
    bal_acc = balanced_accuracy_score(y_true_flat, y_pred_flat)

    # Binarize labels for multiclass ROC
    classes = np.array([0, 1, 2])
    y_true_bin = label_binarize(y_true_flat, classes=classes)  # (N, 3)

    # Micro-average ROC
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probs_flat.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Micro-average ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Reconstruction ROC | AUC = {roc_auc:.3f} | Balanced Acc = {bal_acc:.3f}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reconstruction_roc.png")
    plt.close()


@torch.no_grad()
def plot_latent_space(model, dataloader, device, output_dir, save_path="latent_space.png"):
    model.eval()

    all_mu = []
    all_labels = []

    for x_batch, y_batch in dataloader:
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
    train_recon_losses,
    val_recon_losses,
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
    plt.plot(epochs, train_recon_losses, label="train recon")
    plt.plot(epochs, val_recon_losses, label="val recon")
    plt.xlabel("epoch")
    plt.ylabel("reconstruction loss")
    plt.title("Reconstruction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_recon.png")
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