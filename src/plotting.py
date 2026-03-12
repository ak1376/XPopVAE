import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


@torch.no_grad()
def plot_reconstruction(model, dataloader, device, output_dir, sample_idx=0):
    model.eval()

    x_batch, _ = next(iter(dataloader))
    x_batch = x_batch.to(device)
    x_recon, mu, logvar, z = model(x_batch)

    x_true = x_batch[sample_idx, 0].cpu().numpy()
    x_hat = x_recon[sample_idx, 0].cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.plot(x_true, label="true")
    plt.plot(x_hat, label="reconstruction")
    plt.legend()
    plt.xlabel("position")
    plt.ylabel("value")
    plt.title(f"Sample {sample_idx}: true vs reconstruction")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reconstruction_sample_{sample_idx}.png")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(x_true, x_hat, alpha=0.3, s=10)
    min_val = min(x_true.min(), x_hat.min())
    max_val = max(x_true.max(), x_hat.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("true values")
    plt.ylabel("reconstructed values")
    plt.title("Pointwise true vs reconstructed")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reconstruction_scatter_sample_{sample_idx}.png")
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