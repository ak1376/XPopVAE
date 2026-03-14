from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.loss import vae_loss
from src.model import ConvVAE
from src.plotting import plot_latent_space, plot_loss_curves, plot_reconstruction
from src.train import evaluate, train_one_epoch


def extract_mu(model, dataloader, device):
    model.eval()
    mu_all = []
    labels_all = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits, mu, logvar, z = model(x)
            mu_all.append(mu.cpu().numpy())
            labels_all.append(y.cpu().numpy())

    mu_all = np.concatenate(mu_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return mu_all, labels_all


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


def main():
    # Load in the datasets
    training_dataset = np.load("experiments/IM_symmetric/processed_data/discovery_train.npy")
    validation_dataset = np.load("experiments/IM_symmetric/processed_data/discovery_val.npy")
    target_dataset = np.load("experiments/IM_symmetric/processed_data/target.npy")

    # Load in the VAE YAML
    with open("/sietch_colab/akapoor/XPopVAE/config_files/model_hyperparams/vae.yaml", "r") as f:
        vae_config = yaml.safe_load(f)

    # Get the hyperparameters from the YAML file
    input_length = 7183
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

    training_dataset_torch = torch.tensor(training_dataset, dtype=torch.float32).unsqueeze(1)
    validation_dataset_torch = torch.tensor(validation_dataset, dtype=torch.float32).unsqueeze(1)
    target_dataset_torch = torch.tensor(target_dataset, dtype=torch.float32).unsqueeze(1)

    # Dummy labels:
    # 0 = CEU/discovery
    # 1 = YRI/target
    training_dataset_torch = TensorDataset(
        training_dataset_torch, torch.zeros(len(training_dataset_torch), dtype=torch.long)
    )
    validation_dataset_torch = TensorDataset(
        validation_dataset_torch, torch.zeros(len(validation_dataset_torch), dtype=torch.long)
    )
    target_dataset_torch = TensorDataset(
        target_dataset_torch, torch.ones(len(target_dataset_torch), dtype=torch.long)
    )

    train_loader = DataLoader(training_dataset_torch, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset_torch, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset_torch, batch_size=batch_size, shuffle=False)

    out_path = Path("model_outputs")
    out_path.mkdir(exist_ok=True)

    # Visualize one batch of training data
    a = next(iter(train_loader))[0]
    print("Batch shape:", a.shape)
    print("Batch dtype:", a.dtype)
    print("Batch min/max:", a.min().item(), a.max().item())

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
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # Pass one batch through the model with shape tracing
    x_batch = next(iter(train_loader))[0].to(device)

    with torch.no_grad():
        logits, mu, logvar, z = model(x_batch, verbose=True)

    print("\nFinal outputs:")
    print("logits shape:", logits.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("z shape:", z.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, train_recon_losses, train_kl_losses = [], [], []
    val_losses, val_recon_losses, val_kl_losses = [], [], []

    for epoch in range(num_epochs):
        train_loss, train_recon, train_kl = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=vae_loss,
            beta=beta,
        )

        val_loss, val_recon, val_kl = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=vae_loss,
            beta=beta,
        )

        train_losses.append(train_loss)
        train_recon_losses.append(train_recon)
        train_kl_losses.append(train_kl)

        val_losses.append(val_loss)
        val_recon_losses.append(val_recon)
        val_kl_losses.append(val_kl)

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"train_kl={train_kl:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_kl={val_kl:.6f}"
        )

    # Existing plots for validation
    plot_reconstruction(model, val_loader, device, output_dir=out_path)
    plot_latent_space(model, val_loader, device, output_dir=out_path)
    plot_loss_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_recon_losses=train_recon_losses,
        val_recon_losses=val_recon_losses,
        train_kl_losses=train_kl_losses,
        val_kl_losses=val_kl_losses,
        output_dir=out_path,
    )

    # Evaluate on target dataset
    target_loss, target_recon, target_kl = evaluate(
        model=model,
        dataloader=target_loader,
        device=device,
        loss_fn=vae_loss,
        beta=beta,
    )

    print("\nTarget set evaluation:")
    print(f"target_loss  = {target_loss:.6f}")
    print(f"target_recon = {target_recon:.6f}")
    print(f"target_kl    = {target_kl:.6f}")

    target_out_path = out_path / "target"
    target_out_path.mkdir(exist_ok=True)

    plot_reconstruction(model, target_loader, device, output_dir=target_out_path)
    plot_latent_space(model, target_loader, device, output_dir=target_out_path)

    # -----------------------------------------
    # Shared-coordinate latent PCA:
    # fit on CEU discovery train, project CEU val + YRI target
    # -----------------------------------------
    train_mu, _ = extract_mu(model, train_loader, device)
    val_mu, _ = extract_mu(model, val_loader, device)
    target_mu, _ = extract_mu(model, target_loader, device)

    shared_pca_path = out_path / "latent_pca_ceu_basis_val_vs_target.png"
    plot_latent_pca_shared_basis(
        reference_mu=train_mu,
        ceu_mu=val_mu,
        yri_mu=target_mu,
        output_path=shared_pca_path,
        reference_name="CEU discovery train",
        ceu_name="CEU validation",
        yri_name="YRI target",
    )


if __name__ == "__main__":
    main()