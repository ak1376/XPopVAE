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
from src.plotting import plot_latent_space, plot_loss_curves, plot_reconstruction, plot_example_masked_input_heatmap, plot_latent_pca_shared_basis
from src.train import evaluate, train_one_epoch
from src.masking import Masker

def extract_mu(model, dataloader, device):
    model.eval()
    mu_all = []
    labels_all = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                # training-style loader: (x, label)
                x, y = batch
            elif len(batch) == 4:
                # val/test-style loader: (masked_x, x_true, mask, label)
                x, _, _, y = batch
            else:
                raise ValueError(f"Unexpected batch structure of length {len(batch)}")

            x = x.to(device)
            logits, mu, logvar, z = model(x)
            mu_all.append(mu.cpu().numpy())
            labels_all.append(y.cpu().numpy())

    mu_all = np.concatenate(mu_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return mu_all, labels_all

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

    # Masking hyperparameters:
    alpha = float(vae_config['masking']['alpha_masked'])
    block_length = int(vae_config['masking']['block_len'])
    mask_frac = float(vae_config['masking']['mask_frac'])

    masker = Masker(block_length=block_length, mask_fraction=mask_frac)

    training_dataset_torch = torch.tensor(training_dataset, dtype=torch.float32).unsqueeze(1)  # add channel dim
    validation_dataset_torch = torch.tensor(validation_dataset, dtype=torch.float32).unsqueeze(1)
    target_dataset_torch = torch.tensor(target_dataset, dtype=torch.float32).unsqueeze(1)

    # Precompute the masked dataset for validation (NOT training)
    masked_val_x, val_mask = masker.mask(validation_dataset_torch)
    masked_target_x, target_mask = masker.mask(target_dataset_torch)

    # Plot one example masked input
    out_path = Path("model_outputs")
    out_path.mkdir(exist_ok=True)

    plot_example_masked_input_heatmap(
        original_x=validation_dataset_torch,
        masked_x=masked_val_x,
        mask=val_mask,
        output_path=out_path / "example_masked_input_heatmap_val.png",
        sample_indices=(0, 1, 2, 3, 4),
        snp_start=0,
        snp_count=1000,
    )

    # Dummy labels:
    # 0 = CEU/discovery
    # 1 = YRI/target
    training_dataset_torch = TensorDataset(
        training_dataset_torch, torch.zeros(len(training_dataset_torch), dtype=torch.long)
    )

    validation_dataset_torch = TensorDataset(
        masked_val_x,                 # masked input
        validation_dataset_torch,     # original input (reconstruction target)
        val_mask,                     # binary mask
        torch.zeros(len(validation_dataset_torch), dtype=torch.long)  # label
    )

    # Should this use a different mask than the validation set? Or no mask at all? 
    target_dataset_torch = TensorDataset(
        masked_target_x,              # masked input
        target_dataset_torch,        # original input (reconstruction target)
        target_mask,                 # binary mask
        torch.ones(len(target_dataset_torch), dtype=torch.long)  # label
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

    train_loss_list, train_recon_unmasked_list, train_recon_masked_list, train_kl_list = [], [], [], []
    val_loss_list, val_recon_unmasked_list, val_recon_masked_list, val_kl_list = [], [], [], []

    masker = Masker(block_length=50, mask_fraction=0.2)

    for epoch in range(num_epochs):
        train_loss, train_recon_unmasked, train_recon_masked, train_kl = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=vae_loss,
            masker=masker,
            beta=beta,
            alpha = alpha
        )

        val_loss, val_recon_unmasked, val_recon_masked, val_kl = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=vae_loss,
            beta=beta,
            alpha = alpha
        )

        train_loss_list.append(train_loss)
        train_recon_unmasked_list.append(train_recon_unmasked)
        train_recon_masked_list.append(train_recon_masked)
        train_kl_list.append(train_kl)

        val_loss_list.append(val_loss)
        val_recon_unmasked_list.append(val_recon_unmasked)
        val_recon_masked_list.append(val_recon_masked)
        val_kl_list.append(val_kl)
        

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"train_recon_unmasked={train_recon_unmasked:.6f} | "
            f"train_recon_masked={train_recon_masked:.6f} | "
            f"train_kl={train_kl:.6f} | "
            f"val_recon_unmasked={val_recon_unmasked:.6f} | "
            f"val_recon_masked={val_recon_masked:.6f} | "
            f"val_kl={val_kl:.6f}"
        )

    # Existing plots for validation
    plot_reconstruction(model, val_loader, device, output_dir=out_path)
    plot_latent_space(model, val_loader, device, output_dir=out_path)
    plot_loss_curves(
        train_losses=train_loss_list,
        val_losses=val_loss_list,
        train_recon_unmasked_losses=train_recon_unmasked_list,
        train_recon_masked_losses=train_recon_masked_list,
        val_recon_unmasked_losses=val_recon_unmasked_list,
        val_recon_masked_losses=val_recon_masked_list,
        train_kl_losses=train_kl_list,
        val_kl_losses=val_kl_list,
        output_dir=out_path,
    )

    # Evaluate on target dataset
    target_loss, target_recon_unmasked, target_recon_masked, target_kl = evaluate(
        model=model,
        dataloader=target_loader,
        device=device,
        loss_fn=vae_loss,
        alpha=alpha,
        beta=beta,
    )
    print("\nTarget set evaluation:")
    print(f"target_loss  = {target_loss:.6f}")
    print(f"target_recon_unmasked = {target_recon_unmasked:.6f}")
    print(f"target_recon_masked   = {target_recon_masked:.6f}")
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