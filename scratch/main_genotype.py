from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.loss import vae_loss
from src.model import ConvVAE
from src.plotting import plot_latent_space, plot_loss_curves, plot_reconstruction
from src.train import evaluate, train_one_epoch
import numpy as np
import yaml 

def main():

    # Load in the datasets 
    training_dataset = np.load('experiments/IM_symmetric/processed_data/discovery_train.npy')
    validation_dataset = np.load('experiments/IM_symmetric/processed_data/discovery_val.npy')

    # Load in the VAE YAML
    with open('/sietch_colab/akapoor/XPopVAE/config_files/model_hyperparams/vae.yaml', 'r') as f:
        vae_config = yaml.safe_load(f)

    # Get the hyperparameters from the YAML file 
    input_length = 7183
    in_channels = 1
    hidden_channels = vae_config['model']['hidden_channels']
    kernel_size = int(vae_config['model']['kernel_size'])
    stride = int(vae_config['model']['stride'])
    padding = int(vae_config['model']['padding'])
    latent_dim = int(vae_config['model']['latent_dim'])
    beta = float(vae_config['training']['beta'])
    learning_rate = float(vae_config['training']['lr'])
    batch_size = int(vae_config['training']['batch_size'])
    num_epochs = int(vae_config['training']['max_epochs'])

    training_dataset_torch = torch.tensor(training_dataset, dtype=torch.float32)
    training_dataset_torch = training_dataset_torch.unsqueeze(1)
    validation_dataset_torch = torch.tensor(validation_dataset, dtype=torch.float32)
    validation_dataset_torch = validation_dataset_torch.unsqueeze(1)

    # Add dummy labels (all zeros) since our training loop expects (x, y) pairs
    training_dataset_torch = TensorDataset(training_dataset_torch, torch.zeros(len(training_dataset_torch)))
    validation_dataset_torch = TensorDataset(validation_dataset_torch, torch.zeros(len(validation_dataset_torch)))

    train_loader = DataLoader(training_dataset_torch, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset_torch, batch_size=batch_size, shuffle=False)

    out_path = Path("model_outputs")
    out_path.mkdir(exist_ok=True)

    # Visualize one batch of training data
    a = next(iter(train_loader))[0]  # Get the first batch of inputs
    print("Batch shape:", a.shape)  # Should be (batch_size, 1, 7183)
    print("Batch dtype:", a.dtype)  # Should be torch.float32
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

    # Let's print the number of parameters to make sure it's not too large
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # Pass one batch through the model with shape tracing
    x_batch = next(iter(train_loader))[0].to(device)

    with torch.no_grad():
        logits, mu, logvar, z = model(x_batch, verbose=True)

    print("\nFinal outputs:")
    print("logits shape:", logits.shape)   # expected: (B, 3, 7183)
    print("mu shape:", mu.shape)           # expected: (B, latent_dim)
    print("logvar shape:", logvar.shape)   # expected: (B, latent_dim)
    print("z shape:", z.shape)             # expected: (B, latent_dim)

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


if __name__ == "__main__":
    main()