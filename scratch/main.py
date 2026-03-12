from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import make_two_process_dataset
from src.loss import vae_loss
from src.model import ConvVAE
from src.plotting import plot_latent_space, plot_loss_curves, plot_reconstruction
from src.train import evaluate, train_one_epoch




def main():
    input_length = 1000
    in_channels = 1
    hidden_channels = [32, 64, 128]
    kernel_size = 3
    stride = 2
    padding = 1
    latent_dim = 64
    beta = 1.0
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 100
    n_samples = 2000

    hyperparams = {
        "input_length": input_length,
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "latent_dim": latent_dim,
        "beta": beta,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "n_samples": n_samples,
    }

    out_path = Path("model_outputs")
    out_path.mkdir(exist_ok=True)

    with open(out_path / "hyperparameters.txt", "w") as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")

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

    X, y = make_two_process_dataset(
        n_samples=n_samples,
        input_length=input_length,
        noise_std=0.1,
    )

    train_dataset = TensorDataset(X[:1600], y[:1600])
    val_dataset = TensorDataset(X[1600:], y[1600:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            f"train_recon={train_recon:.6f} | "
            f"train_kl={train_kl:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_recon={val_recon:.6f} | "
            f"val_kl={val_kl:.6f}"
        )

    plot_reconstruction(model, val_loader, device, output_dir=out_path, sample_idx=0)
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