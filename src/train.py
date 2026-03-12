import torch


def train_one_epoch(model, dataloader, optimizer, device, loss_fn, beta=1.0):
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    for x, _ in dataloader:
        x = x.to(device)

        optimizer.zero_grad()

        x_recon, mu, logvar, z = model(x)
        loss, recon_loss, kl_loss = loss_fn(
            x=x,
            x_recon=x_recon,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    n_batches = len(dataloader)
    return (
        total_loss / n_batches,
        total_recon_loss / n_batches,
        total_kl_loss / n_batches,
    )


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn, beta=1.0):
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    for x, _ in dataloader:
        x = x.to(device)

        x_recon, mu, logvar, z = model(x)
        loss, recon_loss, kl_loss = loss_fn(
            x=x,
            x_recon=x_recon,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    n_batches = len(dataloader)
    return (
        total_loss / n_batches,
        total_recon_loss / n_batches,
        total_kl_loss / n_batches,
    )