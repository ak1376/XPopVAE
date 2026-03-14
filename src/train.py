import torch


def train_one_epoch(model, dataloader, optimizer, device, loss_fn, beta=1.0):
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    for x, _ in dataloader:
        x = x.to(device)

        optimizer.zero_grad()

        x_logits, mu, logvar, z = model(x)
        loss, recon_loss, kl_loss = loss_fn(
            x=x,
            logits=x_logits,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        # print("mu abs mean:", mu.abs().mean().item())
        # print("mu abs max:", mu.abs().max().item())
        # print("logvar mean:", logvar.mean().item())
        # print("logvar max:", logvar.max().item())
        # print("logvar min:", logvar.min().item())
        # print("logits abs max:", x_logits.abs().max().item())


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

        logits, mu, logvar, z = model(x)
        loss, recon_loss, kl_loss = loss_fn(
            x=x,
            logits=logits,
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