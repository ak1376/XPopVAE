import torch
import torch.nn.functional as F


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss