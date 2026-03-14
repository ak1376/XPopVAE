import torch
import torch.nn.functional as F

def vae_loss(x, logits, mu, logvar, beta=1.0):
    targets = x.long().squeeze(1)   # (B, L), values in {0,1,2}

    recon_loss = F.cross_entropy(logits, targets, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss