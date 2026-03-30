import torch
import torch.nn.functional as F


def recon_unmasked_loss(logits, targets, mask):
    if mask.all():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits.permute(0, 2, 1)
    unmasked_logits = logits[~mask]
    unmasked_targets = targets[~mask]
    return F.cross_entropy(unmasked_logits, unmasked_targets, reduction="mean")


def recon_masked_loss(logits, targets, mask):
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits.permute(0, 2, 1)
    masked_logits = logits[mask]
    masked_targets = targets[mask]
    return F.cross_entropy(masked_logits, masked_targets, reduction="mean")


def recon_all_loss(logits, targets):
    """
    logits:  [B, C, X]
    targets: [B, X]
    """
    return F.cross_entropy(logits, targets, reduction="mean")


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def phenotype_loss(pheno_pred, pheno_true):
    return F.mse_loss(pheno_pred, pheno_true, reduction="mean")


def vae_loss(
    recon_unmasked, recon_masked, kl, pheno_loss, alpha=1.0, beta=1.0, gamma=1.0
):
    loss = (
        alpha * recon_masked
        + (1 - alpha) * recon_unmasked
        + beta * kl
        + gamma * pheno_loss
    )
    return loss, recon_unmasked, recon_masked, kl, pheno_loss
