import torch
import torch.nn.functional as F


def recon_unmasked_loss(logits, targets, mask):
    if mask.all():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits.permute(0, 2, 1)
    unmasked_logits  = logits[~mask]
    unmasked_targets = targets[~mask]
    return F.cross_entropy(unmasked_logits, unmasked_targets, reduction="mean")

def recon_masked_loss(logits, targets, mask):
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits.permute(0, 2, 1)
    masked_logits  = logits[mask]
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

def orthogonality_loss(z_recon, z_pheno):
    # Centre each subspace across the batch
    z_r = z_recon - z_recon.mean(dim=0)
    z_p = z_pheno - z_pheno.mean(dim=0)
    # Cross-covariance matrix: (recon_dim, pheno_dim)
    cross_cov = z_r.T @ z_p
    return (cross_cov ** 2).sum()


def vae_loss(recon_unmasked, recon_masked, kl, pheno_loss, z_recon, z_pheno, alpha=1.0, beta=1.0, gamma=1.0, lambda_ortho=1e-3):
    ortho_loss = orthogonality_loss(z_recon, z_pheno)
    loss = recon_unmasked + alpha * recon_masked + beta * kl + gamma * pheno_loss + lambda_ortho * ortho_loss
    return loss, recon_unmasked, recon_masked, kl, pheno_loss, ortho_loss