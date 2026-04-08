import torch
import torch.nn.functional as F


def recon_unmasked_loss(logits, targets, mask):
    if mask.all():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits.permute(0, 2, 1)
    return F.cross_entropy(logits[~mask], targets[~mask], reduction="mean")


def recon_masked_loss(logits, targets, mask):
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits.permute(0, 2, 1)
    return F.cross_entropy(logits[mask], targets[mask], reduction="mean")


def recon_all_loss(logits, targets):
    return F.cross_entropy(logits, targets, reduction="mean")


def kl_loss(mu_shared, logvar_shared, mu_private, logvar_private):
    """
    KL divergence summed over both shared and private subspaces.
    Each term: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    """
    kl_shared  = -0.5 * torch.mean(1 + logvar_shared  - mu_shared.pow(2)  - logvar_shared.exp())
    kl_private = -0.5 * torch.mean(1 + logvar_private - mu_private.pow(2) - logvar_private.exp())
    return kl_shared + kl_private


def phenotype_loss(pheno_pred, pheno_true):
    return F.mse_loss(pheno_pred, pheno_true, reduction="mean")


def domain_loss(domain_logits: torch.Tensor, pop_labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy between domain classifier logits and population labels
    (0 = CEU/discovery, 1 = YRI/target).
    """
    return F.cross_entropy(domain_logits, pop_labels, reduction="mean")


def vae_loss(recon_unmasked, recon_masked, kl, pheno_loss, alpha=1.0, beta=1.0, gamma=1.0):
    loss = (
        alpha       * recon_masked
        + (1-alpha) * recon_unmasked
        + beta      * kl
        + gamma     * pheno_loss
    )
    return loss, recon_unmasked, recon_masked, kl, pheno_loss