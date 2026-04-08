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


def domain_loss(domain_logits: torch.Tensor, pop_labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy between predicted domain logits and ground-truth population
    labels (0 = CEU/discovery, 1 = YRI/target).

    Parameters
    ----------
    domain_logits : (B, num_domains)  raw logits from the domain classifier
    pop_labels    : (B,)              integer labels, 0 or 1

    Returns
    -------
    Scalar CE loss tensor (with grad).
    """
    return F.cross_entropy(domain_logits, pop_labels, reduction="mean")


def vae_loss(
    recon_unmasked,
    recon_masked,
    kl,
    pheno_loss,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    """
    Core VAE loss (reconstruction + KL + phenotype).
    Domain loss is added on top of this in train_one_epoch so that
    the adaptive delta scaling lives in one place.

    Returns the weighted total and each component for logging.
    """
    loss = (
        alpha * recon_masked
        + (1 - alpha) * recon_unmasked
        + beta  * kl
        + gamma * pheno_loss
    )
    return loss, recon_unmasked, recon_masked, kl, pheno_loss