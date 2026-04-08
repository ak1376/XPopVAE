import torch
from src.loss import recon_unmasked_loss, recon_masked_loss, kl_loss, phenotype_loss, domain_loss


# =============================================================================
# GRL lambda schedule (Ganin et al. 2016)
# =============================================================================

def compute_grl_lambda(current_epoch: int, total_epochs: int, lambda_max: float = 1.0) -> float:
    import math
    p = current_epoch / max(total_epochs, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# =============================================================================
# Domain accuracy (no grad)
# =============================================================================

@torch.no_grad()
def domain_accuracy(domain_logits: torch.Tensor, pop_labels: torch.Tensor) -> float:
    return (domain_logits.argmax(dim=1) == pop_labels).float().mean().item()


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    loss_fn,
    masker,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    use_grl: bool = False,
    delta: float = 1.0,
):
    """
    Returns 7 floats:
        total_loss, recon_unmasked, recon_masked, kl, pheno, domain_ce, domain_acc
    """
    model.train()

    total_loss           = 0.0
    total_recon_unmasked = 0.0
    total_recon_masked   = 0.0
    total_kl             = 0.0
    total_pheno          = 0.0
    total_domain         = 0.0
    total_domain_acc     = 0.0

    for x, pheno, pop_label in dataloader:
        x         = x.to(device)
        pheno     = pheno.to(device)
        pop_label = pop_label.to(device)

        if masker is not None:
            input_x, mask = masker.mask(x)
        else:
            input_x = x
            mask    = torch.zeros(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
        input_x = input_x.to(device)
        mask    = mask.to(device)

        optimizer.zero_grad()

        (out,
         mu_shared, logvar_shared,
         mu_private, logvar_private,
         z_shared, z_private,
         pheno_pred, domain_logits) = model(input_x)

        targets = x.squeeze(1).long()

        recon_u = recon_unmasked_loss(out, targets, mask)
        recon_m = recon_masked_loss(out, targets, mask)
        kl      = kl_loss(mu_shared, logvar_shared, mu_private, logvar_private)

        ceu_mask = (pop_label == 0)
        pheno_l  = (
            phenotype_loss(pheno_pred[ceu_mask], pheno[ceu_mask])
            if ceu_mask.any()
            else torch.tensor(0.0, device=device)
        )

        loss, recon_u_val, recon_m_val, kl_val, pheno_val = loss_fn(
            recon_unmasked=recon_u,
            recon_masked=recon_m,
            kl=kl,
            pheno_loss=pheno_l,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        # GRL domain loss on shared subspace only
        if use_grl and domain_logits is not None:
            d_loss = domain_loss(domain_logits, pop_label)
            d_acc  = domain_accuracy(domain_logits, pop_label)  # kept for logging only
            loss   = loss + delta * d_loss                       # fixed weight, no d_acc scaling
            total_domain     += d_loss.item()
            total_domain_acc += d_acc
        else:
            total_domain     += 0.0
            total_domain_acc += 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss           += loss.item()
        total_recon_unmasked += recon_u_val.item()
        total_recon_masked   += recon_m_val.item()
        total_kl             += kl_val.item()
        total_pheno          += pheno_val.item()

    n = len(dataloader)
    return (
        total_loss           / n,
        total_recon_unmasked / n,
        total_recon_masked   / n,
        total_kl             / n,
        total_pheno          / n,
        total_domain         / n,
        total_domain_acc     / n,
    )


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Returns 5 floats: total_loss, recon_unmasked, recon_masked, kl, pheno_loss.
    """
    model.eval()

    total_loss           = 0.0
    total_recon_unmasked = 0.0
    total_recon_masked   = 0.0
    total_kl             = 0.0
    total_pheno          = 0.0

    for input_x, x, pheno, mask, pop_label in dataloader:
        input_x   = input_x.to(device)
        x         = x.to(device)
        pheno     = pheno.to(device)
        mask      = mask.to(device)
        pop_label = pop_label.to(device)

        (out,
         mu_shared, logvar_shared,
         mu_private, logvar_private,
         z_shared, z_private,
         pheno_pred, _domain_logits) = model(input_x)

        targets = x.squeeze(1).long()

        recon_u = recon_unmasked_loss(out, targets, mask)
        recon_m = recon_masked_loss(out, targets, mask)
        kl      = kl_loss(mu_shared, logvar_shared, mu_private, logvar_private)
        pheno_l = phenotype_loss(pheno_pred, pheno)

        loss, recon_u_val, recon_m_val, kl_val, pheno_val = loss_fn(
            recon_unmasked=recon_u,
            recon_masked=recon_m,
            kl=kl,
            pheno_loss=pheno_l,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        total_loss           += loss.item()
        total_recon_unmasked += recon_u_val.item()
        total_recon_masked   += recon_m_val.item()
        total_kl             += kl_val.item()
        total_pheno          += pheno_val.item()

    n = len(dataloader)
    return (
        total_loss           / n,
        total_recon_unmasked / n,
        total_recon_masked   / n,
        total_kl             / n,
        total_pheno          / n,
    )


# =============================================================================
# Early stopping
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience       = patience
        self.min_delta      = min_delta
        self.best_loss      = float("inf")
        self.num_bad_epochs = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss      = val_loss
            self.num_bad_epochs = 0
            return True, False
        else:
            self.num_bad_epochs += 1
            return False, self.num_bad_epochs >= self.patience