import torch
from src.loss import (
    recon_unmasked_loss,
    recon_masked_loss,
    kl_loss,
    phenotype_loss,
    domain_loss,
    mmd_loss,
)

# =============================================================================
# GRL lambda schedule (kept for use_grl=True configs)
# =============================================================================


def compute_grl_lambda(
    current_epoch: int,
    total_epochs: int,
    lambda_max: float = 1.0,
) -> float:
    """
    Ganin et al. (2016) schedule:
        lambda = lambda_max * (2 / (1 + exp(-10 * p)) - 1)
    where p = current_epoch / total_epochs in [0, 1].
    """
    import math

    p = current_epoch / max(total_epochs, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# =============================================================================
# Domain-accuracy helper
# =============================================================================


@torch.no_grad()
def domain_accuracy(domain_logits: torch.Tensor, pop_labels: torch.Tensor) -> float:
    preds = domain_logits.argmax(dim=1)
    return (preds == pop_labels).float().mean().item()


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
    # --- adversarial GRL on z_task ---
    use_grl: bool = False,
    # --- MMD on z_task ---
    use_mmd: bool = False,
    delta: float = 1.0,   # weight for both GRL and MMD
    # --- cooperative domain classifier on z_domain ---
    use_domain_clf: bool = False,
    lam_domain: float = 1.0,
):
    model.train()

    total_loss = 0.0
    total_recon_unmasked = 0.0
    total_recon_masked = 0.0
    total_kl_loss = 0.0
    total_phenotype_loss = 0.0
    total_grl_loss = 0.0
    total_grl_acc = 0.0
    total_domain_clf_loss = 0.0
    total_domain_acc = 0.0
    total_mmd_loss = 0.0
    total_z_task_var = 0.0
    has_z_domain = model.task_dim < model.latent_dim
    total_z_domain_var = 0.0 if has_z_domain else None
    batch_ceu_fracs = []

    for x, pheno, pop_label in dataloader:
        x = x.to(device)
        pheno = pheno.to(device)
        pop_label = pop_label.to(device)
        batch_ceu_fracs.append((pop_label == 0).float().mean().item())

        if masker is not None:
            input_x, mask = masker.mask(x)
        else:
            input_x = x
            mask = torch.zeros(
                x.shape[0], x.shape[2], dtype=torch.bool, device=x.device
            )
        input_x = input_x.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        out, mu, logvar, z, pheno_pred, grl_logits, domain_logits = model(input_x)
        targets = x.squeeze(1).long()

        stats = model.latent_stats(mu)
        total_z_task_var += stats["z_task_var"]
        if has_z_domain:
            total_z_domain_var += stats["z_domain_var"]

        recon_unmasked = recon_unmasked_loss(out, targets, mask)
        recon_masked_l = recon_masked_loss(out, targets, mask)
        kl = kl_loss(mu, logvar)

        ceu_mask = pop_label == 0
        if ceu_mask.any():
            pheno_l = phenotype_loss(pheno_pred[ceu_mask], pheno[ceu_mask])
        else:
            pheno_l = torch.tensor(0.0, device=device, requires_grad=False)

        loss, recon_u_val, recon_m_val, kl_val, pheno_val = loss_fn(
            recon_unmasked=recon_unmasked,
            recon_masked=recon_masked_l,
            kl=kl,
            pheno_loss=pheno_l,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        # ------------------------------------------------------------------
        # Adversarial GRL on z_task (optional)
        # ------------------------------------------------------------------
        if use_grl and grl_logits is not None:
            g_loss = domain_loss(grl_logits, pop_label)
            g_acc = domain_accuracy(grl_logits, pop_label)
            loss = loss + delta * g_loss
            total_grl_loss += g_loss.item()
            total_grl_acc += g_acc

        # ------------------------------------------------------------------
        # MMD on z_task — aligns source/target marginals (optional)
        # ------------------------------------------------------------------
        if use_mmd:
            ceu_z_task = mu[pop_label == 0, : model.task_dim]
            yri_z_task = mu[pop_label == 1, : model.task_dim]
            m_loss = mmd_loss(ceu_z_task, yri_z_task)
            loss = loss + delta * m_loss
            total_mmd_loss += m_loss.item()

        # ------------------------------------------------------------------
        # Cooperative domain classifier on z_domain — pulls ancestry info in
        # ------------------------------------------------------------------
        if use_domain_clf and domain_logits is not None:
            d_loss = domain_loss(domain_logits, pop_label)
            d_acc = domain_accuracy(domain_logits, pop_label)
            loss = loss + lam_domain * d_loss
            total_domain_clf_loss += d_loss.item()
            total_domain_acc += d_acc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon_unmasked += recon_u_val.item()
        total_recon_masked += recon_m_val.item()
        total_kl_loss += kl_val.item()
        total_phenotype_loss += pheno_val.item()

    n = len(dataloader)
    return (
        total_loss / n,
        total_recon_unmasked / n,
        total_recon_masked / n,
        total_kl_loss / n,
        total_phenotype_loss / n,
        total_grl_loss / n,
        total_grl_acc / n,
        total_domain_clf_loss / n,
        total_domain_acc / n,
        total_mmd_loss / n,
        total_z_task_var / n,
        total_z_domain_var / n if has_z_domain else None,
        batch_ceu_fracs,
    )


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Evaluate on an eval-style loader (5-tuple batches).
    Returns 5 floats: total_loss, recon_unmasked, recon_masked, kl, pheno_loss.
    """
    model.eval()

    total_loss = 0.0
    total_recon_unmasked = 0.0
    total_recon_masked = 0.0
    total_kl_loss = 0.0
    total_phenotype_loss = 0.0

    for input_x, x, pheno, mask, pop_label in dataloader:
        input_x = input_x.to(device)
        x = x.to(device)
        pheno = pheno.to(device)
        mask = mask.to(device)
        pop_label = pop_label.to(device)

        out, mu, logvar, z, pheno_pred, _grl_logits, _domain_logits = model(input_x)
        targets = x.squeeze(1).long()

        recon_unmasked = recon_unmasked_loss(out, targets, mask)
        recon_masked_l = recon_masked_loss(out, targets, mask)
        kl = kl_loss(mu, logvar)
        pheno_l = phenotype_loss(pheno_pred, pheno)

        loss, recon_u_val, recon_m_val, kl_val, pheno_val = loss_fn(
            recon_unmasked=recon_unmasked,
            recon_masked=recon_masked_l,
            kl=kl,
            pheno_loss=pheno_l,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        total_loss += loss.item()
        total_recon_unmasked += recon_u_val.item()
        total_recon_masked += recon_m_val.item()
        total_kl_loss += kl_val.item()
        total_phenotype_loss += pheno_val.item()

    n = len(dataloader)
    return (
        total_loss / n,
        total_recon_unmasked / n,
        total_recon_masked / n,
        total_kl_loss / n,
        total_phenotype_loss / n,
    )


# =============================================================================
# Early stopping
# =============================================================================


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.num_bad_epochs = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
            return True, False
        else:
            self.num_bad_epochs += 1
            return False, self.num_bad_epochs >= self.patience
