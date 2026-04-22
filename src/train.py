import torch
from src.loss import (
    recon_unmasked_loss,
    recon_masked_loss,
    kl_loss,
    phenotype_loss,
    domain_loss,
)

# =============================================================================
# GRL lambda schedule
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

    Starts near 0, ramps smoothly to lambda_max.
    """
    import math

    p = current_epoch / max(total_epochs, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# =============================================================================
# Domain-accuracy helper (no grad, for adaptive weighting / logging)
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
    # --- GRL ---
    use_grl: bool = False,
    delta: float = 1.0,
):
    """
    Train for one epoch.

    GRL domain loss
    ---------------
    When use_grl=True the model returns domain_logits, and we add:

        effective_delta * domain_loss

    to the total loss, where:

        effective_delta = delta * domain_acc

    The adaptive scaling means the encoder is pushed hardest when the
    domain classifier is most accurate (latent space still domain-
    separable). As the encoder fools the classifier, the signal naturally
    shrinks toward zero.

    The GRL layer's reversal strength (lambda) is set externally via
    model.set_grl_lambda() before each epoch — use compute_grl_lambda()
    in run_vae.py for the Ganin et al. warmup schedule.

    Latent subspace diagnostics
    ---------------------------
    Each epoch we track the average per-dimension variance of z_shared
    and z_pop across all batches. If z_pop_var trends to zero, the encoder
    is collapsing the population-specific subspace into z_shared, meaning
    the split is not working as intended.

    Returns
    -------
    Tuple of 9 floats:
        total_loss, recon_unmasked, recon_masked, kl, pheno,
        domain_ce, domain_acc, z_shared_var, z_pop_var
    """
    model.train()

    total_loss = 0.0
    total_recon_unmasked = 0.0
    total_recon_masked = 0.0
    total_kl_loss = 0.0
    total_phenotype_loss = 0.0
    total_domain_loss = 0.0
    total_domain_acc = 0.0
    total_z_shared_var = 0.0
    has_z_pop = model.shared_dim < model.latent_dim
    total_z_pop_var = 0.0 if has_z_pop else None

    for x, pheno, pop_label in dataloader:
        x = x.to(device)
        pheno = pheno.to(device)
        pop_label = pop_label.to(device)

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

        out, mu, logvar, z, pheno_pred, domain_logits = model(input_x)
        targets = x.squeeze(1).long()

        # latent subspace diagnostics — logged every batch, averaged over epoch
        stats = model.latent_stats(mu)
        total_z_shared_var += stats["z_shared_var"]
        if has_z_pop:
            total_z_pop_var += stats["z_pop_var"]

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
        # GRL domain loss
        # ------------------------------------------------------------------
        if use_grl and domain_logits is not None:
            d_loss = domain_loss(domain_logits, pop_label)
            d_acc = domain_accuracy(domain_logits, pop_label)

            loss = loss + delta * d_loss

            total_domain_loss += d_loss.item()
            total_domain_acc += d_acc
        else:
            total_domain_loss += 0.0
            total_domain_acc += 0.0

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
        total_domain_loss / n,
        total_domain_acc / n,
        total_z_shared_var / n,
        total_z_pop_var / n if has_z_pop else None,
    )


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Evaluate on an eval-style loader (5-tuple batches).

    Returns 5 floats: total_loss, recon_unmasked, recon_masked, kl, pheno_loss.
    Domain logits are ignored — evaluation doesn't update weights.
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

        out, mu, logvar, z, pheno_pred, _domain_logits = model(input_x)
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
