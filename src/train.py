import torch
from src.loss import recon_unmasked_loss, recon_masked_loss, kl_loss, phenotype_loss


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    loss_fn,
    masker,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    lambda_ortho=1e-3,
):
    model.train()

    total_loss           = 0.0
    total_recon_unmasked = 0.0
    total_recon_masked   = 0.0
    total_kl_loss        = 0.0
    total_phenotype_loss = 0.0
    total_ortho_loss     = 0.0

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

        logits, mu, logvar, z, pheno_pred = model(input_x)
        targets = x.squeeze(1).long()

        recon_unmasked = recon_unmasked_loss(logits, targets, mask)
        recon_masked   = recon_masked_loss(logits, targets, mask)
        kl             = kl_loss(mu, logvar)

        # only compute phenotype loss on CEU (pop_label == 0)
        ceu_mask = (pop_label == 0)
        if ceu_mask.any():
            pheno_loss_val = phenotype_loss(pheno_pred[ceu_mask], pheno[ceu_mask])
        else:
            pheno_loss_val = torch.tensor(0.0, device=device)

        loss, recon_unmasked_val, recon_masked_val, kl_val, pheno_val, ortho_val = loss_fn(
            recon_unmasked=recon_unmasked,
            recon_masked=recon_masked,
            kl=kl,
            pheno_loss=pheno_loss_val,
            z_recon=z[:, :model.recon_latent_dim],
            z_pheno=z[:, model.recon_latent_dim:],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_ortho=lambda_ortho,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss           += loss.item()
        total_recon_unmasked += recon_unmasked_val.item()
        total_recon_masked   += recon_masked_val.item()
        total_kl_loss        += kl_val.item()
        total_phenotype_loss += pheno_val.item()
        total_ortho_loss     += ortho_val.item()

    n = len(dataloader)
    return (
        total_loss           / n,
        total_recon_unmasked / n,
        total_recon_masked   / n,
        total_kl_loss        / n,
        total_phenotype_loss / n,
        total_ortho_loss     / n,
    )


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    loss_fn,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    lambda_ortho=1e-3,
):
    model.eval()

    total_loss           = 0.0
    total_recon_unmasked = 0.0
    total_recon_masked   = 0.0
    total_kl_loss        = 0.0
    total_phenotype_loss = 0.0
    total_ortho_loss     = 0.0

    for input_x, x, pheno, mask, pop_label in dataloader:
        input_x   = input_x.to(device)
        x         = x.to(device)
        pheno     = pheno.to(device)
        mask      = mask.to(device)
        pop_label = pop_label.to(device)

        logits, mu, logvar, z, pheno_pred = model(input_x)
        targets = x.squeeze(1).long()

        recon_unmasked = recon_unmasked_loss(logits, targets, mask)
        recon_masked   = recon_masked_loss(logits, targets, mask)
        kl             = kl_loss(mu, logvar)

        # only compute phenotype loss on CEU (pop_label == 0)
        # val loader is all CEU so this is a no-op there, but safe for mixed loaders
        ceu_mask = (pop_label == 0)
        if ceu_mask.any():
            pheno_loss_val = phenotype_loss(pheno_pred[ceu_mask], pheno[ceu_mask])
        else:
            pheno_loss_val = torch.tensor(0.0, device=device)

        loss, recon_unmasked_val, recon_masked_val, kl_val, pheno_val, ortho_val = loss_fn(
            recon_unmasked=recon_unmasked,
            recon_masked=recon_masked,
            kl=kl,
            pheno_loss=pheno_loss_val,
            z_recon=z[:, :model.recon_latent_dim],
            z_pheno=z[:, model.recon_latent_dim:],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_ortho=lambda_ortho,
        )

        total_loss           += loss.item()
        total_recon_unmasked += recon_unmasked_val.item()
        total_recon_masked   += recon_masked_val.item()
        total_kl_loss        += kl_val.item()
        total_phenotype_loss += pheno_val.item()
        total_ortho_loss     += ortho_val.item()

    n = len(dataloader)
    return (
        total_loss           / n,
        total_recon_unmasked / n,
        total_recon_masked   / n,
        total_kl_loss        / n,
        total_phenotype_loss / n,
        total_ortho_loss     / n,
    )


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