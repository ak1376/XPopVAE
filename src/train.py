import torch
import torch.nn.functional as F
from src.loss import recon_unmasked_loss, recon_masked_loss, kl_loss, phenotype_loss


# ------------------------------------------------------------------
# Gradient Reversal Layer
# Applied externally to mu before passing to domain_head.
# During the forward pass it is an identity; during the backward pass
# it multiplies gradients by -lambda_grl, pushing the encoder to
# produce population-invariant representations.
# ------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.save_for_backward(torch.tensor(lambda_grl))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_grl,) = ctx.saved_tensors
        return -lambda_grl.item() * grad_output, None


def grad_reverse(x, lambda_grl):
    return GradientReversalFunction.apply(x, lambda_grl)


# ------------------------------------------------------------------
# training
# ------------------------------------------------------------------
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
    lambda_grl=1.0,
):
    model.train()

    total_loss = 0.0
    total_recon_unmasked = 0.0
    total_recon_masked = 0.0
    total_kl_loss = 0.0
    total_phenotype_loss = 0.0
    total_domain_loss = 0.0

    for x, pheno, pop_label in dataloader:
        x = x.to(device)
        pheno = pheno.to(device)
        pop_label = pop_label.to(device)

        # dynamic masking: fresh every batch, every epoch
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

        logits, mu, logvar, z, pheno_pred, _ = model(input_x)
        targets = x.squeeze(1).long()

        recon_unmasked = recon_unmasked_loss(logits, targets, mask)
        recon_masked = recon_masked_loss(logits, targets, mask)
        kl = kl_loss(mu, logvar)

        # Phenotype supervision on discovery (pop_label == 0) only.
        # Slice to discovery individuals so the MSE mean denominator
        # reflects only supervised samples.
        disc_idx = (pop_label == 0).nonzero(as_tuple=True)[0]
        if disc_idx.numel() > 0:
            pheno_loss_val = phenotype_loss(
                pheno_pred[disc_idx],
                pheno[disc_idx],
            )
        else:
            # edge case: batch contains only target individuals
            pheno_loss_val = torch.tensor(0.0, device=device, requires_grad=True)

        # Domain adversarial loss via GRL.
        # Apply gradient reversal to mu, then pass through domain_head.
        # The reversed gradients push the encoder toward population-invariant
        # representations while the domain head itself learns to discriminate.
        mu_reversed = grad_reverse(mu, lambda_grl)
        domain_logits = model.domain_head(mu_reversed)
        domain_loss = F.cross_entropy(domain_logits, pop_label)

        # Total loss: VAE terms + domain adversarial term
        vae_loss_val, recon_unmasked_val, recon_masked_val, kl_val, pheno_val = loss_fn(
            recon_unmasked=recon_unmasked,
            recon_masked=recon_masked,
            kl=kl,
            pheno_loss=pheno_loss_val,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        loss = vae_loss_val + domain_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon_unmasked += recon_unmasked_val.item()
        total_recon_masked += recon_masked_val.item()
        total_kl_loss += kl_val.item()
        total_phenotype_loss += pheno_val.item()
        total_domain_loss += domain_loss.item()

    n_batches = len(dataloader)
    return (
        total_loss / n_batches,
        total_recon_unmasked / n_batches,
        total_recon_masked / n_batches,
        total_kl_loss / n_batches,
        total_phenotype_loss / n_batches,
        total_domain_loss / n_batches,
    )


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn, alpha=1.0, beta=1.0, gamma=1.0):
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

        logits, mu, logvar, z, pheno_pred, _ = model(input_x)
        targets = x.squeeze(1).long()

        recon_unmasked = recon_unmasked_loss(logits, targets, mask)
        recon_masked = recon_masked_loss(logits, targets, mask)
        kl = kl_loss(mu, logvar)
        pheno_loss_val = phenotype_loss(pheno_pred, pheno)

        loss, recon_unmasked_val, recon_masked_val, kl_val, pheno_val = loss_fn(
            recon_unmasked=recon_unmasked,
            recon_masked=recon_masked,
            kl=kl,
            pheno_loss=pheno_loss_val,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        total_loss += loss.item()
        total_recon_unmasked += recon_unmasked_val.item()
        total_recon_masked += recon_masked_val.item()
        total_kl_loss += kl_val.item()
        total_phenotype_loss += pheno_val.item()

    n_batches = len(dataloader)
    return (
        total_loss / n_batches,
        total_recon_unmasked / n_batches,
        total_recon_masked / n_batches,
        total_kl_loss / n_batches,
        total_phenotype_loss / n_batches,
    )


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
            should_stop = self.num_bad_epochs >= self.patience
            return False, should_stop
