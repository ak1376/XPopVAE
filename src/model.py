import torch
import torch.nn as nn
from torch.autograd import Function

# =============================================================================
# Gradient Reversal
# =============================================================================


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """Scales gradients by -lambda_ during the backward pass."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


# =============================================================================
# ConvVAE
# =============================================================================


class ConvVAE(nn.Module):
    """
    Convolutional VAE with a split latent space for domain adaptation.

    The latent vector mu (shape: latent_dim) is partitioned into two parts:

        mu = [ z_task (task_dim) | z_domain (latent_dim - task_dim) ]

        z_task:   task-relevant subspace.
                  Used by the phenotype head, MMD alignment, and optionally
                  an adversarial GRL classifier.

        z_domain: domain-specific subspace.
                  Optionally used by a cooperative domain classifier (no
                  gradient reversal) that pulls ancestry info INTO z_domain.

        decoder:  receives the full reparameterized z so reconstruction
                  leverages both subspaces.

    Domain adaptation flags (all independent, any combination works):
        use_grl=True        adversarial GRL classifier on z_task
        use_domain_clf=True cooperative classifier on z_domain (no reversal)
        use_mmd             handled in train.py, not in the model

    Forward returns a 7-tuple:
        out, mu, logvar, z, pheno_pred, grl_logits, domain_logits
        grl_logits    is None when use_grl=False
        domain_logits is None when use_domain_clf=False

    Parameters
    ----------
    task_dim : int or None
        Dimensions in z_task. Defaults to latent_dim // 2.
    """

    def __init__(
        self,
        input_length,
        in_channels,
        hidden_channels,
        kernel_size,
        stride,
        padding,
        latent_dim,
        num_classes=3,
        use_batchnorm=False,
        activation="elu",
        pheno_dim=1,
        pheno_hidden_dim=None,
        # --- split latent space ---
        task_dim: int = None,
        # --- adversarial GRL on z_task ---
        use_grl: bool = False,
        grl_hidden_dim: int = 256,
        # --- cooperative domain clf on z_domain ---
        use_domain_clf: bool = False,
        domain_clf_hidden_dim: int = 256,
        # --- shared ---
        num_domains: int = 2,
    ):
        super().__init__()

        self.input_length = input_length
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.use_grl = use_grl
        self.encoder_lengths = [input_length]

        self.task_dim = task_dim if task_dim is not None else latent_dim // 2
        assert self.task_dim <= latent_dim, (
            f"task_dim ({self.task_dim}) must be <= latent_dim ({latent_dim})"
        )
        domain_dim = latent_dim - self.task_dim

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        enc_layers = []
        current_in = in_channels
        current_length = input_length

        for current_out in hidden_channels:
            enc_layers.append(
                nn.Conv1d(
                    in_channels=current_in,
                    out_channels=current_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            if use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(current_out))
            enc_layers.append(self._get_activation())
            current_length = self.compute_conv1d_output_length(
                current_length, kernel_size, stride, padding
            )
            self.encoder_lengths.append(current_length)
            current_in = current_out

        self.encoder = nn.Sequential(*enc_layers)

        self.final_channels = hidden_channels[-1]
        self.final_length = current_length
        self.flat_dim = self.final_channels * self.final_length

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # ------------------------------------------------------------------
        # Phenotype head  (z_task only)
        # ------------------------------------------------------------------
        if pheno_hidden_dim is None:
            self.pheno_head = nn.Linear(self.task_dim, pheno_dim)
        else:
            self.pheno_head = nn.Sequential(
                nn.Linear(self.task_dim, pheno_hidden_dim),
                self._get_activation(),
                nn.Linear(pheno_hidden_dim, pheno_dim),
            )

        # ------------------------------------------------------------------
        # Adversarial GRL domain classifier (z_task only)
        # Reversed gradients push z_task to be domain-invariant.
        # ------------------------------------------------------------------
        if use_grl:
            self.grl = GradientReversalLayer(lambda_=1.0)
            if grl_hidden_dim is None:
                self.grl_classifier = nn.Linear(self.task_dim, num_domains)
            else:
                self.grl_classifier = nn.Sequential(
                    nn.Linear(self.task_dim, grl_hidden_dim),
                    self._get_activation(),
                    nn.Linear(grl_hidden_dim, num_domains),
                )
        else:
            self.grl = None
            self.grl_classifier = None

        # ------------------------------------------------------------------
        # Cooperative domain classifier (z_domain only, no gradient reversal)
        # Normal gradients pull ancestry information INTO z_domain.
        # ------------------------------------------------------------------
        if use_domain_clf and domain_dim > 0:
            if domain_clf_hidden_dim is None:
                self.domain_classifier = nn.Linear(domain_dim, num_domains)
            else:
                self.domain_classifier = nn.Sequential(
                    nn.Linear(domain_dim, domain_clf_hidden_dim),
                    self._get_activation(),
                    nn.Linear(domain_clf_hidden_dim, num_domains),
                )
        else:
            self.domain_classifier = None

        # ------------------------------------------------------------------
        # Decoder  (full z — uses both z_task and z_domain)
        # ------------------------------------------------------------------
        dec_layers = []
        decoder_channels = list(reversed(hidden_channels))
        target_lengths = list(reversed(self.encoder_lengths[:-1]))
        current_in = decoder_channels[0]
        current_length = self.encoder_lengths[-1]

        for i, target_length in enumerate(target_lengths):
            is_last = i == len(target_lengths) - 1
            current_out = decoder_channels[i + 1] if not is_last else num_classes

            base_length = (current_length - 1) * stride - 2 * padding + kernel_size
            output_padding = target_length - base_length

            if output_padding not in [0, 1]:
                raise ValueError(
                    f"Invalid output_padding={output_padding} for "
                    f"{current_length} -> {target_length}"
                )

            dec_layers.append(
                nn.ConvTranspose1d(
                    in_channels=current_in,
                    out_channels=current_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            if not is_last:
                if use_batchnorm:
                    dec_layers.append(nn.BatchNorm1d(current_out))
                dec_layers.append(self._get_activation())

            current_in = current_out
            current_length = target_length

        self.decoder = nn.Sequential(*dec_layers)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_activation(self):
        if self.activation == "elu":
            return nn.ELU()
        if self.activation == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation: {self.activation}")

    @staticmethod
    def compute_conv1d_output_length(length, kernel_size, stride, padding):
        return ((length + 2 * padding - kernel_size) // stride) + 1

    @staticmethod
    def compute_transpose_output_length(
        length, kernel_size, stride, padding, output_padding
    ):
        return (length - 1) * stride - 2 * padding + kernel_size + output_padding

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def set_grl_lambda(self, lambda_: float):
        """Update the GRL reversal strength. Call each epoch from the training loop."""
        if self.grl is not None:
            self.grl.set_lambda(lambda_)

    def latent_stats(self, mu: torch.Tensor) -> dict:
        """Variance diagnostics for the two latent subspaces."""
        z_task = mu[:, : self.task_dim]
        z_domain = mu[:, self.task_dim :]
        z_domain_var = (
            z_domain.var(dim=0).mean().item() if z_domain.shape[1] > 0 else None
        )
        return {
            "z_task_var": z_task.var(dim=0).mean().item(),
            "z_domain_var": z_domain_var,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, verbose=False):
        """
        Returns a 7-tuple:
            out          (B, num_classes, L)  decoder logits
            mu           (B, latent_dim)      posterior mean
            logvar       (B, latent_dim)      posterior log-variance
            z            (B, latent_dim)      reparameterized sample
            pheno_pred   (B, pheno_dim)       predicted from z_task
            grl_logits   (B, num_domains) or None  adversarial clf on z_task
            domain_logits(B, num_domains) or None  cooperative clf on z_domain
        """
        if verbose:
            print(f"Input: {x.shape}")

        h = x
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if verbose:
                print(f"Encoder layer {i:02d} ({layer.__class__.__name__}): {h.shape}")

        h_flat = torch.flatten(h, start_dim=1)
        if verbose:
            print(f"Flattened: {h_flat.shape}")

        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)
        z = self.reparameterize(mu, logvar)

        z_task = mu[:, : self.task_dim]    # (B, task_dim)
        z_domain = mu[:, self.task_dim :]  # (B, latent_dim - task_dim)

        pheno_pred = self.pheno_head(z_task)

        grl_logits = (
            self.grl_classifier(self.grl(z_task))
            if self.grl is not None
            else None
        )

        domain_logits = (
            self.domain_classifier(z_domain)
            if self.domain_classifier is not None
            else None
        )

        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(x.size(0), self.final_channels, self.final_length)
        out = h_dec
        for i, layer in enumerate(self.decoder):
            out = layer(out)
            if verbose:
                print(
                    f"Decoder layer {i:02d} ({layer.__class__.__name__}): {out.shape}"
                )

        expected_shape = (x.size(0), self.num_classes, x.size(2))
        if out.shape != expected_shape:
            raise ValueError(
                f"Decoder output shape {out.shape} does not match "
                f"expected {expected_shape}"
            )

        return out, mu, logvar, z, pheno_pred, grl_logits, domain_logits
