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

    The latent vector mu (shape: latent_dim) is partitioned into two halves:

        mu = [ z_shared (shared_dim) | z_pop (latent_dim - shared_dim) ]

        z_shared: domain-invariant subspace.
                  Used by the phenotype head and the GRL domain classifier.
                  The GRL pushes z_shared to be indistinguishable across domains,
                  so phenotype prediction is forced to rely only on features
                  that generalise across populations.

        z_pop:    population-specific subspace.
                  Free to encode LD structure, allele frequency differences,
                  and other population-specific signals needed for reconstruction.
                  Not seen by the phenotype head or the GRL.

        decoder:  receives the full z (reparameterized from the full mu/logvar),
                  so reconstruction can use both subspaces.

    Parameters
    ----------
    shared_dim : int or None
        Number of dimensions in z_shared. Defaults to latent_dim // 2.
        Must be <= latent_dim. Set to latent_dim to recover the original
        (unsplit) behaviour.
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
        # --- domain adaptation ---
        use_grl: bool = False,
        grl_hidden_dim: int = 256,
        num_domains: int = 2,
        # --- split latent space ---
        shared_dim: int = None,
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

        # shared_dim defaults to half the latent space
        self.shared_dim = shared_dim if shared_dim is not None else latent_dim // 2
        assert (
            self.shared_dim <= latent_dim
        ), f"shared_dim ({self.shared_dim}) must be <= latent_dim ({latent_dim})"

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
        # Phenotype head  (z_shared only)
        # ------------------------------------------------------------------
        if pheno_hidden_dim is None:
            self.pheno_head = nn.Linear(self.shared_dim, pheno_dim)
        else:
            self.pheno_head = nn.Sequential(
                nn.Linear(self.shared_dim, pheno_hidden_dim),
                self._get_activation(),
                nn.Linear(pheno_hidden_dim, pheno_dim),
            )

        # ------------------------------------------------------------------
        # Domain classifier (z_shared only, through GRL)
        # ------------------------------------------------------------------
        if use_grl:
            self.grl = GradientReversalLayer(lambda_=1.0)
            if grl_hidden_dim is None:
                self.domain_classifier = nn.Linear(self.shared_dim, num_domains)
            else:
                self.domain_classifier = nn.Sequential(
                    nn.Linear(self.shared_dim, grl_hidden_dim),
                    self._get_activation(),
                    nn.Linear(grl_hidden_dim, num_domains),
                )
        else:
            self.grl = None
            self.domain_classifier = None

        # ------------------------------------------------------------------
        # Decoder  (full z — unchanged)
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
        """Update the GRL reversal strength. Call this each epoch from the training loop."""
        if self.grl is not None:
            self.grl.set_lambda(lambda_)

    def latent_stats(self, mu: torch.Tensor) -> dict:
        """
        Returns variance diagnostics for the two latent subspaces.
        Log these each epoch to check that z_pop doesn't collapse.

        Usage in training loop:
            stats = model.latent_stats(mu)
            log('z_shared_var', stats['z_shared_var'])
            log('z_pop_var',    stats['z_pop_var'])
        """
        z_shared = mu[:, : self.shared_dim]
        z_pop = mu[:, self.shared_dim :]
        z_pop_var = z_pop.var(dim=0).mean().item() if z_pop.shape[1] > 0 else None
        return {
            "z_shared_var": z_shared.var(dim=0).mean().item(),
            "z_pop_var": z_pop_var,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, verbose=False):
        """
        Returns
        -------
        out          : (B, num_classes, L)  decoder logits
        mu           : (B, latent_dim)      full posterior mean
        logvar       : (B, latent_dim)      full posterior log-variance
        z            : (B, latent_dim)      reparameterized sample (used by decoder)
        pheno_pred   : (B, pheno_dim)       predicted from z_shared = mu[:, :shared_dim]
        domain_logits: (B, num_domains)     predicted from GRL(z_shared), or None
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

        # ── Split mu into the two subspaces ───────────────────────────────
        # z_shared: domain-invariant, used for phenotype + GRL
        # z_pop:    population-specific, only used implicitly via full z in decoder
        z_shared = mu[:, : self.shared_dim]  # (B, shared_dim)
        # z_pop  = mu[:, self.shared_dim:]   # (B, latent_dim - shared_dim)
        #   not needed explicitly here — decoder uses full z

        # Phenotype prediction from z_shared (deterministic path)
        pheno_pred = self.pheno_head(z_shared)

        # Domain classification through reversed gradients on z_shared only
        if self.use_grl:
            z_shared_rev = self.grl(z_shared)
            domain_logits = self.domain_classifier(z_shared_rev)
        else:
            domain_logits = None

        # Decoder — full z, unchanged
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
                f"expected logits shape {expected_shape}"
            )

        return out, mu, logvar, z, pheno_pred, domain_logits
