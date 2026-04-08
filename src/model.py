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
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


# =============================================================================
# ConvVAE with split latent space (shared + private)
# =============================================================================

class ConvVAE(nn.Module):
    def __init__(
        self,
        input_length,
        in_channels,
        hidden_channels,
        kernel_size,
        stride,
        padding,
        latent_dim_shared,       # domain-invariant subspace
        latent_dim_private,      # population-specific subspace
        num_classes=3,
        use_batchnorm=False,
        activation="elu",
        pheno_dim=1,
        pheno_hidden_dim=None,
        use_grl: bool = False,
        grl_hidden_dim=None,     # None → single linear domain classifier
        num_domains: int = 2,
    ):
        super().__init__()

        self.input_length       = input_length
        self.in_channels        = in_channels
        self.hidden_channels    = hidden_channels
        self.kernel_size        = kernel_size
        self.stride             = stride
        self.padding            = padding
        self.latent_dim_shared  = latent_dim_shared
        self.latent_dim_private = latent_dim_private
        self.latent_dim         = latent_dim_shared + latent_dim_private  # total
        self.num_classes        = num_classes
        self.use_batchnorm      = use_batchnorm
        self.activation         = activation
        self.use_grl            = use_grl
        self.encoder_lengths    = [input_length]

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        enc_layers = []
        current_in     = in_channels
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
        self.final_length   = current_length
        self.flat_dim       = self.final_channels * self.final_length

        # ------------------------------------------------------------------
        # Split bottleneck
        # Four projection heads instead of two
        # ------------------------------------------------------------------
        self.fc_mu_shared      = nn.Linear(self.flat_dim, latent_dim_shared)
        self.fc_logvar_shared  = nn.Linear(self.flat_dim, latent_dim_shared)
        self.fc_mu_private     = nn.Linear(self.flat_dim, latent_dim_private)
        self.fc_logvar_private = nn.Linear(self.flat_dim, latent_dim_private)

        # Decoder input = cat([z_shared, z_private])
        self.fc_decode = nn.Linear(latent_dim_shared + latent_dim_private, self.flat_dim)

        # ------------------------------------------------------------------
        # Phenotype head — mu_shared only
        # ------------------------------------------------------------------
        if pheno_hidden_dim is None:
            self.pheno_head = nn.Linear(latent_dim_shared, pheno_dim)
        else:
            self.pheno_head = nn.Sequential(
                nn.Linear(latent_dim_shared, pheno_hidden_dim),
                self._get_activation(),
                nn.Linear(pheno_hidden_dim, pheno_dim),
            )

        # ------------------------------------------------------------------
        # Domain classifier — mu_shared only, behind GRL
        # ------------------------------------------------------------------
        if use_grl:
            self.grl = GradientReversalLayer(lambda_=1.0)
            if grl_hidden_dim is None:
                self.domain_classifier = nn.Linear(latent_dim_shared, num_domains)
            else:
                self.domain_classifier = nn.Sequential(
                    nn.Linear(latent_dim_shared, grl_hidden_dim),
                    self._get_activation(),
                    nn.Linear(grl_hidden_dim, num_domains),
                )
        else:
            self.grl               = None
            self.domain_classifier = None

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        dec_layers       = []
        decoder_channels = list(reversed(hidden_channels))
        target_lengths   = list(reversed(self.encoder_lengths[:-1]))
        current_in       = decoder_channels[0]
        current_length   = self.encoder_lengths[-1]

        for i, target_length in enumerate(target_lengths):
            is_last     = i == len(target_lengths) - 1
            current_out = decoder_channels[i + 1] if not is_last else num_classes

            base_length    = (current_length - 1) * stride - 2 * padding + kernel_size
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

            current_in     = current_out
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def set_grl_lambda(self, lambda_: float):
        if self.grl is not None:
            self.grl.set_lambda(lambda_)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, verbose=False):
        """
        Returns
        -------
        out            : (B, num_classes, L)
        mu_shared      : (B, latent_dim_shared)
        logvar_shared  : (B, latent_dim_shared)
        mu_private     : (B, latent_dim_private)
        logvar_private : (B, latent_dim_private)
        z_shared       : (B, latent_dim_shared)
        z_private      : (B, latent_dim_private)
        pheno_pred     : (B, 1)
        domain_logits  : (B, num_domains) or None
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

        # Split bottleneck
        mu_shared      = self.fc_mu_shared(h_flat)
        logvar_shared  = torch.clamp(self.fc_logvar_shared(h_flat),  min=-6.0, max=6.0)
        mu_private     = self.fc_mu_private(h_flat)
        logvar_private = torch.clamp(self.fc_logvar_private(h_flat), min=-6.0, max=6.0)

        z_shared  = self.reparameterize(mu_shared,  logvar_shared)
        z_private = self.reparameterize(mu_private, logvar_private)

        # Phenotype: shared subspace only
        pheno_pred = self.pheno_head(mu_shared)

        # Domain classifier: shared subspace only, behind GRL
        if self.use_grl:
            domain_logits = self.domain_classifier(self.grl(mu_shared))
        else:
            domain_logits = None

        # Decoder: full concatenated latent
        z     = torch.cat([z_shared, z_private], dim=1)
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(x.size(0), self.final_channels, self.final_length)

        out = h_dec
        for i, layer in enumerate(self.decoder):
            out = layer(out)
            if verbose:
                print(f"Decoder layer {i:02d} ({layer.__class__.__name__}): {out.shape}")

        expected_shape = (x.size(0), self.num_classes, x.size(2))
        if out.shape != expected_shape:
            raise ValueError(
                f"Decoder output shape {out.shape} does not match "
                f"expected {expected_shape}"
            )

        return (
            out,
            mu_shared, logvar_shared,
            mu_private, logvar_private,
            z_shared, z_private,
            pheno_pred, domain_logits,
        )