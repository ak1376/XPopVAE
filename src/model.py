import torch
import torch.nn as nn


class ConvVAE(nn.Module):
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
        num_domains=2,
        use_batchnorm=False,
        activation="elu",
        pheno_dim=1,
        pheno_hidden_dim=None,
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
        self.num_domains = num_domains
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.encoder_lengths = [input_length]

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
                current_length,
                kernel_size,
                stride,
                padding,
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

        # phenotype head: predict phenotype from mu
        if pheno_hidden_dim is None:
            self.pheno_head = nn.Linear(latent_dim, pheno_dim)
        else:
            self.pheno_head = nn.Sequential(
                nn.Linear(latent_dim, pheno_hidden_dim),
                self._get_activation(),
                nn.Linear(pheno_hidden_dim, pheno_dim),
            )

        # domain head: predict population membership from mu
        # outputs raw logits for num_domains classes (CEU=0, YRI=1)
        # GRL is applied externally in train.py before this head is called
        self.domain_head = nn.Linear(latent_dim, num_domains)

        dec_layers = []
        decoder_channels = list(reversed(hidden_channels))
        target_lengths = list(reversed(self.encoder_lengths[:-1]))
        current_in = decoder_channels[0]
        current_length = self.encoder_lengths[-1]

        for i, target_length in enumerate(target_lengths):
            is_last = i == len(target_lengths) - 1

            if not is_last:
                current_out = decoder_channels[i + 1]
            else:
                current_out = num_classes

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

    def forward(self, x, verbose=False):
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

        # phenotype prediction from mu (no GRL — pheno head is not adversarial)
        pheno_pred = self.pheno_head(mu)

        # domain logits returned from mu directly — GRL applied externally
        # in train.py before passing mu to domain_head during training
        domain_logits = self.domain_head(mu)

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
                f"Decoder output shape {out.shape} does not match expected logits shape {expected_shape}"
            )

        return out, mu, logvar, z, pheno_pred, domain_logits
