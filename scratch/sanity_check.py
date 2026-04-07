import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/sietch_colab/akapoor/XPopVAE")
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvVAE

# load checkpoint
checkpoint = torch.load(
    "/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/vae/default/vae_outputs/checkpoints/best_model.pt",
    map_location="cpu"
)
vae_config = checkpoint["vae_config"]
input_length = checkpoint["input_length"]

model = ConvVAE(
    input_length=input_length,
    in_channels=1,
    hidden_channels=vae_config["model"]["hidden_channels"],
    kernel_size=int(vae_config["model"]["kernel_size"]),
    stride=int(vae_config["model"]["stride"]),
    padding=int(vae_config["model"]["padding"]),
    latent_dim=int(vae_config["model"]["latent_dim"]),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# load real genotypes for comparison
val_geno = torch.tensor(
    np.load("/sietch_colab/akapoor/XPopVAE/experiments/IM_symmetric/processed_data/0/rep0/discovery_val.npy"),
    dtype=torch.float32
)  # [N, L]

N = val_geno.shape[0]
latent_dim = int(vae_config["model"]["latent_dim"])

with torch.no_grad():
    # --- real encoder output ---
    z_real, _, _, _, _ = model(val_geno.unsqueeze(1))
    # that gives logits; let's get mu directly
    h = val_geno.unsqueeze(1)
    for layer in model.encoder:
        h = layer(h)
    h_flat = torch.flatten(h, start_dim=1)
    mu_real = model.fc_mu(h_flat)

    # --- random z from prior ---
    z_random = torch.randn(N, latent_dim)

    # decode both
    def decode(z):
        h_dec = model.fc_decode(z)
        h_dec = h_dec.view(N, model.final_channels, model.final_length)
        return model.decoder(h_dec)  # [N, 3, L]

    logits_real   = decode(mu_real)
    logits_random = decode(z_random)

    preds_real   = logits_real.argmax(dim=1)    # [N, L]
    preds_random = logits_random.argmax(dim=1)  # [N, L]

targets = val_geno.long()  # [N, L]

from sklearn.metrics import balanced_accuracy_score

ba_real   = balanced_accuracy_score(targets.numpy().flatten(), preds_real.numpy().flatten())
ba_random = balanced_accuracy_score(targets.numpy().flatten(), preds_random.numpy().flatten())

print(f"Balanced accuracy — real encoder mu:  {ba_real:.4f}")
print(f"Balanced accuracy — random z ~ N(0,1): {ba_random:.4f}")