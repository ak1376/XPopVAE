#!/usr/bin/env python3
"""
Analyze z_pheno distribution for CEU vs YRI.
Runs PCA on the 16-dim pheno latent and plots marginal distributions.

Usage:
    python analyze_z_pheno.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# paths — edit if needed
# ------------------------------------------------------------------
PROJECT_ROOT = Path("/sietch_colab/akapoor/XPopVAE")
CHECKPOINT   = PROJECT_ROOT / "experiments/IM_symmetric/vae/default/vae_outputs/checkpoints/best_model.pt"
CEU_VAL      = PROJECT_ROOT / "experiments/IM_symmetric/processed_data/0/rep0/validation_discovery.npy"
YRI_TEST     = PROJECT_ROOT / "experiments/IM_symmetric/processed_data/0/rep0/target.npy"
OUTPUT_DIR   = PROJECT_ROOT / "experiments/IM_symmetric/vae/default/vae_outputs/z_pheno_analysis"

# ------------------------------------------------------------------
# model config — must match the checkpoint
# ------------------------------------------------------------------
LATENT_DIM      = 512
PHENO_LATENT_DIM = 16
HIDDEN_CHANNELS  = [32, 64, 128, 256, 256, 256]
KERNEL_SIZE      = 17
STRIDE           = 2
PADDING          = 8
PHENO_HIDDEN_DIM = 64

# ------------------------------------------------------------------
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ConvVAE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# load checkpoint
# ------------------------------------------------------------------
ckpt = torch.load(CHECKPOINT, map_location=device)
input_length = ckpt["input_length"]

model = ConvVAE(
    input_length=input_length,
    in_channels=1,
    hidden_channels=HIDDEN_CHANNELS,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    latent_dim=LATENT_DIM,
    use_batchnorm=False,
    activation="elu",
    pheno_dim=1,
    pheno_hidden_dim=PHENO_HIDDEN_DIM,
    pheno_latent_dim=PHENO_LATENT_DIM,
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
print(f"recon_latent_dim={model.recon_latent_dim}  pheno_latent_dim={model.pheno_latent_dim}")


# ------------------------------------------------------------------
# extract mu_pheno for a dataset
# ------------------------------------------------------------------
def extract_mu_pheno(npy_path: Path, batch_size: int = 256) -> np.ndarray:
    data = np.load(npy_path)
    x    = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
    all_mu_pheno = []

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(device)
            _, mu, _, _, _ = model(batch)
            # pheno slice is the last pheno_latent_dim dims of mu
            mu_pheno = mu[:, model.recon_latent_dim:].cpu().numpy()
            all_mu_pheno.append(mu_pheno)

    return np.concatenate(all_mu_pheno, axis=0)


print("Extracting CEU z_pheno...")
ceu_z = extract_mu_pheno(CEU_VAL)
print(f"  CEU z_pheno shape: {ceu_z.shape}")

print("Extracting YRI z_pheno...")
yri_z = extract_mu_pheno(YRI_TEST)
print(f"  YRI z_pheno shape: {yri_z.shape}")


# ------------------------------------------------------------------
# PCA on z_pheno (fit on CEU, project both)
# ------------------------------------------------------------------
pca = PCA(n_components=min(PHENO_LATENT_DIM, 8))
pca.fit(ceu_z)
ceu_pca = pca.transform(ceu_z)
yri_pca = pca.transform(yri_z)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PC1 vs PC2
ax = axes[0]
ax.scatter(ceu_pca[:, 0], ceu_pca[:, 1], alpha=0.3, s=8, label="CEU val", color="steelblue")
ax.scatter(yri_pca[:, 0], yri_pca[:, 1], alpha=0.3, s=8, label="YRI test", color="darkorange")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("z_pheno PCA — fit on CEU, both projected")
ax.legend()

# PC3 vs PC4 if available
ax = axes[1]
if pca.n_components_ >= 4:
    ax.scatter(ceu_pca[:, 2], ceu_pca[:, 3], alpha=0.3, s=8, label="CEU val", color="steelblue")
    ax.scatter(yri_pca[:, 2], yri_pca[:, 3], alpha=0.3, s=8, label="YRI test", color="darkorange")
    ax.set_xlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)")
    ax.set_ylabel(f"PC4 ({pca.explained_variance_ratio_[3]*100:.1f}% var)")
    ax.set_title("z_pheno PCA — PC3 vs PC4")
    ax.legend()
else:
    ax.axis("off")

plt.tight_layout()
pca_path = OUTPUT_DIR / "z_pheno_pca.png"
plt.savefig(pca_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved PCA plot: {pca_path}")


# ------------------------------------------------------------------
# marginal distributions — one panel per pheno dim
# ------------------------------------------------------------------
n_dims  = PHENO_LATENT_DIM
n_cols  = 4
n_rows  = int(np.ceil(n_dims / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes = axes.flatten()

for dim in range(n_dims):
    ax = axes[dim]
    ax.hist(ceu_z[:, dim], bins=50, alpha=0.5, density=True, label="CEU", color="steelblue")
    ax.hist(yri_z[:, dim], bins=50, alpha=0.5, density=True, label="YRI", color="darkorange")
    ax.set_title(f"z_pheno dim {dim}")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    if dim == 0:
        ax.legend(fontsize=8)

# hide unused panels
for i in range(n_dims, len(axes)):
    axes[i].axis("off")

plt.suptitle("z_pheno marginal distributions: CEU vs YRI", fontsize=13, y=1.01)
plt.tight_layout()
marginals_path = OUTPUT_DIR / "z_pheno_marginals.png"
plt.savefig(marginals_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved marginals plot: {marginals_path}")


# ------------------------------------------------------------------
# summary statistics — mean shift and std ratio per dim
# ------------------------------------------------------------------
ceu_mean = ceu_z.mean(axis=0)
yri_mean = yri_z.mean(axis=0)
ceu_std  = ceu_z.std(axis=0)
yri_std  = yri_z.std(axis=0)

mean_shift    = np.abs(yri_mean - ceu_mean)
std_ratio     = yri_std / (ceu_std + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(n_dims), mean_shift, color="steelblue")
axes[0].set_xlabel("z_pheno dimension")
axes[0].set_ylabel("|YRI mean - CEU mean|")
axes[0].set_title("Mean shift per z_pheno dimension")
axes[0].axhline(mean_shift.mean(), color="red", linestyle="--", label=f"avg={mean_shift.mean():.3f}")
axes[0].legend()

axes[1].bar(range(n_dims), std_ratio, color="darkorange")
axes[1].set_xlabel("z_pheno dimension")
axes[1].set_ylabel("YRI std / CEU std")
axes[1].set_title("Std ratio per z_pheno dimension")
axes[1].axhline(1.0, color="red", linestyle="--", label="ratio=1 (no shift)")
axes[1].legend()

plt.tight_layout()
shift_path = OUTPUT_DIR / "z_pheno_shift.png"
plt.savefig(shift_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved shift plot: {shift_path}")

print("\n--- Summary ---")
print(f"Mean absolute shift across dims: {mean_shift.mean():.4f}  (max: {mean_shift.max():.4f} at dim {mean_shift.argmax()})")
print(f"Std ratio across dims:           {std_ratio.mean():.4f}  (max: {std_ratio.max():.4f} at dim {std_ratio.argmax()})")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")