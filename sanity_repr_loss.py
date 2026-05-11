"""
sanity_repr_loss.py

Stage 1 sanity check: verify repr_loss has correct gradient signal
by treating z directly as a learnable parameter (no encoder).

Starting from two overlapping clusters, we ask:
  does minimizing repr_loss move source and target points into separated clusters?

If yes: the loss is correct. Encoder parametrization is a separate concern (Stage 2).
If no:  the loss itself has a bug.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from scipy.stats import ks_2samp


def repr_loss(z, domain_labels, eps=0.1):
    """
    Pairwise contrastive representational loss (upper triangle, no double counting).
      same-domain pair  (a == b): D(y, z)        (pull together)
      cross-domain pair (a != b): 1 / D(y, z)    (push apart)

    eps=0.1 prevents gradient explosion when cross-domain pairs are nearly coincident.
    """
    N = z.shape[0]

    upper = torch.triu(torch.ones(N, N, device=z.device), diagonal=1).bool()

    diff = z - z.t()
    D = diff.pow(2)

    labels_i = domain_labels.expand(N, N)
    labels_j = domain_labels.t().expand(N, N)

    same  = (labels_i == labels_j).float()
    cross = 1.0 - same

    same_pairs  = same[upper]
    cross_pairs = cross[upper]
    D_upper     = D[upper]

    same_loss  = (same_pairs  * D_upper).sum()         / same_pairs.sum().clamp(min=1)
    cross_loss = (cross_pairs / (D_upper + eps)).sum() / cross_pairs.sum().clamp(min=1)

    return same_loss + cross_loss


def main():
    num_samples = 100
    n_epochs = 10000
    snapshot_interval = 20
    lr = 1e-3

    # Both domains initialized as tight overlapping clusters near 0
    # Small noise so pairwise distances aren't exactly zero
    torch.manual_seed(42)
    z_source = nn.Parameter(torch.randn(num_samples, 1) * 0.001)
    z_target = nn.Parameter(torch.randn(num_samples, 1) * 0.001)

    source_labels = torch.zeros(num_samples, 1)
    target_labels = torch.ones(num_samples, 1)
    labels = torch.cat([source_labels, target_labels])  # (2N, 1)

    optimizer = torch.optim.SGD([z_source, z_target], lr=lr)

    # Tracking
    loss_history = []
    ks_history   = []
    snapshots    = []  # (epoch, z_src_np, z_tgt_np)

    for epoch in range(n_epochs):
        z = torch.cat([z_source, z_target])  # (2N, 1)
        loss = repr_loss(z, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        z_np = z.detach().numpy().flatten()
        ks, _ = ks_2samp(z_np[:num_samples], z_np[num_samples:])
        ks_history.append(ks)

        if epoch % snapshot_interval == 0:
            snapshots.append((
                epoch,
                z_np[:num_samples].copy(),
                z_np[num_samples:].copy()
            ))
            print(
                f"Epoch {epoch:4d} | loss={loss.item():.4f} | KS={ks:.4f} "
                f"| src_mean={z_np[:num_samples].mean():.3f} "
                f"| tgt_mean={z_np[num_samples:].mean():.3f}"
            )

    # --- Loss and KS over training ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(loss_history, linewidth=1.5, color="steelblue")
    ax1.set_ylabel("repr_loss")
    ax1.set_title("Stage 1: repr_loss with z as direct parameter")

    ax2.plot(ks_history, linewidth=1.5, color="tomato")
    ax2.set_ylabel("KS statistic")
    ax2.set_xlabel("Epoch")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="max separation")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("repr_loss_training.png", dpi=150)
    plt.close()
    print("Saved repr_loss_training.png")

    # --- Animation of z distributions ---
    all_z = np.concatenate([np.concatenate([s, t]) for _, s, t in snapshots])
    z_min, z_max = all_z.min() - 0.2, all_z.max() + 0.2

    fig, ax = plt.subplots(figsize=(7, 4))

    def update(frame_idx):
        ax.clear()
        epoch, z_src, z_tgt = snapshots[frame_idx]
        bins = np.linspace(z_min, z_max, 30)
        ax.hist(z_src, bins=bins, alpha=0.6, label="source", color="steelblue", density=True)
        ax.hist(z_tgt, bins=bins, alpha=0.6, label="target",  color="tomato",   density=True)
        ax.set_xlim(z_min, z_max)
        ax.set_xlabel("z")
        ax.set_ylabel("density")
        ax.set_title(f"Latent z distribution — epoch {epoch}")
        ax.legend(loc="upper right")

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=150, repeat=True)
    ani.save("repr_loss_animation.gif", writer="pillow", fps=8)
    plt.close()
    print("Saved repr_loss_animation.gif")


if __name__ == "__main__":
    main()