"""
sanity_mmd_grl.py

Sanity check: does MMD + GRL merge two separated distributions
while preserving within-domain variance?

Setup:
  - z_source initialized at N(-1, 0.5)
  - z_target initialized at N(+1, 0.5)
  - MMD loss with GRL: reversed gradient pulls distributions together
  - No repr_loss — purely testing MMD+GRL merging behavior

Success criteria:
  - KS statistic decreases toward 0 (distributions mix)
  - src_std and tgt_std stay close to their initial values (~0.5)
  - src_mean and tgt_mean converge toward each other
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from torch.autograd import Function
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lam, = ctx.saved_tensors
        return -lam * grad_output, None


def grad_reverse(x, lam=1.0):
    return GradientReversalFn.apply(x, lam)


# ---------------------------------------------------------------------------
# MMD loss with optional GRL
# ---------------------------------------------------------------------------
def median_bandwidth(z_src, z_tgt):
    """Median heuristic: set sigma to median pairwise distance across all points."""
    z_all = torch.cat([z_src, z_tgt], dim=0)
    D = (z_all - z_all.t()).pow(2)
    upper = torch.triu(D, diagonal=1)
    median_dist = upper[upper > 0].median()
    return median_dist.item()


def mmd_loss(z_src, z_tgt, lam=1.0, sigma=None):
    """
    MMD loss with GRL.

    GRL reverses the gradient, turning the MMD minimization
    (which would merge distributions) into a merging force on z directly.

    sigma: kernel bandwidth. If None, uses median heuristic.
    """
    # Apply GRL
    z_src_grl = grad_reverse(z_src, lam)
    z_tgt_grl = grad_reverse(z_tgt, lam)

    if sigma is None:
        # Compute median bandwidth on detached values (no gradient through sigma)
        with torch.no_grad():
            sigma = median_bandwidth(z_src, z_tgt)

    def rbf_kernel(a, b, sigma):
        D = (a - b.t()).pow(2)
        return torch.exp(-D / (2 * sigma ** 2))

    K_ss = rbf_kernel(z_src_grl, z_src_grl, sigma)
    K_tt = rbf_kernel(z_tgt_grl, z_tgt_grl, sigma)
    K_st = rbf_kernel(z_src_grl, z_tgt_grl, sigma)

    N = z_src.shape[0]
    M = z_tgt.shape[0]

    # Exclude diagonal for within-domain terms
    mask_s = (1 - torch.eye(N, device=z_src.device))
    mask_t = (1 - torch.eye(M, device=z_tgt.device))

    mmd = (K_ss * mask_s).sum() / (N * (N - 1)) \
        + (K_tt * mask_t).sum() / (M * (M - 1)) \
        - 2 * K_st.sum() / (N * M)

    return mmd


def main():
    num_samples = 100
    n_epochs    = 10000
    snapshot_interval = 20
    lr  = 1e-3
    lam = 1.0

    torch.manual_seed(42)
    mu, std = 2.0, 0.5
    src_data = torch.tensor(np.random.normal(-mu/2, std, (num_samples, 1)), dtype=torch.float32)
    tgt_data = torch.tensor(np.random.normal( mu/2, std, (num_samples, 1)), dtype=torch.float32)

    z_source = nn.Parameter(src_data.clone())
    z_target = nn.Parameter(tgt_data.clone())

    optimizer = torch.optim.SGD([z_source, z_target], lr=lr)

    loss_history = []
    ks_history   = []
    snapshots    = []

    for epoch in range(n_epochs):
        loss = mmd_loss(z_source, z_target, lam=lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        src_np = z_source.detach().numpy().flatten()
        tgt_np = z_target.detach().numpy().flatten()
        ks, _  = ks_2samp(src_np, tgt_np)
        ks_history.append(ks)

        if epoch % snapshot_interval == 0:
            snapshots.append((epoch, src_np.copy(), tgt_np.copy()))
            print(
                f"Epoch {epoch:5d} | mmd={loss.item():.4f} | KS={ks:.4f} "
                f"| src_mean={src_np.mean():.3f} | tgt_mean={tgt_np.mean():.3f} "
                f"| src_std={src_np.std():.3f} | tgt_std={tgt_np.std():.3f}"
            )

    # --- Loss and KS curves ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(loss_history, linewidth=1.5, color="steelblue")
    ax1.set_ylabel("MMD (forward value)")
    ax1.set_title(f"MMD + GRL (λ={lam}) — z as direct parameter")

    ax2.plot(ks_history, linewidth=1.5, color="tomato")
    ax2.set_ylabel("KS statistic")
    ax2.set_xlabel("Epoch")
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, label="perfect mixing")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("mmd_grl_training.png", dpi=150)
    plt.close()
    print("Saved mmd_grl_training.png")

    # --- Animation ---
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
        ax.set_title(f"Latent z — epoch {epoch}  |  KS={ks_history[frame_idx * snapshot_interval]:.3f}")
        ax.legend(loc="upper right")

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=150, repeat=True)
    ani.save("mmd_grl_animation.gif", writer="pillow", fps=8)
    plt.close()
    print("Saved mmd_grl_animation.gif")


if __name__ == "__main__":
    main()