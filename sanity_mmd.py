"""
sanity_mmd.py

Domain adaptation sanity check:
  - Encoder (MLP: x → z) + Predictor (z → y, regression)
  - Task loss: MSE on source training data only
  - MMD alignment: minimize MMD(z_src_train, z_tgt_train) directly (no GRL)
  - Evaluate: source val MSE and held-out target test MSE

Data splits:
  src_train (200): x ~ N(-1, 0.5),  y = x² + noise  — task loss + MMD
  tgt_train (200): x ~ N(+1, 0.5),  unlabeled        — MMD only
  src_val   (100): x ~ N(-1, 0.5),  y = x² + noise  — source evaluation
  tgt_test  (100): x ~ N(+1, 0.5),  y = x² + noise  — held-out target evaluation

NOTE: y = x² so P_src(y) ≈ P_tgt(y) ≈ N(1, ...) — both domains have same label
  distribution. This is required for domain adaptation to help. With y = 2x, the
  label distributions differ (-2 vs +2), so alignment actively hurts prediction.

Loss: task_loss + lam * MMD(z_src_train, z_tgt_train)

Success criteria:
  - MMD decreases toward 0 (latent distributions align)
  - KS statistic decreases (domains mix in latent space)
  - tgt_mse drops compared to lam=0 baseline (adaptation helps)
  - src_val_mse stays low (task performance preserved)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------------
# MMD loss  (direct minimization, no GRL)
# ---------------------------------------------------------------------------
def median_bandwidth(z_src, z_tgt):
    z_all = torch.cat([z_src, z_tgt], dim=0)
    D = (z_all.unsqueeze(1) - z_all.unsqueeze(0)).pow(2).sum(-1)
    upper = torch.triu(D, diagonal=1)
    return upper[upper > 0].median().item()


def rbf_kernel(a, b, sigma):
    D = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
    return torch.exp(-D / (2 * sigma ** 2))


def mmd_loss(z_src, z_tgt, sigma=None):
    if sigma is None:
        with torch.no_grad():
            sigma = median_bandwidth(z_src, z_tgt)
    N, M = z_src.shape[0], z_tgt.shape[0]
    K_ss = rbf_kernel(z_src, z_src, sigma)
    K_tt = rbf_kernel(z_tgt, z_tgt, sigma)
    K_st = rbf_kernel(z_src, z_tgt, sigma)
    mask_s = 1 - torch.eye(N, device=z_src.device)
    mask_t = 1 - torch.eye(M, device=z_tgt.device)
    mmd = (K_ss * mask_s).sum() / (N * (N - 1)) \
        + (K_tt * mask_t).sum() / (M * (M - 1)) \
        - 2 * K_st.sum() / (N * M)
    # return mmd.clamp(min=0)
    return mmd


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.net = nn.Linear(latent_dim, 1)

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def make_split(mu, n, std=0.5, noise_std=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = rng.normal(mu, std, (n, 1)).astype(np.float32)
    y = (x ** 2 + noise_std * rng.standard_normal(x.shape)).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    n_train    = 200
    n_val      = 100
    n_epochs   = 5000
    log_every  = 50
    snap_every = 50
    lr         = 3e-3
    lam        = 1.0
    latent_dim = 2

    # --- data ---
    src_x_train, src_y_train = make_split(-5.0, n_train, rng=rng)
    src_x_val,   src_y_val   = make_split(-5.0, n_val,   rng=rng)
    tgt_x_train, _           = make_split( 5.0, n_train, rng=rng)   # unlabeled
    tgt_x_test,  tgt_y_test  = make_split( 5.0, n_val,   rng=rng)

    # Normalize y by source training statistics so task loss stays O(1)
    # regardless of domain gap size. Apply same transform to all splits.
    y_mean = src_y_train.mean()
    y_std  = src_y_train.std().clamp(min=1e-6)
    src_y_train = (src_y_train - y_mean) / y_std
    src_y_val   = (src_y_val   - y_mean) / y_std
    tgt_y_test  = (tgt_y_test  - y_mean) / y_std

    # --- models ---
    encoder   = Encoder(latent_dim=latent_dim)
    predictor = Predictor(latent_dim=latent_dim)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )
    mse_fn = nn.MSELoss()

    # --- history ---
    train_mse_hist = []
    val_mse_hist   = []
    tgt_mse_hist   = []
    mmd_hist       = []
    ks_hist        = []
    snapshots      = []      # (epoch, z_src_np, z_tgt_np)

    for epoch in range(n_epochs):
        encoder.train(); predictor.train()

        z_src = encoder(src_x_train)
        z_tgt = encoder(tgt_x_train)

        task_loss  = mse_fn(predictor(z_src), src_y_train)
        domain_mmd = mmd_loss(z_src, z_tgt)
        total_loss = task_loss + lam * domain_mmd

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # --- diagnostics ---
        encoder.eval(); predictor.eval()
        with torch.no_grad():
            val_mse = mse_fn(predictor(encoder(src_x_val)),  src_y_val).item()
            tgt_mse = mse_fn(predictor(encoder(tgt_x_test)), tgt_y_test).item()

            z_s = z_src.detach().numpy()
            z_t = z_tgt.detach().numpy()
            z_s_1d = np.linalg.norm(z_s, axis=1)
            z_t_1d = np.linalg.norm(z_t, axis=1)
            ks, _ = ks_2samp(z_s_1d, z_t_1d)

        train_mse_hist.append(task_loss.item())
        val_mse_hist.append(val_mse)
        tgt_mse_hist.append(tgt_mse)
        mmd_hist.append(domain_mmd.item())
        ks_hist.append(ks)

        if epoch % snap_every == 0:
            snapshots.append((epoch, z_s.copy(), z_t.copy()))

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:5d} | task={task_loss.item():.4f} | MMD={domain_mmd.item():.4f} "
                f"| src_val={val_mse:.4f} | tgt_test={tgt_mse:.4f} | KS={ks:.4f}"
            )

    # --- training curves ---
    epochs = np.arange(n_epochs)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(epochs, train_mse_hist, label="train MSE (src)", color="steelblue", linewidth=1.2)
    axes[0].plot(epochs, val_mse_hist,   label="val MSE (src)",   color="navy",      linewidth=1.2, linestyle="--")
    axes[0].plot(epochs, tgt_mse_hist,   label="test MSE (tgt)",  color="tomato",    linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"Encoder + Predictor + MMD alignment  (λ={lam}, y=x²)")
    axes[0].legend()

    axes[1].plot(epochs, mmd_hist, color="purple", linewidth=1.2)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("MMD")

    axes[2].plot(epochs, ks_hist, color="tomato", linewidth=1.2)
    axes[2].axhline(0.0, color="gray", linestyle="--", linewidth=0.8, label="perfect mixing")
    axes[2].set_ylabel("KS statistic")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("mmd_da_training.png", dpi=150)
    plt.close()
    print("Saved mmd_da_training.png")

    # --- final latent distributions ---
    encoder.eval()
    with torch.no_grad():
        z_s_final = encoder(src_x_train).numpy()
        z_t_final = encoder(tgt_x_train).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z_s_final[:, 0], z_s_final[:, 1], alpha=0.5, s=10, label="source", color="steelblue")
    ax.scatter(z_t_final[:, 0], z_t_final[:, 1], alpha=0.5, s=10, label="target",  color="tomato")
    ax.set_xlabel("z₀"); ax.set_ylabel("z₁")
    ax.set_title(f"Final latent (2D)  |  KS={ks_hist[-1]:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("mmd_da_final_z.png", dpi=150)
    plt.close()
    print("Saved mmd_da_final_z.png")

    # --- scatter animation over training ---
    all_z = np.concatenate([np.vstack([zs, zt]) for _, zs, zt in snapshots])
    z0_min, z0_max = all_z[:, 0].min() - 0.3, all_z[:, 0].max() + 0.3
    z1_min, z1_max = all_z[:, 1].min() - 0.3, all_z[:, 1].max() + 0.3

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx):
        ax.clear()
        epoch, z_s, z_t = snapshots[frame_idx]
        ax.scatter(z_s[:, 0], z_s[:, 1], alpha=0.5, s=8, label="source", color="steelblue")
        ax.scatter(z_t[:, 0], z_t[:, 1], alpha=0.5, s=8, label="target",  color="tomato")
        ax.set_xlim(z0_min, z0_max); ax.set_ylim(z1_min, z1_max)
        ax.set_xlabel("z₀"); ax.set_ylabel("z₁")
        ei = frame_idx * snap_every
        ax.set_title(
            f"Latent z — epoch {epoch}  |  "
            f"KS={ks_hist[ei]:.3f}  MMD={mmd_hist[ei]:.4f}  "
            f"tgt_mse={tgt_mse_hist[ei]:.4f}"
        )
        ax.legend(loc="upper right")

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=150, repeat=True)
    ani.save("mmd_da_animation.gif", writer="pillow", fps=8)
    plt.close()
    print("Saved mmd_da_animation.gif")


if __name__ == "__main__":
    main()
