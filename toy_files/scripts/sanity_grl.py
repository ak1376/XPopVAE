"""
sanity_dann.py

Sanity check: domain classifier + GRL with z as direct learnable parameter.

Setup:
  - z_source initialized at N(-1, 0.5)
  - z_target initialized at N(+1, 0.5)
  - Domain classifier: z -> linear -> sigmoid -> domain probability
  - GRL between z and classifier: encoder (z) is pushed to fool the classifier
  - Joint training: z and classifier updated together each step

Success criteria:
  - KS statistic decreases toward 0 (distributions mix)
  - src_std and tgt_std stay near their initial values (~0.5)
  - src_mean and tgt_mean converge toward each other
  - domain_acc drops toward 0.5 (classifier can't distinguish domains)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Domain classifier
# ---------------------------------------------------------------------------
class DomainClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z)


def main():
    num_samples = 100
    n_epochs    = 10000
    snapshot_interval = 20
    lr_z    = 1e-3
    lr_disc = 1e-3
    lam     = 1.0

    torch.manual_seed(42)
    mu, std = 2.0, 0.5
    src_data = torch.tensor(np.random.normal(-mu/2, std, (num_samples, 1)), dtype=torch.float32)
    tgt_data = torch.tensor(np.random.normal( mu/2, std, (num_samples, 1)), dtype=torch.float32)

    z_source = nn.Parameter(src_data.clone())
    z_target = nn.Parameter(tgt_data.clone())

    source_labels = torch.zeros(num_samples, 1)
    target_labels = torch.ones(num_samples, 1)

    classifier = DomainClassifier(input_dim=1, hidden_dim=16)

    # Separate optimizers so we can control lr independently if needed
    optimizer_z    = torch.optim.SGD([z_source, z_target], lr=lr_z)
    optimizer_disc = torch.optim.SGD(classifier.parameters(), lr=lr_disc)

    loss_history = []
    acc_history  = []
    ks_history   = []
    snapshots    = []

    for epoch in range(n_epochs):
        # --- Forward pass ---
        # Apply GRL before passing z to classifier
        z_src_grl = grad_reverse(z_source, lam)
        z_tgt_grl = grad_reverse(z_target, lam)

        z_all_grl = torch.cat([z_src_grl, z_tgt_grl], dim=0)
        labels    = torch.cat([source_labels, target_labels], dim=0)

        logits = classifier(z_all_grl)
        loss   = F.binary_cross_entropy_with_logits(logits, labels)

        # --- Backward + update ---
        optimizer_z.zero_grad()
        optimizer_disc.zero_grad()
        loss.backward()
        optimizer_z.step()
        optimizer_disc.step()

        # --- Diagnostics (no gradient) ---
        with torch.no_grad():
            src_np  = z_source.numpy().flatten()
            tgt_np  = z_target.numpy().flatten()
            ks, _   = ks_2samp(src_np, tgt_np)

            z_all   = torch.cat([z_source, z_target], dim=0)
            preds   = (torch.sigmoid(classifier(z_all)) > 0.5).float()
            acc     = (preds == labels).float().mean().item()

        loss_history.append(loss.item())
        ks_history.append(ks)
        acc_history.append(acc)

        if epoch % snapshot_interval == 0:
            snapshots.append((epoch, src_np.copy(), tgt_np.copy()))
            print(
                f"Epoch {epoch:5d} | loss={loss.item():.4f} | KS={ks:.4f} "
                f"| src_mean={src_np.mean():.3f} | tgt_mean={tgt_np.mean():.3f} "
                f"| src_std={src_np.std():.3f} | tgt_std={tgt_np.std():.3f} "
                f"| disc_acc={acc:.3f}"
            )

    # --- Loss, accuracy, KS curves ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax1.plot(loss_history, linewidth=1.5, color="steelblue")
    ax1.set_ylabel("BCE loss")
    ax1.set_title(f"Domain classifier + GRL (λ={lam}) — z as direct parameter")

    ax2.plot(acc_history, linewidth=1.5, color="purple")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance (0.5)")
    ax2.set_ylabel("Disc accuracy")
    ax2.legend()

    ax3.plot(ks_history, linewidth=1.5, color="tomato")
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, label="perfect mixing")
    ax3.set_ylabel("KS statistic")
    ax3.set_xlabel("Epoch")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("dann_training.png", dpi=150)
    plt.close()
    print("Saved dann_training.png")

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
        epoch_idx = frame_idx * snapshot_interval
        ax.set_title(
            f"Latent z — epoch {epoch}  |  "
            f"KS={ks_history[epoch_idx]:.3f}  |  "
            f"disc_acc={acc_history[epoch_idx]:.3f}"
        )
        ax.legend(loc="upper right")

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=150, repeat=True)
    ani.save("dann_animation.gif", writer="pillow", fps=8)
    plt.close()
    print("Saved dann_animation.gif")


if __name__ == "__main__":
    main()