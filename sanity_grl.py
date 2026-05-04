"""
Minimal GRL sanity check.

  x_source ~ N(-mu/2, sigma_x²)   label 0
  x_target ~ N(+mu/2, sigma_x²)   label 1

  Feature extractor : z = w * x   (w is the ONLY trainable param)
  Discriminator     : logit = d * z  (d FROZEN throughout)

Compares:
  - No GRL: w minimises domain loss → disc improves
  - GRL:    w maximises domain loss → disc degrades
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm as scipy_norm

MU       = 2.0
SIGMA_X  = 0.5
N_EPOCHS = 300
N_MOVIE  = 150
N_EVAL   = 4000

# All outputs from this script are saved here.
# This folder will be created automatically next to wherever you run the script.
OUTPUT_DIR = Path("grl_sanity_outputs")


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(_ctx, grad):
        return -grad


class ToyModel(nn.Module):
    def __init__(self, w_init=0.5, d_init=1.0, freeze_disc=True):
        super().__init__()
        self.w    = nn.Parameter(torch.tensor([[w_init]]))
        self.disc = nn.Linear(1, 1, bias=False)
        self.disc.weight.data.fill_(d_init)
        self.disc.weight.requires_grad = not freeze_disc

    def forward(self, x, use_grl):
        z = self.w * x
        z = GRL.apply(z) if use_grl else z
        return self.disc(z)


def make_eval_data(mu):
    rng   = np.random.default_rng(42)
    x_src = rng.normal(-mu / 2, SIGMA_X, N_EVAL)
    x_tgt = rng.normal(+mu / 2, SIGMA_X, N_EVAL)
    x     = np.concatenate([x_src, x_tgt])
    y     = np.array([0] * N_EVAL + [1] * N_EVAL, dtype=float)
    return x, y


def acc_empirical(w_val, x_eval, y_eval):
    preds = (w_val * x_eval > 0).astype(float)
    return (preds == y_eval).mean()


def run(use_grl, mu=MU, w_init=0.5, lr=0.1, freeze_disc=True):
    model     = ToyModel(w_init=w_init, freeze_disc=freeze_disc)
    params    = [model.w] if freeze_disc else list(model.parameters())
    optimizer = optim.SGD(params, lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x      = torch.tensor([[-mu / 2], [mu / 2]])
    labels = torch.tensor([[0.0],     [1.0]])

    x_eval, y_eval = make_eval_data(mu)

    w_hist, d_hist, loss_hist, acc_hist = [], [], [], []
    for _ in range(N_EPOCHS):
        logits = model(x, use_grl=use_grl)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        w_val = model.w.item()
        w_hist.append(w_val)
        d_hist.append(model.disc.weight.item())
        loss_hist.append(loss.item())
        acc_hist.append(acc_empirical(w_val, x_eval, y_eval))

    return np.array(w_hist), np.array(d_hist), np.array(loss_hist), np.array(acc_hist)


def make_movie(w_hist, mu, label, filename, color):
    x_eval, y_eval = make_eval_data(mu)
    n      = len(w_hist)
    epochs = np.arange(1, n + 1)

    max_abs_w = max(abs(w_hist).max(), 1.0)
    z_lim     = min(max_abs_w * (mu / 2 + 3 * SIGMA_X) * 1.25, 30.0)
    z_range   = np.linspace(-z_lim, z_lim, 600)

    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.6])
    ax_w, ax_dist = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.suptitle(f'{label}  (μ={mu}, d frozen=1)', fontsize=13, fontweight='bold')

    ax_w.plot(epochs, w_hist, color='lightgray', lw=1.5, zorder=1)
    ax_w.axhline(0, color='k', ls='--', lw=0.9, alpha=0.5, label='w=0')
    w_trace, = ax_w.plot([], [], color=color, lw=2.2, zorder=2)
    w_dot,   = ax_w.plot([], [], 'o', color='red', ms=9, zorder=3)
    pad = max(abs(w_hist).max() * 0.12, 0.3)
    ax_w.set_xlim(0, n + 1); ax_w.set_ylim(w_hist.min() - pad, w_hist.max() + pad)
    ax_w.set_xlabel('Epoch'); ax_w.set_ylabel('w')
    ax_w.set_title('w over time  (red dot = current epoch)')
    ax_w.legend(fontsize=9); ax_w.grid(True, alpha=0.25)

    ax_dist.axvline(0, color='k', ls='--', lw=1.5, alpha=0.85, zorder=4,
                    label='Decision boundary z=0')
    ax_dist.axvspan(-z_lim, 0, alpha=0.05, color='steelblue', zorder=0)
    ax_dist.axvspan(0,  z_lim, alpha=0.05, color='crimson',   zorder=0)
    ax_dist.text(-z_lim * 0.75, 0.02, '← predict SOURCE', fontsize=8,
                 color='steelblue', va='bottom', transform=ax_dist.get_xaxis_transform())
    ax_dist.text( z_lim * 0.05, 0.02, 'predict TARGET →', fontsize=8,
                 color='crimson', va='bottom', transform=ax_dist.get_xaxis_transform())
    src_line, = ax_dist.plot([], [], color='steelblue', lw=2.5,
                              label='Source z ~ N(−w·μ/2, (|w|·σ_x)²)')
    tgt_line, = ax_dist.plot([], [], color='crimson',   lw=2.5,
                              label='Target z ~ N(+w·μ/2, (|w|·σ_x)²)')
    ax_dist.set_xlim(-z_lim, z_lim); ax_dist.set_ylim(0, None)
    ax_dist.set_xlabel('z  (feature space = w · x)'); ax_dist.set_ylabel('Density')
    ax_dist.set_title('Exact feature distributions in z-space')
    ax_dist.legend(fontsize=9, loc='upper right'); ax_dist.grid(True, alpha=0.25)
    status_box = ax_dist.text(0.02, 0.96, '', transform=ax_dist.transAxes,
                               fontsize=10, va='top', ha='left',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                         alpha=0.9, edgecolor='gray'))

    def init():
        w_trace.set_data([], []); w_dot.set_data([], [])
        src_line.set_data([], []); tgt_line.set_data([], [])
        status_box.set_text('')
        return w_trace, w_dot, src_line, tgt_line, status_box

    def update(frame):
        w = w_hist[frame]
        w_trace.set_data(epochs[:frame + 1], w_hist[:frame + 1])
        w_dot.set_data([epochs[frame]], [w])
        std_z   = max(abs(w) * SIGMA_X, 0.05)
        src_pdf = scipy_norm.pdf(z_range, loc=-w * mu / 2, scale=std_z)
        tgt_pdf = scipy_norm.pdf(z_range, loc= w * mu / 2, scale=std_z)
        src_line.set_data(z_range, src_pdf); tgt_line.set_data(z_range, tgt_pdf)
        ax_dist.set_ylim(0, max(src_pdf.max(), tgt_pdf.max(), 0.1) * 1.15)
        acc = acc_empirical(w, x_eval, y_eval)
        if   w >  0.05: s = f'w={w:+.3f}  source LEFT ✓ target RIGHT ✓  acc={acc*100:.1f}%'
        elif w < -0.05: s = f'w={w:+.3f}  source RIGHT ✗ target LEFT ✗  acc={acc*100:.1f}%'
        else:           s = f'w={w:+.3f}  distributions overlap  acc≈50%'
        status_box.set_text(s)
        return w_trace, w_dot, src_line, tgt_line, status_box

    animation.FuncAnimation(fig, update, frames=n, init_func=init,
                             interval=60, blit=False).save(
        filename, writer=PillowWriter(fps=20))
    print(f"  Saved → {filename}"); plt.close(fig)


def main():
    torch.manual_seed(0)
    epochs = np.arange(1, N_EPOCHS + 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {OUTPUT_DIR.resolve()}")

    print("Running No GRL (frozen disc) ...")
    w_no,  d_no,  L_no,  _ = run(use_grl=False, freeze_disc=True)

    print("Running GRL (frozen disc) ...")
    w_grl, d_grl, L_grl, _ = run(use_grl=True,  freeze_disc=True)

    print("Running No GRL (trainable disc) ...")
    w_no_td,  d_no_td,  L_no_td,  _ = run(use_grl=False, freeze_disc=False)

    print("Running GRL (trainable disc) ...")
    w_grl_td, d_grl_td, L_grl_td, _ = run(use_grl=True,  freeze_disc=False)

    runs = [
        ('No GRL  frozen d',    'steelblue',  w_no,     d_no,     L_no),
        ('GRL     frozen d',    'darkorange', w_grl,    d_grl,    L_grl),
        ('No GRL  trainable d', 'mediumseagreen', w_no_td, d_no_td, L_no_td),
        ('GRL     trainable d', 'crimson',    w_grl_td, d_grl_td, L_grl_td),
    ]

    curves_path = OUTPUT_DIR / 'training_curves.npz'
    np.savez(
        curves_path,
        w_no=w_no, d_no=d_no, L_no=L_no,
        w_grl=w_grl, d_grl=d_grl, L_grl=L_grl,
        w_no_td=w_no_td, d_no_td=d_no_td, L_no_td=L_no_td,
        w_grl_td=w_grl_td, d_grl_td=d_grl_td, L_grl_td=L_grl_td,
    )
    print(f"Saved → {curves_path}")

    fig, axes = plt.subplots(3, len(runs), figsize=(5 * len(runs), 11), sharex=True)
    fig.suptitle(f'GRL sanity check  (μ={MU}, σ_x={SIGMA_X})',
                 fontsize=12, fontweight='bold')

    for col, (label, color, w_h, d_h, L_h) in enumerate(runs):
        axes[0, col].plot(epochs, w_h, color=color, lw=2)
        axes[0, col].axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        axes[0, col].set_title(label, fontsize=10, fontweight='bold')
        axes[0, col].set_ylabel('w'); axes[0, col].grid(True, alpha=0.25)

        axes[1, col].plot(epochs, d_h, color=color, lw=2)
        axes[1, col].axhline(1, color='k', ls='--', lw=0.8, alpha=0.5, label='d=1 (init)')
        axes[1, col].set_ylabel('d'); axes[1, col].grid(True, alpha=0.25)
        axes[1, col].legend(fontsize=8)

        axes[2, col].plot(epochs, L_h, color=color, lw=2)
        axes[2, col].axhline(np.log(2), color='k', ls='--', lw=0.8, alpha=0.5,
                              label=f'log(2)≈{np.log(2):.3f}')
        axes[2, col].set_ylabel('Domain loss'); axes[2, col].set_xlabel('Epoch')
        axes[2, col].grid(True, alpha=0.25); axes[2, col].legend(fontsize=8)

    plt.tight_layout()
    summary_path = OUTPUT_DIR / 'sanity_grl.png'
    plt.savefig(summary_path, dpi=150)
    print(f"Saved → {summary_path}")

    for label, w_h, d_h, L_h in [
        ('No GRL  frozen d',    w_no,     d_no,     L_no),
        ('GRL     frozen d',    w_grl,    d_grl,    L_grl),
        ('No GRL  trainable d', w_no_td,  d_no_td,  L_no_td),
        ('GRL     trainable d', w_grl_td, d_grl_td, L_grl_td),
    ]:
        print(f"  {label:25s}: w={w_h[-1]:+.4f}  d={d_h[-1]:+.4f}  loss={L_h[-1]:.4f}")

    movie_runs = [
        ('No GRL  frozen d',    w_no,     'steelblue',      OUTPUT_DIR / 'no_grl.gif'),
        ('GRL     frozen d',    w_grl,    'darkorange',     OUTPUT_DIR / 'grl.gif'),
        ('GRL     trainable d', w_grl_td, 'crimson',        OUTPUT_DIR / 'grl_trainable_disc.gif'),
    ]
    for label, w_h, color, fname in movie_runs:
        print(f"\nMaking movie: {fname} ({N_MOVIE} epochs) ...")
        make_movie(w_h[:N_MOVIE], mu=MU, label=label, filename=fname, color=color)

    print("\nDone.")


if __name__ == "__main__":
    main()
