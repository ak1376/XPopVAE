"""
Minimal GRL sanity check — matrix W edition.

  x_source ~ N(-mu/2 · 1, sigma_x² · I)   label 0   (1 = D_IN-dim all-ones)
  x_target ~ N(+mu/2 · 1, sigma_x² · I)   label 1

  Feature extractor : z = W x    (W is D_OUT×D_IN, the only trainable param)
  Discriminator     : logit = d · z   (d frozen throughout)

Movies per run (4 panels):
  Top-left     W heatmap (activation map) at the current epoch
  Bottom-left  ‖W‖_F over epochs (trace + dot)
  Top-right    2-D scatter of source / target in z-space + decision boundary
  Bottom-right 1-D KDE of z projected onto the disc direction (distributions)
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
from scipy.stats import gaussian_kde

MU         = 2.0
SIGMA_X    = 0.5
D_IN       = 8        # input dimensionality
D_OUT      = 8        # feature (z) dimensionality — scatter shows first 2 dims
BATCH_SIZE = 32       # mini-batch size per class per step
N_EPOCHS   = 300
N_MOVIE    = 150
N_EVAL     = 600      # eval points per class

OUTPUT_DIR = Path("grl_sanity_outputs")


# ── Gradient-Reversal Layer ──────────────────────────────────────────────────

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(_ctx, grad):
        return -grad


# ── Model ─────────────────────────────────────────────────────────────────────

class ToyModel(nn.Module):
    def __init__(self, d_in=D_IN, d_out=D_OUT, freeze_disc=True):
        super().__init__()
        self.W    = nn.Parameter(torch.randn(d_out, d_in) * 0.3)
        self.disc = nn.Linear(d_out, 1, bias=False)
        nn.init.constant_(self.disc.weight, 1.0 / d_out ** 0.5)
        self.disc.weight.requires_grad = not freeze_disc

    def forward(self, x, use_grl):
        z = x @ self.W.t()                     # (B, d_out)
        z = GRL.apply(z) if use_grl else z
        return self.disc(z)                     # (B, 1)


# ── Data helpers ──────────────────────────────────────────────────────────────

def sample_batch(mu, d_in, n, rng):
    x_src = rng.normal(-mu / 2, SIGMA_X, (n, d_in)).astype(np.float32)
    x_tgt = rng.normal( mu / 2, SIGMA_X, (n, d_in)).astype(np.float32)
    x     = np.concatenate([x_src, x_tgt])
    y     = np.array([0.0] * n + [1.0] * n, dtype=np.float32)
    return x, y


def make_eval_data(mu, d_in, n=N_EVAL, seed=42):
    return sample_batch(mu, d_in, n, np.random.default_rng(seed))


def disc_acc(W_np, disc_w, x_eval, y_eval):
    z     = x_eval @ W_np.T        # (2n, d_out)
    logit = z @ disc_w.ravel()     # (2n,)
    return ((logit > 0) == y_eval.astype(bool)).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def run(use_grl, mu=MU, d_in=D_IN, d_out=D_OUT, lr=0.05, freeze_disc=True):
    torch.manual_seed(0)
    rng       = np.random.default_rng(1)
    model     = ToyModel(d_in=d_in, d_out=d_out, freeze_disc=freeze_disc)
    params    = [model.W] if freeze_disc else list(model.parameters())
    optimizer = optim.SGD(params, lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x_eval, y_eval = make_eval_data(mu, d_in)

    W_hist, disc_hist, loss_hist, acc_hist = [], [], [], []
    for _ in range(N_EPOCHS):
        x_np, y_np = sample_batch(mu, d_in, BATCH_SIZE, rng)
        x_t  = torch.from_numpy(x_np)
        y_t  = torch.from_numpy(y_np).unsqueeze(1)

        logits = model(x_t, use_grl=use_grl)
        loss   = criterion(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        W_np  = model.W.detach().numpy().copy()
        dw_np = model.disc.weight.detach().numpy().copy()  # (1, d_out)
        W_hist.append(W_np)
        disc_hist.append(dw_np)
        loss_hist.append(loss.item())
        acc_hist.append(disc_acc(W_np, dw_np, x_eval, y_eval))

    return (np.stack(W_hist),       # (N, d_out, d_in)
            np.stack(disc_hist),    # (N, 1, d_out)
            np.array(loss_hist),
            np.array(acc_hist))


# ── Movie ─────────────────────────────────────────────────────────────────────

def make_movie(W_hist, disc_w_fixed, mu, d_in, label, filename, color):
    """
    W_hist      : (n_frames, d_out, d_in)
    disc_w_fixed: (d_out,)  fixed discriminator weights
    """
    x_eval, y_eval = make_eval_data(mu, d_in, seed=99)
    n_frames       = len(W_hist)
    epochs         = np.arange(1, n_frames + 1)
    d_out, d_in_   = W_hist[0].shape
    disc_w         = disc_w_fixed.ravel()       # (d_out,)
    src_mask       = y_eval == 0
    tgt_mask       = y_eval == 1

    w_abs_max = max(float(np.abs(W_hist).max()), 0.3)
    frob_hist = np.linalg.norm(W_hist.reshape(n_frames, -1), axis=1)

    # pre-compute z and disc projections for every frame
    z_all_frames = np.stack([x_eval @ W_hist[t].T for t in range(n_frames)])
    # (n_frames, 2*N_EVAL, d_out)
    proj_all   = z_all_frames @ disc_w          # (n_frames, 2*N_EVAL)
    proj_min   = proj_all.min() - 0.5
    proj_max   = proj_all.max() + 0.5
    proj_range = np.linspace(proj_min, proj_max, 300)

    # ── figure layout ──
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            width_ratios=[1.0, 1.6],
                            height_ratios=[1.3, 1.0])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_frob = fig.add_subplot(gs[1, 0])
    ax_scat = fig.add_subplot(gs[0, 1])
    ax_proj = fig.add_subplot(gs[1, 1])

    fig.suptitle(f'{label}   μ={mu}  D_IN={d_in_}  D_OUT={d_out}',
                 fontsize=11, fontweight='bold')

    # ── W heatmap (activation map) ──
    im = ax_heat.imshow(W_hist[0], aspect='auto', cmap='RdBu_r',
                        vmin=-w_abs_max, vmax=w_abs_max, interpolation='nearest')
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.05, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    ax_heat.set_title('W  (activation map)', fontsize=9)
    ax_heat.set_xlabel('input feature', fontsize=8)
    ax_heat.set_ylabel('output dim', fontsize=8)
    ax_heat.set_xticks(range(d_in_))
    ax_heat.set_yticks(range(d_out))
    epoch_lbl = ax_heat.text(0.03, 0.97, 'Epoch 1',
                              transform=ax_heat.transAxes, fontsize=8, va='top',
                              bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

    # ── Frobenius norm trace ──
    ax_frob.plot(epochs, frob_hist, color='lightgray', lw=1.5, zorder=1)
    frob_trace, = ax_frob.plot([], [], color=color, lw=2.0, zorder=2)
    frob_dot,   = ax_frob.plot([], [], 'o', color='red', ms=6, zorder=3)
    ax_frob.set_xlim(0, n_frames + 1)
    ax_frob.set_ylim(0, frob_hist.max() * 1.15)
    ax_frob.set_xlabel('Epoch', fontsize=8)
    ax_frob.set_ylabel('‖W‖_F', fontsize=8)
    ax_frob.set_title('Frobenius norm of W', fontsize=9)
    ax_frob.grid(True, alpha=0.25)

    # ── 2-D z scatter ──
    scat_src = ax_scat.scatter([], [], s=8, alpha=0.35, color='steelblue',
                               label='Source', linewidths=0)
    scat_tgt = ax_scat.scatter([], [], s=8, alpha=0.35, color='crimson',
                               label='Target', linewidths=0)
    ax_scat.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_scat.axvline(0, color='k', lw=0.4, alpha=0.3)
    if d_out == 2 and abs(disc_w[1]) > 1e-9:
        db_line, = ax_scat.plot([], [], 'k--', lw=1.5, label='Decision boundary')
    else:
        db_line = None
    ax_scat.set_xlabel('z₁', fontsize=9)
    ax_scat.set_ylabel('z₂', fontsize=9)
    ax_scat.set_title(f'Feature space  z = Wx  (z₁,z₂ of {d_out})', fontsize=9)
    ax_scat.legend(fontsize=8, loc='upper right')
    ax_scat.grid(True, alpha=0.2)
    scat_acc = ax_scat.text(0.02, 0.03, '', transform=ax_scat.transAxes,
                            fontsize=9, va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9))

    # ── 1-D KDE along disc direction ──
    ax_proj.axvline(0, color='k', ls='--', lw=1.5, alpha=0.7,
                    label='Decision boundary')
    ax_proj.axvspan(proj_min, 0, alpha=0.05, color='steelblue')
    ax_proj.axvspan(0, proj_max, alpha=0.05, color='crimson')
    src_kde_line, = ax_proj.plot([], [], color='steelblue', lw=2.5, label='Source')
    tgt_kde_line, = ax_proj.plot([], [], color='crimson',   lw=2.5, label='Target')
    ax_proj.set_xlim(proj_min, proj_max)
    ax_proj.set_ylim(0, 1)
    ax_proj.set_xlabel('d · z  (discriminator projection)', fontsize=8)
    ax_proj.set_ylabel('KDE density', fontsize=8)
    ax_proj.set_title('Distributions along disc direction', fontsize=9)
    ax_proj.legend(fontsize=8, loc='upper right')
    ax_proj.grid(True, alpha=0.25)

    all_artists = [im, epoch_lbl, frob_trace, frob_dot,
                   scat_src, scat_tgt, scat_acc,
                   src_kde_line, tgt_kde_line]
    if db_line is not None:
        all_artists.append(db_line)

    def init():
        im.set_data(W_hist[0])
        epoch_lbl.set_text('Epoch 1')
        frob_trace.set_data([], [])
        frob_dot.set_data([], [])
        scat_src.set_offsets(np.empty((0, 2)))
        scat_tgt.set_offsets(np.empty((0, 2)))
        scat_acc.set_text('')
        src_kde_line.set_data([], [])
        tgt_kde_line.set_data([], [])
        if db_line is not None:
            db_line.set_data([], [])
        return all_artists

    def update(frame):
        W    = W_hist[frame]
        z_fr = z_all_frames[frame]      # (2*N_EVAL, d_out)
        proj = proj_all[frame]          # (2*N_EVAL,)

        # heatmap
        im.set_data(W)
        epoch_lbl.set_text(f'Epoch {frame + 1}')

        # Frobenius trace
        frob_trace.set_data(epochs[:frame + 1], frob_hist[:frame + 1])
        frob_dot.set_data([epochs[frame]], [frob_hist[frame]])

        # scatter (first 2 z-dims)
        z2 = z_fr[:, :2]
        scat_src.set_offsets(z2[src_mask])
        scat_tgt.set_offsets(z2[tgt_mask])
        pad = 0.4
        ax_scat.set_xlim(z2[:, 0].min() - pad, z2[:, 0].max() + pad)
        ax_scat.set_ylim(z2[:, 1].min() - pad, z2[:, 1].max() + pad)
        if db_line is not None:
            xl = np.array(ax_scat.get_xlim())
            yl = -(disc_w[0] / disc_w[1]) * xl
            db_line.set_data(xl, yl)

        acc = disc_acc(W, disc_w, x_eval, y_eval)
        scat_acc.set_text(f'Disc acc = {acc * 100:.1f}%')

        # KDE projections
        proj_src = proj[src_mask]
        proj_tgt = proj[tgt_mask]
        try:
            kde_src = gaussian_kde(proj_src)(proj_range)
            kde_tgt = gaussian_kde(proj_tgt)(proj_range)
        except Exception:
            kde_src = kde_tgt = np.zeros_like(proj_range)
        src_kde_line.set_data(proj_range, kde_src)
        tgt_kde_line.set_data(proj_range, kde_tgt)
        ax_proj.set_ylim(0, max(kde_src.max(), kde_tgt.max(), 0.1) * 1.15)

        return all_artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  init_func=init, interval=60, blit=False)
    ani.save(filename, writer=PillowWriter(fps=20))
    print(f"  Saved → {filename}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {OUTPUT_DIR.resolve()}")

    configs = [
        ('No GRL  frozen d',    False, True,  'steelblue'),
        ('GRL     frozen d',    True,  True,  'darkorange'),
        ('No GRL  trainable d', False, False, 'mediumseagreen'),
        ('GRL     trainable d', True,  False, 'crimson'),
    ]

    results = {}
    for label, use_grl, freeze_disc, color in configs:
        print(f"Running {label} ...")
        results[label] = run(use_grl=use_grl, freeze_disc=freeze_disc) + (color,)

    epochs = np.arange(1, N_EPOCHS + 1)

    # ── summary training-curves plot ──
    fig, axes = plt.subplots(3, len(configs),
                             figsize=(5 * len(configs), 10), sharex=True)
    fig.suptitle(
        f'GRL sanity check  (μ={MU}, σ_x={SIGMA_X}, D_IN={D_IN}, D_OUT={D_OUT})',
        fontsize=11, fontweight='bold',
    )
    for col, (label, *_) in enumerate(configs):
        W_h, _, L_h, acc_h, color = results[label]
        frob = np.linalg.norm(W_h.reshape(N_EPOCHS, -1), axis=1)

        axes[0, col].plot(epochs, frob, color=color, lw=2)
        axes[0, col].set_title(label, fontsize=9, fontweight='bold')
        axes[0, col].set_ylabel('‖W‖_F')
        axes[0, col].grid(True, alpha=0.25)

        axes[1, col].plot(epochs, L_h, color=color, lw=2)
        axes[1, col].axhline(np.log(2), color='k', ls='--', lw=0.8, alpha=0.5,
                             label=f'log 2 ≈ {np.log(2):.3f}')
        axes[1, col].set_ylabel('Domain loss')
        axes[1, col].grid(True, alpha=0.25)
        axes[1, col].legend(fontsize=8)

        axes[2, col].plot(epochs, acc_h, color=color, lw=2)
        axes[2, col].axhline(0.5, color='k', ls='--', lw=0.8, alpha=0.5,
                             label='chance')
        axes[2, col].set_ylabel('Disc acc')
        axes[2, col].set_xlabel('Epoch')
        axes[2, col].set_ylim(0, 1.05)
        axes[2, col].grid(True, alpha=0.25)
        axes[2, col].legend(fontsize=8)

    plt.tight_layout()
    summary_path = OUTPUT_DIR / 'sanity_grl.png'
    plt.savefig(summary_path, dpi=150)
    print(f"Saved → {summary_path}")

    # ── terminal summary ──
    for label, *_ in configs:
        W_h, _, L_h, acc_h, _ = results[label]
        print(f"  {label:25s}: ‖W‖_F={np.linalg.norm(W_h[-1]):.4f}  "
              f"loss={L_h[-1]:.4f}  acc={acc_h[-1]*100:.1f}%")

    # ── movies ──
    movie_configs = [
        ('No GRL  frozen d',    OUTPUT_DIR / 'no_grl.gif'),
        ('GRL     frozen d',    OUTPUT_DIR / 'grl.gif'),
        ('GRL     trainable d', OUTPUT_DIR / 'grl_trainable_disc.gif'),
    ]
    for label, fname in movie_configs:
        W_h, disc_h, _, _, color = results[label]
        disc_w = disc_h[-1].ravel()
        print(f"\nMaking movie: {fname}  ({N_MOVIE} frames) ...")
        make_movie(W_h[:N_MOVIE], disc_w, mu=MU, d_in=D_IN,
                   label=label, filename=fname, color=color)

    print("\nDone.")


if __name__ == "__main__":
    main()
