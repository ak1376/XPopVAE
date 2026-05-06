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
N_EPOCHS = 500
N_MOVIE  = 100
N_EVAL      = 4000
ALPHA_TASK  = 0.6   # weight on task loss; set to 0.0 to disable

# All outputs from this script are saved here.
# This folder will be created automatically next to wherever you run the script.
OUTPUT_DIR = Path("grl_sanity_outputs")


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class ToyModel(nn.Module):
    def __init__(self, w_init=0.5, d_init=1.0, freeze_disc=True):
        super().__init__()
        self.w         = nn.Parameter(torch.tensor([[w_init]]))
        self.task_head = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.task_head.weight, 1.0)
        self.disc      = nn.Linear(1, 1, bias=False)
        self.disc.weight.data.fill_(d_init)
        self.disc.weight.requires_grad = not freeze_disc

    def forward(self, x, use_grl, lam=1.0):
        z            = self.w * x
        task_pred    = self.task_head(z)           # straight gradient — no GRL
        z_dom        = GRL.apply(z, lam) if use_grl else z
        domain_logit = self.disc(z_dom)
        return domain_logit, task_pred


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


def run(use_grl, mu=MU, w_init=0.5, lr=0.1, freeze_disc=True, alpha_task=ALPHA_TASK):
    model     = ToyModel(w_init=w_init, freeze_disc=freeze_disc)
    params    = ([model.w, model.task_head.weight]
                 if freeze_disc else list(model.parameters()))
    optimizer = optim.SGD(params, lr=lr)
    bce       = nn.BCEWithLogitsLoss()
    mse       = nn.MSELoss()

    x      = torch.tensor([[-mu / 2], [mu / 2]])
    labels = torch.tensor([[0.0],     [1.0]])

    x_eval, y_eval = make_eval_data(mu)

    w_hist, d_hist, domain_loss_hist, task_loss_hist, acc_hist, lam_hist = [], [], [], [], [], []
    for epoch in range(N_EPOCHS):
        lam = 2.0 / (1.0 + np.exp(-10.0 * epoch / N_EPOCHS)) - 1.0
        domain_logit, task_pred = model(x, use_grl=use_grl, lam=lam)
        domain_loss = bce(domain_logit, labels)
        task_loss   = mse(task_pred, x) if alpha_task > 0 else torch.tensor(0.0)
        loss        = domain_loss + alpha_task * task_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        w_val = model.w.item()
        w_hist.append(w_val)
        d_hist.append(model.disc.weight.item())
        domain_loss_hist.append(domain_loss.item())
        task_loss_hist.append(task_loss.item())
        acc_hist.append(acc_empirical(w_val, x_eval, y_eval))
        lam_hist.append(lam)

    return (np.array(w_hist), np.array(d_hist),
            np.array(domain_loss_hist), np.array(task_loss_hist),
            np.array(acc_hist), np.array(lam_hist))


def make_movie(w_hist, d_hist, lam_hist, mu, label, filename, color):
    x_eval, y_eval = make_eval_data(mu)
    n      = len(w_hist)
    epochs = np.arange(1, n + 1)

    max_abs_w = max(abs(w_hist).max(), 1.0)
    z_lim     = min(max_abs_w * (mu / 2 + 3 * SIGMA_X) * 1.25, 30.0)
    z_range   = np.linspace(-z_lim, z_lim, 600)

    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.6])
    ax_w, ax_dist = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    disc_label = 'trainable d' if d_hist.std() > 1e-6 else f'd frozen={d_hist[0]:.1f}'
    fig.suptitle(f'{label}  (μ={mu}, {disc_label})', fontsize=13, fontweight='bold')

    # w trace (left y-axis)
    ax_w.plot(epochs, w_hist, color='lightgray', lw=1.2, zorder=1)
    ax_w.axhline(0, color='k', ls='--', lw=0.9, alpha=0.4)
    w_trace, = ax_w.plot([], [], color=color,    lw=2.2, zorder=2, label='w')
    w_dot,   = ax_w.plot([], [], 'o', color='red', ms=9, zorder=3)
    pad = max(abs(w_hist).max() * 0.12, 0.3)
    ax_w.set_xlim(0, n + 1); ax_w.set_ylim(w_hist.min() - pad, w_hist.max() + pad)
    ax_w.set_ylabel('w', color=color); ax_w.tick_params(axis='y', labelcolor=color)
    ax_w.set_xlabel('Epoch')

    # d trace (right y-axis)
    ax_d = ax_w.twinx()
    ax_d.plot(epochs, d_hist, color='slategray', lw=1.2, ls=':', zorder=1)
    d_trace, = ax_d.plot([], [], color='slategray', lw=2.0, ls='--', zorder=2, label='d')
    d_pad = max(abs(d_hist).max() * 0.12, 0.3)
    ax_d.set_ylim(d_hist.min() - d_pad, d_hist.max() + d_pad)
    ax_d.set_ylabel('d', color='slategray'); ax_d.tick_params(axis='y', labelcolor='slategray')

    # λ schedule (right axis, offset outward so it doesn't collide with d)
    ax_lam = ax_w.twinx()
    ax_lam.spines['right'].set_position(('outward', 55))
    ax_lam.plot(epochs, lam_hist, color='purple', lw=1.0, ls=':', alpha=0.4, zorder=1)
    lam_trace, = ax_lam.plot([], [], color='purple', lw=1.5, ls=':', zorder=2, label='λ')
    ax_lam.set_ylim(-0.05, 1.15)
    ax_lam.set_ylabel('λ', color='purple')
    ax_lam.tick_params(axis='y', labelcolor='purple')

    lines = [w_trace, d_trace, lam_trace]
    ax_w.legend(lines, ['w', 'd', 'λ'], fontsize=9, loc='upper left')
    ax_w.set_title('w and d over time  (red dot = current w)')
    ax_w.grid(True, alpha=0.25)

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
        d_trace.set_data([], []); lam_trace.set_data([], [])
        src_line.set_data([], []); tgt_line.set_data([], [])
        status_box.set_text('')
        return w_trace, w_dot, d_trace, lam_trace, src_line, tgt_line, status_box

    def update(frame):
        w = w_hist[frame]
        w_trace.set_data(epochs[:frame + 1], w_hist[:frame + 1])
        w_dot.set_data([epochs[frame]], [w])
        d_trace.set_data(epochs[:frame + 1], d_hist[:frame + 1])
        lam_trace.set_data(epochs[:frame + 1], lam_hist[:frame + 1])
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
        return w_trace, w_dot, d_trace, lam_trace, src_line, tgt_line, status_box

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
    w_no,     d_no,     DL_no,     TL_no,     _, lam_no     = run(use_grl=False, freeze_disc=True)

    print("Running GRL (frozen disc) ...")
    w_grl,    d_grl,    DL_grl,    TL_grl,    _, lam_grl    = run(use_grl=True,  freeze_disc=True)

    print("Running No GRL (trainable disc) ...")
    w_no_td,  d_no_td,  DL_no_td,  TL_no_td,  _, lam_no_td  = run(use_grl=False, freeze_disc=False)

    print("Running GRL (trainable disc) ...")
    w_grl_td, d_grl_td, DL_grl_td, TL_grl_td, _, lam_grl_td = run(use_grl=True,  freeze_disc=False)

    runs = [
        ('No GRL  frozen d',    'steelblue',       w_no,     d_no,     DL_no,     TL_no),
        ('GRL     frozen d',    'darkorange',      w_grl,    d_grl,    DL_grl,    TL_grl),
        ('No GRL  trainable d', 'mediumseagreen',  w_no_td,  d_no_td,  DL_no_td,  TL_no_td),
        ('GRL     trainable d', 'crimson',         w_grl_td, d_grl_td, DL_grl_td, TL_grl_td),
    ]

    curves_path = OUTPUT_DIR / 'training_curves.npz'
    np.savez(
        curves_path,
        w_no=w_no,       d_no=d_no,       DL_no=DL_no,       TL_no=TL_no,
        w_grl=w_grl,     d_grl=d_grl,     DL_grl=DL_grl,     TL_grl=TL_grl,
        w_no_td=w_no_td, d_no_td=d_no_td, DL_no_td=DL_no_td, TL_no_td=TL_no_td,
        w_grl_td=w_grl_td, d_grl_td=d_grl_td, DL_grl_td=DL_grl_td, TL_grl_td=TL_grl_td,
    )
    print(f"Saved → {curves_path}")

    fig, axes = plt.subplots(4, len(runs), figsize=(5 * len(runs), 14), sharex=True)
    fig.suptitle(
        f'GRL sanity check  (μ={MU}, σ_x={SIGMA_X}, α_task={ALPHA_TASK})',
        fontsize=12, fontweight='bold')

    for col, (label, color, w_h, d_h, DL_h, TL_h) in enumerate(runs):
        axes[0, col].plot(epochs, w_h, color=color, lw=2)
        axes[0, col].axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        axes[0, col].set_title(label, fontsize=10, fontweight='bold')
        axes[0, col].set_ylabel('w'); axes[0, col].grid(True, alpha=0.25)

        axes[1, col].plot(epochs, d_h, color=color, lw=2)
        axes[1, col].axhline(1, color='k', ls='--', lw=0.8, alpha=0.5, label='d=1 (init)')
        axes[1, col].set_ylabel('d'); axes[1, col].grid(True, alpha=0.25)
        axes[1, col].legend(fontsize=8)

        axes[2, col].plot(epochs, DL_h, color=color, lw=2)
        axes[2, col].axhline(np.log(2), color='k', ls='--', lw=0.8, alpha=0.5,
                              label=f'log(2)≈{np.log(2):.3f}')
        axes[2, col].set_ylabel('Domain loss'); axes[2, col].grid(True, alpha=0.25)
        axes[2, col].legend(fontsize=8)

        axes[3, col].plot(epochs, TL_h, color=color, lw=2)
        axes[3, col].set_ylabel('Task loss (MSE)'); axes[3, col].set_xlabel('Epoch')
        axes[3, col].grid(True, alpha=0.25)

    plt.tight_layout()
    summary_path = OUTPUT_DIR / 'sanity_grl.png'
    plt.savefig(summary_path, dpi=150)
    print(f"Saved → {summary_path}")

    for label, color, w_h, d_h, DL_h, TL_h in runs:
        print(f"  {label:25s}: w={w_h[-1]:+.4f}  d={d_h[-1]:+.4f}"
              f"  domain_loss={DL_h[-1]:.4f}  task_loss={TL_h[-1]:.4f}")

    movie_runs = [
        ('No GRL  frozen d',    w_no,     d_no,     lam_no,     'steelblue',       OUTPUT_DIR / 'no_grl_frozen_d.gif'),
        ('GRL     frozen d',    w_grl,    d_grl,    lam_grl,    'darkorange',      OUTPUT_DIR / 'grl_frozen_d.gif'),
        ('No GRL  trainable d', w_no_td,  d_no_td,  lam_no_td,  'mediumseagreen',  OUTPUT_DIR / 'no_grl_trainable_d.gif'),
        ('GRL     trainable d', w_grl_td, d_grl_td, lam_grl_td, 'crimson',         OUTPUT_DIR / 'grl_trainable_d.gif'),
    ]
    for label, w_h, d_h, lam_h, color, fname in movie_runs:
        print(f"\nMaking movie: {fname} ({N_MOVIE} epochs) ...")
        make_movie(w_h[:N_MOVIE], d_h[:N_MOVIE], lam_h[:N_MOVIE], mu=MU, label=label, filename=fname, color=color)

    print("\nDone.")


if __name__ == "__main__":
    main()
