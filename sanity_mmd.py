"""
sanity_mmd.py

Domain adaptation sanity check:
  - Encoder (MLP: x → z) + Predictor (z → y, regression)
  - Task loss: MSE on source training data only
  - MMD alignment: minimize MMD(z_src_train, z_tgt_train) directly (no GRL)
  - Evaluate: source val MSE and held-out target test MSE
  - Optuna hyperparameter search; objective = best source val MSE (no tgt leakage)

Data splits:
  src_train (200): x ~ N(-5, 0.5),  y = x² + noise  — task loss + MMD
  tgt_train (200): x ~ N(+5, 0.5),  unlabeled        — MMD only
  src_val   (100): x ~ N(-5, 0.5),  y = x² + noise  — source evaluation
  tgt_test  (100): x ~ N(+5, 0.5),  y = x² + noise  — held-out target evaluation

NOTE: y = x² so P_src(y) ≈ P_tgt(y) — covariate shift only.
  With y = 2x, label distributions differ and alignment hurts prediction.

Latent space split:
  z = [z_task (dim task_dim) | z_nuisance (dim nuisance_dim)]
  - z_task:     used for prediction + MMD alignment
  - z_nuisance: junk drawer with domain classifier pulling domain info in

Loss: task_loss + lam_mmd * MMD(z_task_src, z_task_tgt)
    + lam_domain * BCE(domain_clf(z_nuisance), domain_label)

Optuna objective: best source val_mse (NOT tgt_mse — no leakage)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ks_2samp
import optuna


# ---------------------------------------------------------------------------
# MMD loss
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
    return mmd


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
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


class DomainClassifier(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_split(mu, n, std=0.5, noise_std=0.2, rng=None):
    x_task     = rng.normal(0,   std, (n, 1)).astype(np.float32)
    x_nuisance = rng.normal(mu, 1.0, (n, 1)).astype(np.float32)
    x = np.concatenate([x_task, x_nuisance], axis=1)
    y = (x_task ** 2 + noise_std * rng.standard_normal((n, 1))).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_data(seed=0, n_train=200, n_val=100):
    """Generate and standardize all data splits from a single seed."""
    rng = np.random.default_rng(seed)
    src_x_train, src_y_train = make_split(-5.0, n_train, rng=rng)
    src_x_val,   src_y_val   = make_split(-5.0, n_val,   rng=rng)
    tgt_x_train, _           = make_split( 5.0, n_train, rng=rng)
    tgt_x_test,  tgt_y_test  = make_split( 5.0, n_val,   rng=rng)

    # Standardize using source train stats only (no leakage)
    x_mean = src_x_train.mean(dim=0, keepdim=True)
    x_std  = src_x_train.std(dim=0,  keepdim=True).clamp(min=1e-6)
    src_x_train = (src_x_train - x_mean) / x_std
    src_x_val   = (src_x_val   - x_mean) / x_std
    tgt_x_train = (tgt_x_train - x_mean) / x_std
    tgt_x_test  = (tgt_x_test  - x_mean) / x_std

    y_mean = src_y_train.mean()
    y_std  = src_y_train.std().clamp(min=1e-6)
    src_y_train = (src_y_train - y_mean) / y_std
    src_y_val   = (src_y_val   - y_mean) / y_std
    tgt_y_test  = (tgt_y_test  - y_mean) / y_std

    return dict(
        src_x_train=src_x_train, src_y_train=src_y_train,
        src_x_val=src_x_val,     src_y_val=src_y_val,
        tgt_x_train=tgt_x_train,
        tgt_x_test=tgt_x_test,   tgt_y_test=tgt_y_test,
        n_train=n_train,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_once(cfg, data, model_seed=42, snap_every=None):
    """
    Train with given config dict.
    Returns (best_val_mse, hist, models, snapshots).

    snap_every=None disables snapshot collection (used during Optuna search
    to avoid memory overhead). tgt_test is only used inside torch.no_grad()
    for diagnostics — never touches the optimizer or early-stopping criterion.
    """
    torch.manual_seed(model_seed)

    task_dim     = cfg['task_dim']
    nuisance_dim = cfg['nuisance_dim']
    hidden_dim   = cfg['hidden_dim']
    lr           = cfg['lr']
    lam_mmd      = cfg['lam_mmd']
    lam_domain   = cfg['lam_domain']
    patience     = cfg['patience']
    n_epochs     = cfg.get('n_epochs', 10000)
    min_delta    = cfg.get('min_delta', 1e-4)

    n_train = data['n_train']

    encoder    = Encoder(input_dim=2, hidden_dim=hidden_dim,
                         latent_dim=task_dim + nuisance_dim)
    predictor  = Predictor(latent_dim=task_dim)
    domain_clf = DomainClassifier(latent_dim=nuisance_dim,
                                  hidden_dim=max(4, hidden_dim // 2))
    optimizer  = torch.optim.Adam(
        list(encoder.parameters()) +
        list(predictor.parameters()) +
        list(domain_clf.parameters()),
        lr=lr,
    )
    mse_fn = nn.MSELoss()

    src_domain_labels = torch.zeros(n_train, 1)
    tgt_domain_labels = torch.ones(n_train,  1)

    best_val_mse      = float('inf')
    epochs_no_improve = 0

    train_mse_hist  = []
    val_mse_hist    = []
    tgt_mse_hist    = []
    mmd_hist        = []
    ks_hist         = []
    domain_acc_hist = []
    snapshots       = []

    for epoch in range(n_epochs):
        encoder.train(); predictor.train(); domain_clf.train()

        z_src = encoder(data['src_x_train'])
        z_tgt = encoder(data['tgt_x_train'])

        z_task_src     = z_src[:, :task_dim]
        z_task_tgt     = z_tgt[:, :task_dim]
        z_nuisance_src = z_src[:, task_dim:]
        z_nuisance_tgt = z_tgt[:, task_dim:]

        task_loss  = mse_fn(predictor(z_task_src), data['src_y_train'])
        domain_mmd = mmd_loss(z_task_src, z_task_tgt)

        z_nuisance_all    = torch.cat([z_nuisance_src, z_nuisance_tgt], dim=0)
        domain_labels_all = torch.cat([src_domain_labels, tgt_domain_labels], dim=0)
        domain_logits     = domain_clf(z_nuisance_all)
        domain_clf_loss   = F.binary_cross_entropy_with_logits(
            domain_logits, domain_labels_all)

        total_loss = task_loss + lam_mmd * domain_mmd + lam_domain * domain_clf_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        encoder.eval(); predictor.eval(); domain_clf.eval()
        with torch.no_grad():
            val_mse = mse_fn(
                predictor(encoder(data['src_x_val'])[:, :task_dim]),
                data['src_y_val'],
            ).item()
            tgt_mse = mse_fn(
                predictor(encoder(data['tgt_x_test'])[:, :task_dim]),
                data['tgt_y_test'],
            ).item()

            z_s = z_task_src.detach().numpy()
            z_t = z_task_tgt.detach().numpy()
            ks_vals = [ks_2samp(z_s[:, d], z_t[:, d])[0]
                       for d in range(z_s.shape[1])]
            ks = max(ks_vals)

            domain_preds = (torch.sigmoid(domain_logits) > 0.5).float()
            domain_acc   = (domain_preds == domain_labels_all).float().mean().item()

        train_mse_hist.append(task_loss.item())
        val_mse_hist.append(val_mse)
        tgt_mse_hist.append(tgt_mse)
        mmd_hist.append(domain_mmd.item())
        ks_hist.append(ks)
        domain_acc_hist.append(domain_acc)

        if snap_every is not None and epoch % snap_every == 0:
            snapshots.append((epoch, z_s.copy(), z_t.copy()))

        if val_mse < best_val_mse - min_delta:
            best_val_mse      = val_mse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    hist = dict(
        train_mse=train_mse_hist, val_mse=val_mse_hist, tgt_mse=tgt_mse_hist,
        mmd=mmd_hist, ks=ks_hist, domain_acc=domain_acc_hist,
    )
    return best_val_mse, hist, (encoder, predictor, domain_clf), snapshots


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial, data, beta_obj=0.5):
    """
    Objective = val_mse + beta_obj * final_KS

    val_mse:   source generalisation (labelled signal)
    final_KS:  z_task alignment at convergence (label-free proxy for target generalisation)

    final_KS is the *outcome* of alignment after training — it is NOT the MMD
    loss term that the optimiser directly minimises, so it is not circular.
    Different hyperparameter configs converge to different KS values; this
    penalises configs where z_task is still misaligned after training stops.
    """
    cfg = {
        'lr':           trial.suggest_float('lr',          1e-4, 1e-2,  log=True),
        'lam_mmd':      trial.suggest_float('lam_mmd',     0.01, 10.0,  log=True),
        'lam_domain':   trial.suggest_float('lam_domain',  0.01, 10.0,  log=True),
        'patience':     trial.suggest_int  ('patience',    100,  1000),
        'hidden_dim':   trial.suggest_categorical('hidden_dim',   [32, 64, 128]),
        'task_dim':     trial.suggest_categorical('task_dim',     [2, 4]),
        'nuisance_dim': trial.suggest_categorical('nuisance_dim', [2, 4]),
        'n_epochs':  10000,
        'min_delta': 1e-4,
    }
    best_val_mse, hist, _, _ = train_once(cfg, data, model_seed=42, snap_every=None)
    final_ks = float(np.mean(hist['ks'][-50:]))   # average last 50 epochs for stability
    return best_val_mse + beta_obj * final_ks


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(hist, cfg, data, models, snapshots):
    encoder, predictor, _ = models
    task_dim     = cfg['task_dim']
    nuisance_dim = cfg['nuisance_dim']
    snap_every   = cfg.get('snap_every', 50)

    epochs = np.arange(len(hist['train_mse']))

    # Training curves
    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)

    axes[0].plot(epochs, hist['train_mse'], label="train MSE (src)",
                 color="steelblue", linewidth=1.2)
    axes[0].plot(epochs, hist['val_mse'],   label="val MSE (src)",
                 color="navy",      linewidth=1.2, linestyle="--")
    axes[0].plot(epochs, hist['tgt_mse'],   label="test MSE (tgt)",
                 color="tomato",    linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("MSE")
    param_str = (
        f"lr={cfg['lr']:.5f}  lam_mmd={cfg['lam_mmd']:.3f}  "
        f"lam_dom={cfg['lam_domain']:.3f}  patience={cfg['patience']}\n"
        f"task_dim={task_dim}  nuisance_dim={nuisance_dim}  hidden={cfg['hidden_dim']}"
    )
    axes[0].set_title(f"Best Optuna config\n{param_str}")
    axes[0].legend()

    axes[1].plot(epochs, hist['mmd'], color="purple", linewidth=1.2)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("MMD (z_task)")

    axes[2].plot(epochs, hist['ks'], color="tomato", linewidth=1.2)
    axes[2].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("KS (z_task)")

    axes[3].plot(epochs, hist['domain_acc'], color="green", linewidth=1.2)
    axes[3].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance (0.5)")
    axes[3].axhline(1.0, color="gray", linestyle=":",  linewidth=0.8, label="perfect (1.0)")
    axes[3].set_ylabel("Domain acc (z_nuisance)")
    axes[3].set_xlabel("Epoch")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig("mmd_da_training.png", dpi=150)
    plt.close()
    print("Saved mmd_da_training.png")

    # Final latent distributions
    encoder.eval()
    with torch.no_grad():
        z_s_final = encoder(data['src_x_train']).numpy()
        z_t_final = encoder(data['tgt_x_train']).numpy()

    if task_dim >= 2 and nuisance_dim >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(z_s_final[:, 0], z_s_final[:, 1],
                    alpha=0.5, s=10, label="source", color="steelblue")
        ax1.scatter(z_t_final[:, 0], z_t_final[:, 1],
                    alpha=0.5, s=10, label="target",  color="tomato")
        ax1.set_xlabel("z₀"); ax1.set_ylabel("z₁")
        ax1.set_title(f"z_task  |  KS={hist['ks'][-1]:.3f}")
        ax1.legend()

        ax2.scatter(z_s_final[:, task_dim],   z_s_final[:, task_dim + 1],
                    alpha=0.5, s=10, label="source", color="steelblue")
        ax2.scatter(z_t_final[:, task_dim],   z_t_final[:, task_dim + 1],
                    alpha=0.5, s=10, label="target",  color="tomato")
        ax2.set_xlabel(f"z_{task_dim}"); ax2.set_ylabel(f"z_{task_dim + 1}")
        ax2.set_title("z_nuisance (junk drawer)")
        ax2.legend()

        plt.tight_layout()
        plt.savefig("mmd_da_final_z.png", dpi=150)
        plt.close()
        print("Saved mmd_da_final_z.png")

    # z_task[:, 0] vs x_task diagnostic
    # x_task is input dim 0 (the dimension y depends on).
    # If the encoder is consistent across domains, the same x_task value should
    # produce the same z_task[0] regardless of domain — the two clouds should
    # trace the same curve. A systematic offset means x_nuisance is leaking
    # into z_task and the predictor will fail on target even when KS ≈ 0.
    x_task_src = data['src_x_train'][:, 0].numpy()
    x_task_tgt = data['tgt_x_train'][:, 0].numpy()
    z_task_src = z_s_final[:, 0]
    z_task_tgt = z_t_final[:, 0]

    # sort by x_task so a trend line is visible
    s_ord = np.argsort(x_task_src)
    t_ord = np.argsort(x_task_tgt)

    fig, axes = plt.subplots(1, task_dim, figsize=(6 * task_dim, 5))
    if task_dim == 1:
        axes = [axes]
    for d in range(task_dim):
        s_ord = np.argsort(x_task_src)
        t_ord = np.argsort(x_task_tgt)
        axes[d].scatter(x_task_src, z_s_final[:, d],
                        alpha=0.4, s=10, color="steelblue", label="source")
        axes[d].scatter(x_task_tgt, z_t_final[:, d],
                        alpha=0.4, s=10, color="tomato",    label="target")
        axes[d].plot(x_task_src[s_ord], z_s_final[s_ord, d],
                     color="steelblue", linewidth=0.8, alpha=0.6)
        axes[d].plot(x_task_tgt[t_ord], z_t_final[t_ord, d],
                     color="tomato",    linewidth=0.8, alpha=0.6)
        axes[d].set_xlabel("x_task (standardised)")
        axes[d].set_ylabel(f"z_task[{d}]")
        axes[d].set_title(
            f"z_task[{d}] vs x_task\n"
            f"overlapping curves → consistent encoding across domains\n"
            f"offset curves → x_nuisance leaking into z_task"
        )
        axes[d].legend()

    plt.tight_layout()
    plt.savefig("mmd_da_ztask_vs_xtask.png", dpi=150)
    plt.close()
    print("Saved mmd_da_ztask_vs_xtask.png")

    # Scatter animation over training
    if snapshots and task_dim >= 2:
        all_z = np.concatenate([np.vstack([zs, zt]) for _, zs, zt in snapshots])
        z0_min, z0_max = all_z[:, 0].min() - 0.3, all_z[:, 0].max() + 0.3
        z1_min, z1_max = all_z[:, 1].min() - 0.3, all_z[:, 1].max() + 0.3

        fig, ax = plt.subplots(figsize=(6, 6))

        def update(frame_idx):
            ax.clear()
            epoch, z_s, z_t = snapshots[frame_idx]
            ax.scatter(z_s[:, 0], z_s[:, 1], alpha=0.5, s=8,
                       label="source (z_task)", color="steelblue")
            ax.scatter(z_t[:, 0], z_t[:, 1], alpha=0.5, s=8,
                       label="target (z_task)",  color="tomato")
            ax.set_xlim(z0_min, z0_max); ax.set_ylim(z1_min, z1_max)
            ax.set_xlabel("z₀"); ax.set_ylabel("z₁")
            ei = epoch   # epoch == index into hist arrays
            if ei < len(hist['ks']):
                ax.set_title(
                    f"Latent z — epoch {epoch}  |  "
                    f"KS={hist['ks'][ei]:.3f}  MMD={hist['mmd'][ei]:.4f}  "
                    f"tgt_mse={hist['tgt_mse'][ei]:.4f}"
                )
            ax.legend(loc="upper right")

        ani = animation.FuncAnimation(
            fig, update, frames=len(snapshots), interval=150, repeat=True)
        ani.save("mmd_da_animation.gif", writer="pillow", fps=8)
        plt.close()
        print("Saved mmd_da_animation.gif")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_trials = 50
    beta_obj = 0.5   # weight on final_KS in objective; both terms ~O(1) on normalised data

    # Generate data once — all Optuna trials see the same data.
    # tgt_test is never used in the objective (no leakage).
    data = make_data(seed=0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    print(f"Running Optuna search ({n_trials} trials)...")
    print(f"Objective = val_mse + {beta_obj} * final_KS(z_task)")
    study.optimize(
        lambda trial: objective(trial, data, beta_obj=beta_obj),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    obj_label = f"val_mse + {beta_obj}*KS"
    print(f"\nBest trial #{study.best_trial.number}  {obj_label}={study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    sorted_trials = sorted(study.trials, key=lambda t: t.value)
    print("\nTop 5 trials:")
    print(f"  {'#':>4}  {obj_label:>18}  params")
    for t in sorted_trials[:5]:
        param_str = "  ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"  {t.number:>4}  {t.value:>18.4f}  {param_str}")

    # Optuna history plot
    trial_vals  = [t.value for t in study.trials]
    best_so_far = np.minimum.accumulate(trial_vals)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(range(len(trial_vals)), trial_vals,
               s=15, alpha=0.6, color="steelblue", label=obj_label)
    ax.plot(range(len(best_so_far)), best_so_far,
            color="tomato", linewidth=2, label="best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel(obj_label)
    ax.set_title("Optuna optimization history")
    ax.legend()
    plt.tight_layout()
    plt.savefig("mmd_da_optuna_history.png", dpi=150)
    plt.close()
    print("Saved mmd_da_optuna_history.png")

    # Final training with best config + snapshots for all plots
    best_cfg = {
        **study.best_params,
        'n_epochs':   10000,
        'min_delta':  1e-4,
        'snap_every': 50,
    }
    print("\nFinal training with best config...")
    best_val_mse, hist, models, snapshots = train_once(
        best_cfg, data, model_seed=42, snap_every=50)
    print(
        f"Final  best_val_mse={best_val_mse:.4f}"
        f"  tgt_mse(last)={hist['tgt_mse'][-1]:.4f}"
        f"  stopped at epoch {len(hist['train_mse'])}"
    )

    plot_results(hist, best_cfg, data, models, snapshots)


if __name__ == "__main__":
    main()
