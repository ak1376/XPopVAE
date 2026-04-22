import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. Data generation ────────────────────────────────────────────────────────
N_POINTS  = 16
N_SAMPLES = 600
NOISE_STD = 0.05

def make_dataset(n=N_SAMPLES, wave="sine", noise=NOISE_STD):
    t = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    X, Y, D = [], [], []
    domain = 0 if wave == "sine" else 1
    for _ in range(n):
        amp = np.random.uniform(0.5, 2.0)
        label = 0 if amp < 1.0 else (1 if amp < 1.5 else 2)
        x = amp * (np.sin(t) if wave == "sine" else np.cos(t))
        x += np.random.randn(N_POINTS) * noise
        X.append(x); Y.append(label); D.append(domain)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    D = torch.tensor(D, dtype=torch.long)
    return X, Y, D


# ── 2. Gradient Reversal Layer ────────────────────────────────────────────────
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = torch.tensor(0.0)

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambda_)

    def set_lambda(self, val):
        self.lambda_ = torch.tensor(float(val))


# ── 3. Network ────────────────────────────────────────────────────────────────
class GRLNet(nn.Module):
    def __init__(self, input_dim=N_POINTS, hidden_dim=64, n_classes=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.grl = GradientReversal()
        # Slightly stronger domain classifier so it's a real adversary
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        features      = self.feature_extractor(x)
        task_logits   = self.task_classifier(features)
        domain_logits = self.domain_classifier(self.grl(features))
        return task_logits, domain_logits


# ── 4. Lambda schedule ────────────────────────────────────────────────────────
# Capped at max_lambda < 1.0 — prevents adversarial pressure from
# overwhelming the task signal on this small toy problem.
def get_lambda(epoch, total_epochs, max_lambda=0.5):
    p = epoch / total_epochs
    raw = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
    return raw * max_lambda


# ── 5. Training ───────────────────────────────────────────────────────────────
def train(epochs=400, batch_size=64, lr=1e-3, domain_loss_weight=0.3):
    """
    domain_loss_weight: scales the domain loss relative to the task loss.
    Keeping this below 1.0 stops the adversary from drowning the task signal.
    Try values between 0.1 and 1.0 to see the effect.
    """
    Xs, Ys, Ds = make_dataset(wave="sine")
    Xt, Yt, Dt = make_dataset(wave="cosine")

    src_loader = DataLoader(TensorDataset(Xs, Ys, Ds), batch_size=batch_size, shuffle=True,  drop_last=True)
    tgt_loader = DataLoader(TensorDataset(Xt, Yt, Dt), batch_size=batch_size, shuffle=True,  drop_last=True)

    model          = GRLNet()
    optimizer      = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    task_criterion = nn.CrossEntropyLoss()
    dom_criterion  = nn.CrossEntropyLoss()

    history = {k: [] for k in ["epoch", "task_loss", "dom_loss",
                                "src_acc", "tgt_acc", "dom_disc", "lambda"]}

    for epoch in range(1, epochs + 1):
        model.train()
        lam = get_lambda(epoch, epochs)
        model.grl.set_lambda(lam)

        epoch_task, epoch_dom, n = 0.0, 0.0, 0

        for (xs, ys, ds), (xt, _, dt) in zip(src_loader, tgt_loader):
            optimizer.zero_grad()

            task_logits, dom_src = model(xs)
            loss_task = task_criterion(task_logits, ys)

            _, dom_tgt = model(xt)
            loss_dom = dom_criterion(dom_src, ds) + dom_criterion(dom_tgt, dt)

            loss = loss_task + domain_loss_weight * loss_dom
            loss.backward()

            # Gradient clipping keeps training stable
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_task += loss_task.item()
            epoch_dom  += loss_dom.item()
            n += 1

        acc = evaluate(model, Xs, Ys, Xt, Yt)
        history["epoch"].append(epoch)
        history["task_loss"].append(epoch_task / n)
        history["dom_loss"].append(epoch_dom / n)
        history["src_acc"].append(acc["src"])
        history["tgt_acc"].append(acc["tgt"])
        history["dom_disc"].append(acc["dom"])
        history["lambda"].append(lam)

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | λ={lam:.3f} | "
                f"task={epoch_task/n:.3f}  dom={epoch_dom/n:.3f} | "
                f"src={acc['src']:.1f}%  tgt={acc['tgt']:.1f}%  dom_disc={acc['dom']:.1f}%"
            )

    return model, history


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, Xs, Ys, Xt, Yt):
    model.eval()
    with torch.no_grad():
        task_logits, _ = model(Xs)
        src_acc = (task_logits.argmax(1) == Ys).float().mean().item() * 100

        task_logits_t, _ = model(Xt)
        tgt_acc = (task_logits_t.argmax(1) == Yt).float().mean().item() * 100

        Ds    = torch.zeros(len(Xs), dtype=torch.long)
        Dt    = torch.ones(len(Xt),  dtype=torch.long)
        X_all = torch.cat([Xs, Xt])
        D_all = torch.cat([Ds, Dt])
        _, dom_logits = model(X_all)
        dom_acc = (dom_logits.argmax(1) == D_all).float().mean().item() * 100

    return {"src": src_acc, "tgt": tgt_acc, "dom": dom_acc}


# ── 7. Plots ──────────────────────────────────────────────────────────────────
def plot_history(history, save_path="training_history.png"):
    epochs = history["epoch"]
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("GRL Training History", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1 — Losses
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["task_loss"], label="Task loss",   color="#185FA5", linewidth=1.8)
    ax1.plot(epochs, history["dom_loss"],  label="Domain loss", color="#D85A30", linewidth=1.8)
    ax1.set_title("Training losses"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # Panel 2 — Task accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["src_acc"], label="Source (sine)",   color="#185FA5", linewidth=1.8)
    ax2.plot(epochs, history["tgt_acc"], label="Target (cosine)", color="#D85A30", linewidth=1.8)
    ax2.axhline(33.3, color="gray", linestyle="--", linewidth=1, label="Random baseline (3 classes)")
    ax2.set_title("Task accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Panel 3 — Domain discriminability
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history["dom_disc"], color="#3B6D11", linewidth=1.8)
    ax3.axhline(50, color="gray", linestyle="--", linewidth=1, label="Ideal: 50% = fully aligned")
    ax3.axhspan(40, 60, alpha=0.08, color="green", label="Good alignment zone")
    ax3.set_title("Domain discriminability"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("Domain clf accuracy (%)")
    ax3.set_ylim(0, 105); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    # Panel 4 — Lambda schedule
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history["lambda"], color="#533AB7", linewidth=1.8)
    ax4.set_title("Lambda (GRL strength)"); ax4.set_xlabel("Epoch"); ax4.set_ylabel("λ")
    ax4.grid(alpha=0.3)

    # Panel 5 — Source vs target gap
    ax5 = fig.add_subplot(gs[1, 1])
    gap = [s - t for s, t in zip(history["src_acc"], history["tgt_acc"])]
    ax5.plot(epochs, gap, color="#BA7517", linewidth=1.8)
    ax5.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax5.fill_between(epochs, gap, 0, alpha=0.08, color="#BA7517")
    ax5.set_title("Accuracy gap (source − target)"); ax5.set_xlabel("Epoch"); ax5.set_ylabel("Gap (pp)")
    ax5.grid(alpha=0.3)

    # Panel 6 — Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    final_src  = history["src_acc"][-1]
    final_tgt  = history["tgt_acc"][-1]
    final_dom  = history["dom_disc"][-1]
    best_tgt   = max(history["tgt_acc"])
    best_epoch = history["epoch"][history["tgt_acc"].index(best_tgt)]
    summary = (
        f"Final results\n"
        f"─────────────────────\n"
        f"Source accuracy : {final_src:.1f}%\n"
        f"Target accuracy : {final_tgt:.1f}%\n"
        f"Domain disc     : {final_dom:.1f}%\n"
        f"  (ideal = 50%)\n\n"
        f"Best target acc : {best_tgt:.1f}%\n"
        f"  at epoch {best_epoch}\n\n"
        f"Accuracy gap    : {final_src - final_tgt:.1f} pp"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.5))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")
    plt.show()


# ── 8. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("Training GRL network: sine (source) → cosine (target)")
    print("Task: classify amplitude as low / mid / high\n")

    model, history = train(
        epochs=400,
        batch_size=64,
        lr=1e-3,
        domain_loss_weight=0.3,   # try 0.1–1.0 to tune adversarial pressure
    )

    print("\nFinal evaluation on fresh held-out data:")
    Xs, Ys, _ = make_dataset(wave="sine",   n=1000)
    Xt, Yt, _ = make_dataset(wave="cosine", n=1000)
    acc = evaluate(model, Xs, Ys, Xt, Yt)
    print(f"  Source (sine)   task accuracy : {acc['src']:.1f}%")
    print(f"  Target (cosine) task accuracy : {acc['tgt']:.1f}%")
    print(f"  Domain discriminability       : {acc['dom']:.1f}%  (50% = perfectly aligned)")

    plot_history(history)