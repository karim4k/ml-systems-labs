import random
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

from nn import load_data, MLP, cross_entropy, backward

random.seed(42)
np.random.seed(42)

LRS = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
EPOCHS = 50
BATCH = 32
STD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
TAG = f"std{STD}"
X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_data()


def accuracy(net, X, y):
    P, _ = net.forward(X)
    return float((P.argmax(axis=1) == y).mean())


def train(width, lr):
    net = MLP(width=width, seed=42, std=STD)
    rng = np.random.RandomState(42)
    log = []
    n = X_tr.shape[0]
    for epoch in range(1, EPOCHS + 1):
        idx = rng.permutation(n)                      # shuffle every epoch
        losses = []
        for s in range(0, n, BATCH):
            b = idx[s:s + BATCH]
            P, cache = net.forward(X_tr[b])
            losses.append(cross_entropy(P, Y_tr[b]))
            g = backward(net, cache, Y_tr[b])
            for k in net.params:
                net.params[k] -= lr * g[k]
        P_te, _ = net.forward(X_te)
        log.append({
            "width": width, "lr": lr, "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "test_loss": float(cross_entropy(P_te, Y_te)),
            "test_acc": accuracy(net, X_te, y_te),
        })
        print(f"W={width} lr={lr} epoch {epoch:2d} "
              f"train_loss={log[-1]['train_loss']:.4f} "
              f"test_acc={log[-1]['test_acc']:.4f}")
    return log


all_logs = []
for w in WIDTHS:
    for lr in LRS:
        all_logs += train(w, lr)

with open("lab02/results/training_log.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=all_logs[0].keys())
    wr.writeheader()
    wr.writerows(all_logs)

final = [r for r in all_logs if r["epoch"] == EPOCHS]
with open("lab02/results/final_accuracy.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["width", "lr", "final_train_loss", "final_test_acc"])
    for r in final:
        wr.writerow([r["width"], r["lr"], f"{r['train_loss']:.4f}",
                     f"{r['test_acc']:.4f}"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, w in zip(axes, WIDTHS):
    for lr in LRS:
        rows = [r for r in all_logs if r["width"] == w and r["lr"] == lr]
        ax.plot([r["epoch"] for r in rows],
                [r["train_loss"] for r in rows], label=f"lr={lr}")
    ax.set_title(f"Hidden width {w}")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Training loss (cross-entropy)")
axes[0].legend()
plt.tight_layout()
plt.savefig("lab02/results/loss_curves.png", dpi=150)

print("\nFINAL RESULTS")
for r in final:
    print(f"W={r['width']:2d} lr={r['lr']:<5} "
          f"train_loss={r['train_loss']:.4f} test_acc={r['test_acc']:.4f}")