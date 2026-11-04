import csv
import random
import statistics
import time

import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_moons

from engine import Value, vsum

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

EPOCHS = 30      
LR = 0.5
H = 8

X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
rng = np.random.RandomState(42)
W1_0 = rng.randn(H, 2) * 0.5
b1_0 = np.zeros(H)
W2_0 = rng.randn(1, H) * 0.5
b2_0 = np.zeros(1)
Xl, yl = X.tolist(), y.tolist()


def accuracy(W1, b1, W2, b2):
    h = np.maximum(0, X @ W1.T + b1)
    p = 1 / (1 + np.exp(-(h @ W2.T + b2)))
    return float(((p[:, 0] > 0.5).astype(int) == y).mean())


def train_engine(epochs):
    W1 = [[Value(v) for v in row] for row in W1_0]
    b1 = [Value(v) for v in b1_0]
    w2 = [Value(v) for v in W2_0[0]]
    b2 = Value(b2_0[0])
    params = [p for row in W1 for p in row] + b1 + w2 + [b2]
    losses = []
    for _ in range(epochs):
        for p in params:
            p.grad = 0.0
        terms = []
        for xi, yi in zip(Xl, yl):
            hs = [vsum([W1[j][0] * xi[0], W1[j][1] * xi[1], b1[j]]).relu()
                  for j in range(H)]
            p = vsum([w2[j] * hs[j] for j in range(H)] + [b2]).sigmoid()
            terms.append(-(yi * (p + 1e-12).log() + (1 - yi) * ((1 - p) + 1e-12).log()))
        loss = vsum(terms) * (1.0 / len(Xl))
        loss.backward()
        for p in params:
            p.data -= LR * p.grad
        losses.append(loss.data)
    arrs = (np.array([[W1[j][0].data, W1[j][1].data] for j in range(H)]),
            np.array([v.data for v in b1]),
            np.array([[v.data for v in w2]]),
            np.array([b2.data]))
    return losses, accuracy(*arrs)


def train_torch(epochs):
    model = nn.Sequential(nn.Linear(2, H), nn.ReLU(), nn.Linear(H, 1), nn.Sigmoid()).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1_0)); model[0].bias.copy_(torch.tensor(b1_0))
        model[2].weight.copy_(torch.tensor(W2_0)); model[2].bias.copy_(torch.tensor(b2_0))
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    lossf = nn.BCELoss()
    Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = float(((model(Xt) > 0.5).squeeze(1).long().numpy() == y).mean())
    return losses, acc


def timed(fn):
    fn(1)                                   # warm-up
    times, last = [], None
    for _ in range(5):
        t0 = time.perf_counter()
        last = fn(EPOCHS)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), last


t_e, (loss_e, acc_e) = timed(train_engine)
t_t, (loss_t, acc_t) = timed(train_torch)

lines = [
    f"Epochs={EPOCHS}, lr={LR}, hidden={H}, full-batch gradient descent, float64",
    f"Engine : median time {t_e:.3f} s, final loss {loss_e[-1]:.6f}, accuracy {acc_e:.4f}",
    f"PyTorch: median time {t_t:.3f} s, final loss {loss_t[-1]:.6f}, accuracy {acc_t:.4f}",
    f"Engine / PyTorch time ratio: {t_e / t_t:.1f}x",
    f"Max |loss_engine - loss_torch| over epochs: "
    f"{max(abs(a - b) for a, b in zip(loss_e, loss_t)):.3e}",
]
text = "\n".join(lines)
print(text)
with open("lab05/results/moons_timing.txt", "w", encoding="utf-8") as fh:
    fh.write(text)

with open("lab05/results/moons_losses.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["epoch", "engine_loss", "torch_loss"])
    for i, (a, b) in enumerate(zip(loss_e, loss_t), 1):
        w.writerow([i, a, b])