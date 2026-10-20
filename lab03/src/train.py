import csv
import time

import torch
from torch import nn

from common import MODELS, get_loaders, set_seeds

EPOCHS = 5


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


log_rows, final_rows = [], []
for name, make in MODELS.items():
    set_seeds(42)
    train_loader, val_loader, test_loader = get_loaders(128)
    model = make()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    t_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        val_acc = evaluate(model, val_loader)
        log_rows.append({"model": name, "epoch": epoch,
                         "train_loss": round(total_loss / n, 4),
                         "val_acc": round(val_acc, 4)})
        print(f"{name} epoch {epoch} loss={total_loss / n:.4f} val_acc={val_acc:.4f}")
    train_time = time.perf_counter() - t_start

    test_acc = evaluate(model, test_loader)
    final_rows.append({"model": name, "params": n_params,
                       "val_acc": log_rows[-1]["val_acc"],
                       "test_acc": round(test_acc, 4),
                       "train_time_s": round(train_time, 1)})
    print(final_rows[-1])

with open("lab03/results/training_log.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    w.writeheader()
    w.writerows(log_rows)

with open("lab03/results/accuracy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=final_rows[0].keys())
    w.writeheader()
    w.writerows(final_rows)