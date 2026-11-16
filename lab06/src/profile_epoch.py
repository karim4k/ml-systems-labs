import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from common import get_loaders, make_cnn, set_seeds

WORKERS = [0, 1, 2, 4, 8]
LABELS = ["data_loading", "forward", "backward", "optimizer_step"]


def profile_epoch(num_workers, save_table=None):
    set_seeds(42)
    train_loader, _, _ = get_loaders(128, num_workers)
    model = make_cnn()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # warm-up on random data (not counted)
    for _ in range(5):
        xw = torch.randn(128, 1, 28, 28)
        yw = torch.randint(0, 10, (128,))
        loss_fn(model(xw), yw).backward()
        opt.step()
        opt.zero_grad()

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        it = iter(train_loader)          # worker start-up is part of the epoch
        while True:
            with record_function("data_loading"):
                try:
                    x, y = next(it)
                except StopIteration:
                    break
            with record_function("forward"):
                loss = loss_fn(model(x), y)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer_step"):
                opt.step()
                opt.zero_grad()
    wall = time.perf_counter() - t0

    times = {}
    for e in prof.key_averages():
        if e.key in LABELS:
            times[e.key] = e.cpu_time_total / 1000.0      # microseconds -> ms
    if save_table:
        with open(save_table, "w", encoding="utf-8") as f:
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    return wall, times


def main():
    rows = []
    for w in WORKERS:
        wall, t = profile_epoch(w, save_table="lab06/results/profile_table_w0.txt" if w == 0 else None)
        total = sum(t.get(k, 0.0) for k in LABELS)
        row = {"num_workers": w, "epoch_wall_s": round(wall, 2)}
        for k in LABELS:
            row[f"{k}_ms"] = round(t.get(k, 0.0), 1)
            row[f"{k}_share"] = round(t.get(k, 0.0) / total, 3)
        rows.append(row)
        print(row)

    with open("lab06/results/workers_profile.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xs = [str(r["num_workers"]) for r in rows]
    bottom = [0.0] * len(rows)
    for k in LABELS:
        vals = [r[f"{k}_share"] for r in rows]
        ax[0].bar(xs, vals, bottom=bottom, label=k)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax[0].set_xlabel("num_workers")
    ax[0].set_ylabel("Share of profiled time")
    ax[0].legend()
    ax[1].bar(xs, [r["epoch_wall_s"] for r in rows])
    ax[1].set_xlabel("num_workers")
    ax[1].set_ylabel("Epoch wall time (s)")
    plt.tight_layout()
    plt.savefig("lab06/results/workers_profile.png", dpi=150)


if __name__ == "__main__":
    main()