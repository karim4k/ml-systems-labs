import random
import time
import statistics
import csv
import numpy as np

from nn import load_data, MLP, cross_entropy, backward

random.seed(42)
np.random.seed(42)

X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_data()
Xb, Yb = X_tr[:32], Y_tr[:32]
REPEATS = 100

rows = []
for width in [16, 64]:
    net = MLP(width=width, seed=42)

    # warm-up
    P, cache = net.forward(Xb)
    backward(net, cache, Yb)

    fwd, bwd = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        P, cache = net.forward(Xb)
        fwd.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        backward(net, cache, Yb)
        bwd.append((time.perf_counter() - t0) * 1000)

    f, b = statistics.median(fwd), statistics.median(bwd)
    rows.append([width, round(f, 4), round(b, 4), round(b / f, 2)])
    print(f"W={width}: forward={f:.4f} ms, backward={b:.4f} ms, "
          f"backward/forward={b / f:.2f}")

with open("lab02/results/timing.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["width", "forward_median_ms", "backward_median_ms", "ratio"])
    w.writerows(rows)