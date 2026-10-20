import copy
import csv
import statistics
import time

import torch
from torchinfo import summary
from thop import profile

from common import MODELS, set_seeds

set_seeds(42)


def latency(model, bs, repeats):
    model.eval()
    x = torch.randn(bs, 1, 28, 28)
    ts = []
    with torch.no_grad():
        model(x)                                    # warm-up
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


rows = []
for name, make in MODELS.items():
    set_seeds(42)
    model = make()
    n_params = sum(p.numel() for p in model.parameters())

    info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    with open(f"lab03/results/summary_{name}.txt", "w", encoding="utf-8") as f:
        f.write(str(info))

    macs, _ = profile(copy.deepcopy(model), inputs=(torch.randn(1, 1, 28, 28),),
                      verbose=False)

    l1 = latency(model, 1, 100)
    l128 = latency(model, 128, 20)

    rows.append({
        "model": name,
        "params": n_params,
        "thop_macs": int(macs),
        "thop_flops_2xMACs": int(2 * macs),
        "latency_b1_ms": round(l1, 4),
        "latency_b128_ms": round(l128, 4),
    })
    print(rows[-1])

with open("lab03/results/cost.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("torch threads:", torch.get_num_threads())