import csv
import random
import statistics
import time

import numpy as np
import torch
from torch import nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

out_lines = []


def log(s=""):
    print(s)
    out_lines.append(s)


# ---------- Part A: inspect the autograd graph of a small model ----------
small = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
xs = torch.randn(3, 4)
loss = small(xs).pow(2).mean()

log("== Part A: autograd graph (grad_fn chain, from the loss backwards) ==")
seen = set()


def walk(fn, depth=0):
    if fn is None or id(fn) in seen:
        return
    seen.add(id(fn))
    log("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1)


walk(loss.grad_fn)
log()
log("Tensor inspection:")
log(f"  input   shape={tuple(xs.shape)} dtype={xs.dtype} contiguous={xs.is_contiguous()} stride={xs.stride()}")
w = small[0].weight
log(f"  weight  shape={tuple(w.shape)} dtype={w.dtype} contiguous={w.is_contiguous()} stride={w.stride()}")
wt = w.t()
log(f"  weight.t() shape={tuple(wt.shape)} contiguous={wt.is_contiguous()} stride={wt.stride()}")
log()
traced_small = torch.jit.trace(small, xs)
log("== torch.jit.trace graph of the small model ==")
log(str(traced_small.graph))

# ---------- Part B: eager vs traced vs torch.compile ----------
def make_model(d_in, d_hid, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                         nn.Linear(d_hid, d_hid), nn.ReLU(),
                         nn.Linear(d_hid, d_out)).eval()


def bench(fn, x, warmup=3, repeats=100):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


rows = []
log("== Part B: eager vs torch.jit.trace vs torch.compile (median of 100, ms) ==")
for label, batch in [("MLP 512-1024-1024-10, batch 256", 256),
                     ("MLP 512-1024-1024-10, batch 1", 1)]:
    model = make_model(512, 1024, 10)
    x = torch.randn(batch, 512)
    eager = bench(model, x)
    with torch.no_grad():
        traced = torch.jit.trace(model, x)
    tr = bench(traced, x)

    comp_note, comp = "", None
    try:
        compiled = torch.compile(model)
        comp = bench(compiled, x, warmup=3, repeats=100)
    except Exception as e:                        # unsupported on this machine
        comp_note = f"torch.compile failed: {type(e).__name__}: {str(e)[:150]}"
    rows.append({"case": label, "eager_ms": round(eager, 4),
                 "traced_ms": round(tr, 4),
                 "compiled_ms": round(comp, 4) if comp is not None else "n/a"})
    log(f"{label}: eager={eager:.4f}  traced={tr:.4f}  "
        f"compiled={'n/a' if comp is None else f'{comp:.4f}'}")
    if comp_note:
        log("  " + comp_note)

with open("lab05/results/eager_vs_compiled.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# ---------- Part C: contiguous vs non-contiguous tensors ----------
log()
log("== Part C: contiguous vs non-contiguous (median of 20, ms) ==")
n = 2048
x = torch.randn(n, n)
xt = x.t()                                    # a view, not a copy
log(f"x:  shape={tuple(x.shape)} dtype={x.dtype} contiguous={x.is_contiguous()} stride={x.stride()}")
log(f"xt: shape={tuple(xt.shape)} dtype={xt.dtype} contiguous={xt.is_contiguous()} stride={xt.stride()}")


def t_ms(fn, warmup=2, repeats=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


tests = [
    ("x + x            (both contiguous)", lambda: x + x),
    ("xt + x           (mixed layouts)", lambda: xt + x),
    ("x.exp()          (contiguous)", lambda: x.exp()),
    ("xt.exp()         (non-contiguous)", lambda: xt.exp()),
    ("xt.contiguous()  (copy to fix layout)", lambda: xt.contiguous()),
    ("xt.reshape(-1)   (forces a copy)", lambda: xt.reshape(-1)),
    ("x.reshape(-1)    (free view)", lambda: x.reshape(-1)),
    ("x @ x            (matmul)", lambda: x @ x),
    ("xt @ x           (matmul, transposed)", lambda: xt @ x),
]
crows = []
for name, fn in tests:
    ms = t_ms(fn)
    crows.append({"operation": name.strip(), "median_ms": round(ms, 4)})
    log(f"{name}: {ms:.4f}")

with open("lab05/results/contiguity.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=crows[0].keys())
    w.writeheader()
    w.writerows(crows)

with open("lab05/results/graph_inspection.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out_lines))