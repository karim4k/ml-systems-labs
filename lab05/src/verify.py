import random
import numpy as np
import torch
from engine import Value, vsum

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

N = 5
a0, b0, c0 = (np.random.randn(N) for _ in range(3))


def f_values(a, b, c, relu_first=False):
    """((a*b + c)**2).relu().sum()  built from Value nodes."""
    terms = []
    for ai, bi, ci in zip(a, b, c):
        if relu_first:
            terms.append((ai * bi + ci).relu())
        else:
            terms.append(((ai * bi + ci) ** 2).relu())
    return vsum(terms)


def f_float(a, b, c, relu_first=False):
    z = a * b + c
    return float((np.maximum(z, 0) if relu_first else np.maximum(z ** 2, 0)).sum())


def engine_grads(relu_first):
    a = [Value(x) for x in a0]; b = [Value(x) for x in b0]; c = [Value(x) for x in c0]
    out = f_values(a, b, c, relu_first)
    out.backward()
    return out.data, [v.grad for v in a], [v.grad for v in b], [v.grad for v in c]


def torch_grads(relu_first):
    a, b, c = (torch.tensor(x, dtype=torch.float64, requires_grad=True)
               for x in (a0, b0, c0))
    z = a * b + c
    out = (z.relu() if relu_first else (z ** 2).relu()).sum()
    out.backward()
    return out.item(), a.grad.numpy(), b.grad.numpy(), c.grad.numpy()


def numeric_grads(relu_first, h=1e-6):
    grads = []
    for arr in (a0, b0, c0):
        g = np.zeros(N)
        for i in range(N):
            hi = arr.copy(); hi[i] += h
            lo = arr.copy(); lo[i] -= h
            args_hi = [hi if arr is a0 else a0, hi if arr is b0 else b0, hi if arr is c0 else c0]
            args_lo = [lo if arr is a0 else a0, lo if arr is b0 else b0, lo if arr is c0 else c0]
            g[i] = (f_float(*args_hi, relu_first) - f_float(*args_lo, relu_first)) / (2 * h)
        grads.append(g)
    return grads


lines = []
for name, rf in [("((a*b+c)**2).relu().sum()", False),
                 ("(a*b+c).relu().sum()   [extra: exercises the ReLU gate]", True)]:
    val_e, ga, gb, gc = engine_grads(rf)
    val_t, ta, tb, tc = torch_grads(rf)
    na, nb, nc = numeric_grads(rf)
    eng = np.concatenate([ga, gb, gc])
    d_torch = np.abs(eng - np.concatenate([ta, tb, tc])).max()
    d_num = np.abs(eng - np.concatenate([na, nb, nc])).max()
    lines += [
        f"Expression: {name}",
        f"  forward value: engine={val_e:.10f} torch={val_t:.10f}",
        f"  max |engine - torch autograd| = {d_torch:.3e}  "
        f"({'OK' if d_torch < 1e-6 else 'FAIL'}, need < 1e-6)",
        f"  max |engine - numerical|      = {d_num:.3e}  "
        f"({'OK' if d_num < 1e-6 else 'FAIL'}, need < 1e-6)",
        "",
    ]

# Accumulation test: x used twice, f = x*x + x  ->  df/dx = 2x + 1
x = Value(3.0)
y = x * x + x
y.backward()
lines.append(f"Accumulation test x*x + x at x=3: grad={x.grad} (expected 7.0) "
             f"-> {'OK' if abs(x.grad - 7.0) < 1e-12 else 'FAIL'}")

text = "\n".join(lines)
print(text)
with open("lab05/results/autograd_check.txt", "w", encoding="utf-8") as fh:
    fh.write(text)