import numpy as np
from nn import load_data, MLP, cross_entropy, backward

X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_data()
Xs, Ys = X_tr[:32], Y_tr[:32]          # small batch for the check


def loss_fn(net):
    P, _ = net.forward(Xs)
    return cross_entropy(P, Ys)


def check(net, label, n=10, h=1e-5):
    _, cache = net.forward(Xs)
    grads = backward(net, cache, Ys)
    rng = np.random.RandomState(42)
    out = [f"--- {label} ---"]
    passed = 0
    for _ in range(n):
        name = ["W1", "W2", "W3"][rng.randint(3)]
        arr = net.params[name]
        idx = tuple(rng.randint(s) for s in arr.shape)
        old = arr[idx]
        arr[idx] = old + h
        lp = loss_fn(net)
        arr[idx] = old - h
        lm = loss_fn(net)
        arr[idx] = old
        num = (lp - lm) / (2 * h)
        ana = grads[name][idx]
        rel = abs(ana - num) / (abs(ana) + abs(num) + 1e-12)
        ok = rel < 1e-6
        passed += ok
        out.append(f"{name}{idx}: analytic={ana:.4e} numeric={num:.4e} "
                   f"rel_err={rel:.2e} {'OK' if ok else 'FAIL'}")
    out.append(f"Passed {passed}/{n} (need at least 9)")
    return "\n".join(out)


net = MLP(width=64)
text = check(net, "At initialisation")

# After a short training run the gradients are larger, so the check is stronger
for _ in range(200):
    _, cache = net.forward(X_tr)
    g = backward(net, cache, Y_tr)
    for k in net.params:
        net.params[k] -= 0.1 * g[k]
text += "\n\n" + check(net, "After 200 full-batch steps (lr 0.1)")

print(text)
with open("lab02/results/gradient_check.txt", "w") as f:
    f.write(text)