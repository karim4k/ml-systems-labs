import numpy as np
from nn import load_data, MLP, cross_entropy

X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_data()
net = MLP(width=64)

P, _ = net.forward(X_tr)

lines = []
lines.append(f"Train shape: {X_tr.shape}, test shape: {X_te.shape}")
for k, v in net.params.items():
    lines.append(f"{k}: {v.shape}")
lines.append(f"Output shape: {P.shape}  (expected ({X_tr.shape[0]}, 10))")
row_err = np.abs(P.sum(axis=1) - 1).max()
lines.append(f"Max |row sum - 1|: {row_err:.2e}  (must be < 1e-6)")
lines.append(f"Untrained loss: {cross_entropy(P, Y_tr):.4f}  (ln(10) = {np.log(10):.4f})")

text = "\n".join(lines)
print(text)
with open("lab02/results/forward_check.txt", "w") as f:
    f.write(text)