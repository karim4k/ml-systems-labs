import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cost = {r["model"]: r for r in csv.DictReader(open("lab03/results/cost.csv"))}
acc = {r["model"]: r for r in csv.DictReader(open("lab03/results/accuracy.csv"))}

panels = [
    ("params", "Parameters (log scale)", True),
    ("latency_b1_ms", "Latency, batch 1 (ms)", False),
    ("latency_b128_ms", "Latency, batch 128 (ms)", False),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, (key, label, logx) in zip(axes, panels):
    for name in acc:
        x = float(cost[name][key])
        y = float(acc[name]["test_acc"])
        ax.scatter(x, y, s=80)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Test accuracy")
plt.tight_layout()
plt.savefig("lab03/results/accuracy_vs_cost.png", dpi=150)