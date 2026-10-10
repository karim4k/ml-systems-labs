import warnings
warnings.filterwarnings("ignore")
import random
import time
import statistics
import os
import csv

import numpy as np
import joblib
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)

RESULTS = os.path.join("lab01", "results")
os.makedirs(RESULTS, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "logreg": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}

proc = psutil.Process()
rows = []

for name, make in models.items():
    # Training time: 1 warm-up, then median of 5 runs
    make().fit(X_train, y_train)
    train_times = []
    for _ in range(5):
        m = make()
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)
    train_median = statistics.median(train_times)

    model = make().fit(X_train, y_train)
    acc = round(accuracy_score(y_test, model.predict(X_test)), 4)

    # Single-sample inference: warm-up, then 100 repeats, median in ms
    sample = X_test[:1]
    model.predict(sample)
    lat = []
    rss_before = proc.memory_info().rss
    rss_peak = rss_before
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(sample)
        lat.append((time.perf_counter() - t0) * 1000)
        rss_peak = max(rss_peak, proc.memory_info().rss)
    lat_median = statistics.median(lat)

    # Model size
    path = os.path.join(RESULTS, f"model_{name}.joblib")
    joblib.dump(model, path)
    size_bytes = os.path.getsize(path)

    rows.append({
        "model": name,
        "test_accuracy": acc,
        "train_time_median_s": round(train_median, 5),
        "infer_latency_median_ms": round(lat_median, 4),
        "model_size_bytes": size_bytes,
        "model_size_KB": round(size_bytes / 1024, 2),
        "rss_during_inference_MB": round(rss_peak / 1024**2, 1),
    })

with open(os.path.join(RESULTS, "baseline_accuracy.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "test_accuracy"])
    for r in rows:
        w.writerow([r["model"], f"{r['test_accuracy']:.4f}"])

with open(os.path.join(RESULTS, "system_costs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(r)