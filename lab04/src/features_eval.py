import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from preprocess import split, add_features, fit_and_transform


def run(df, cfg):
    X_tr, X_te, y_tr, y_te = split(df, cfg)

    # Duplicates: drop exact duplicates from TRAIN only; test untouched
    full = X_tr.copy()
    full["_y"] = y_tr
    before = len(full)
    full = full.drop_duplicates()
    y_tr = full.pop("_y")
    X_tr = full
    print(f"Train duplicates dropped: {before - len(X_tr)}; test rows kept: {len(X_te)}")

    rows = []
    for label, with_f in [("without_features", False), ("with_features", True)]:
        a, b = X_tr, X_te
        extra = []
        if with_f:
            a, b = add_features(a), add_features(b)
            extra = ["capital_net", "hours_per_age"]
        save = "lab04/results/preprocessor.joblib" if with_f else None
        A, B, _ = fit_and_transform(a, b, extra, cfg, save=save)
        clf = LogisticRegression(max_iter=1000, random_state=cfg["seed"])
        clf.fit(A, y_tr)
        pred = clf.predict(B)
        rows.append({"setting": label,
                     "accuracy": round(accuracy_score(y_te, pred), 4),
                     "f1_>50K": round(f1_score(y_te, pred), 4)})
        print(rows[-1])

    with open("lab04/results/feature_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return X_tr, X_te, y_tr, y_te