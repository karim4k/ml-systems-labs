import os
import hashlib
import numpy as np
import pandas as pd
import yaml

RESULTS = os.path.join("lab04", "results")


def conflict_audit(df):
    feats = [c for c in df.columns if c != "income"]
    g = df.groupby(feats, observed=True, dropna=False)["income"].nunique()
    conflicting_groups = int((g > 1).sum())
    sizes = df.groupby(feats, observed=True, dropna=False).size()
    dup_groups = int((sizes > 1).sum())
    rows_in_conflict = int(sizes[g[g > 1].index].sum()) if conflicting_groups else 0
    return dup_groups, conflicting_groups, rows_in_conflict


def sample_check(df, cfg):
    s = df.sample(50, random_state=cfg["seed"]).copy()
    s["bad_age"] = ~s["age"].between(17, 90)
    s["bad_hours"] = ~s["hours-per-week"].between(1, 99)
    s["bad_gain"] = ~s["capital-gain"].between(0, 99999)
    s["bad_loss"] = ~s["capital-loss"].between(0, 4356)
    s["edu_mismatch"] = False
    pairs = df.groupby("education", observed=True)["education-num"].nunique()
    s["edu_mismatch"] = s["education"].map(pairs).fillna(1) > 1
    s.to_csv(os.path.join(RESULTS, "sample_50_rows.csv"), index=False)
    flags = ["bad_age", "bad_hours", "bad_gain", "bad_loss", "edu_mismatch"]
    return s[flags].sum().to_dict()


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def run(df, cfg, X_tr, X_te, y_tr, y_te):
    dup_groups, conf_groups, conf_rows = conflict_audit(df)
    flags = sample_check(df, cfg)
    lines = [
        f"Duplicate feature-row groups: {dup_groups}",
        f"Groups with conflicting income labels: {conf_groups}",
        f"Rows involved in conflicts: {conf_rows}",
        f"50-row sample impossible-value flags: {flags}",
    ]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "label_audit.txt"), "w") as f:
        f.write(text)

    # Processed dataset = train + test (cleaned features), saved deterministically
    from preprocess import add_features
    proc_tr = add_features(X_tr).assign(income=y_tr.values, split="train")
    proc_te = add_features(X_te).assign(income=y_te.values, split="test")
    proc = pd.concat([proc_tr, proc_te]).reset_index(drop=True)
    path = os.path.join(RESULTS, "processed.csv")
    proc.to_csv(path, index=False)
    digest = hash_file(path)
    with open(os.path.join(RESULTS, "processed_sha256.txt"), "w") as f:
        f.write(digest)
    cfg_out = dict(cfg)
    cfg_out["processed_sha256"] = digest
    with open(os.path.join(RESULTS, "config_with_hash.yaml"), "w") as f:
        yaml.safe_dump(cfg_out, f)
    print("SHA256:", digest)