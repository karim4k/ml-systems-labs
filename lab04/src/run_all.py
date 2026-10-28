import random

import numpy as np
import yaml

import ingest
import validate
import features_eval


def main():
    with open("lab04/config.yaml") as f:
        cfg = yaml.safe_load(f)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    raw = ingest.load_raw()
    df = ingest.to_schema(raw)
    X_tr, X_te, y_tr, y_te = features_eval.run(df, cfg)
    import audit
    audit.run(df, cfg, X_tr, X_te, y_tr, y_te)


if __name__ == "__main__":
    main()
    