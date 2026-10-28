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
    validate.run(df)
    features_eval.run(df, cfg)


if __name__ == "__main__":
    main()