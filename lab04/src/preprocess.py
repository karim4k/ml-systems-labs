import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RESULTS = os.path.join("lab04", "results")
NUM = ["age", "fnlwgt", "education-num", "capital-gain",
       "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation",
       "relationship", "race", "sex", "native-country"]


class Clipper(BaseEstimator, TransformerMixin):
    """Clip each column to the lo/hi percentile learned on train only."""

    def __init__(self, lo=1, hi=99):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.percentile(X, self.lo, axis=0)
        self.high_ = np.percentile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def split(df, cfg):
    X = df.drop(columns="income")
    y = (df["income"].astype(str).str.strip() == ">50K").astype(int)
    return train_test_split(X, y, test_size=cfg["split"]["test_size"],
                            stratify=y, random_state=cfg["seed"])


def add_features(X):
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16).clip(lower=1)
    return X


def build(num, cat, cfg):
    lo, hi = cfg["clip_percentiles"]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clip", Clipper(lo, hi)),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def fit_and_transform(X_tr, X_te, extra, cfg, save=None):
    num = NUM + extra
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    for c in CAT:                       # categoricals -> object, NaN stays NaN
        X_tr[c] = X_tr[c].astype(object)
        X_te[c] = X_te[c].astype(object)
    pre = build(num, CAT, cfg)
    A = pre.fit(X_tr).transform(X_tr)   # fitted on TRAIN only
    B = pre.transform(X_te)
    if save:
        joblib.dump(pre, save)
    return A, B, pre