import random
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)


def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0                      # scale inputs to [0, 1]
    Y = np.eye(10)[y]                 # one-hot labels
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
        X, Y, y, test_size=0.3, stratify=y, random_state=42
    )
    return X_tr, X_te, Y_tr, Y_te, y_tr, y_te


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)   # numerical stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class MLP:
    """64 -> W (ReLU) -> W (ReLU) -> 10 (softmax)"""

    def __init__(self, width=64, seed=42):
        np.random.seed(seed)
        self.params = {
            "W1": np.random.normal(0, 0.01, (64, width)),
            "b1": np.zeros(width),
            "W2": np.random.normal(0, 0.01, (width, width)),
            "b2": np.zeros(width),
            "W3": np.random.normal(0, 0.01, (width, 10)),
            "b3": np.zeros(10),
        }

    def forward(self, X):
        p = self.params
        Z1 = X @ p["W1"] + p["b1"]
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ p["W2"] + p["b2"]
        A2 = np.maximum(0, Z2)
        Z3 = A2 @ p["W3"] + p["b3"]
        P = softmax(Z3)
        cache = (X, Z1, A1, Z2, A2, P)
        return P, cache


def cross_entropy(P, Y):
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))

def backward(net, cache, Y):
    """Gradients of mean cross-entropy w.r.t. all parameters."""
    p = net.params
    X, Z1, A1, Z2, A2, P = cache
    N = X.shape[0]

    dZ3 = (P - Y) / N                  # softmax + cross-entropy
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0)

    dA2 = dZ3 @ p["W3"].T
    dZ2 = dA2 * (Z2 > 0)               # ReLU derivative
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ p["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}