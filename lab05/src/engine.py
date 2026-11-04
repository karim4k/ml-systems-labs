import math


class Value:
    """Scalar node: data, grad, parents and a local derivative rule."""

    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(parents)
        self._op = op
        self._backward = lambda: None     # local derivative rule

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # --- Task 2: operations that build the graph during the forward pass ---
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad          # += : accumulate, never overwrite
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "exponent must be a constant"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # --- extras needed only for Task 5 (sigmoid + BCE) ---
    def sigmoid(self):
        x = max(-60.0, min(60.0, self.data))
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # --- conveniences built from the operations above ---
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    # --- Task 3: topological backward pass ---
    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]            # iterative: no recursion limit
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
        self.grad = 1.0                    # output gradient = 1
        for node in reversed(topo):        # reverse topological order
            node._backward()


def vsum(values):
    """Sum of many Values as one node (the .sum() of the lab)."""
    values = list(values)
    out = Value(sum(v.data for v in values), values, "sum")

    def _backward():
        for v in values:
            v.grad += out.grad
    out._backward = _backward
    return out