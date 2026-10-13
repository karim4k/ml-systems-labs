## Network design (Task 2)

Architecture: input 64 -> hidden 1 (W, ReLU) -> hidden 2 (W, ReLU) -> output 10 (softmax)

- 64 inputs: each digit image is 8x8 pixels.
- 10 outputs: one per digit class (0 to 9).
- W = 64 for the main model (16 and 64 are compared later).

| Layer | Weight | Shape | Bias | Shape |
|---|---|---|---|---|
| Input -> hidden 1 | W1 | (64, W) | b1 | (W,) |
| Hidden 1 -> hidden 2 | W2 | (W, W) | b2 | (W,) |
| Hidden 2 -> output | W3 | (W, 10) | b3 | (10,) |

With W = 64: W1 (64, 64), b1 (64,), W2 (64, 64), b2 (64,), W3 (64, 10), b3 (10,).

Each weight matrix has shape (inputs, outputs) so the matrix products line up:
(batch, 64) @ (64, W) -> (batch, W) -> (batch, W) -> (batch, 10).
Each bias has one value per output neuron. Weights are initialised from
N(0, 0.01) and biases from zero.

## Backpropagation derivation (Task 5)

Notation: N = batch size, Z = pre-activation, A = ReLU(Z), P = softmax(Z3),
Y = one-hot labels, loss L = -(1/N) * sum_n sum_k Y_nk * log P_nk.

1. Softmax combined with cross-entropy simplifies to dZ3 = (P - Y) / N.
2. Output layer: dW3 = A2^T dZ3, db3 = sum over the batch of dZ3.
3. Pass back through W3: dA2 = dZ3 W3^T.
4. ReLU only passes gradient where the input was positive:
   dZ2 = dA2 * 1[Z2 > 0].
5. Hidden layer 2: dW2 = A1^T dZ2, db2 = sum over the batch of dZ2.
6. Pass back again: dA1 = dZ2 W2^T, dZ1 = dA1 * 1[Z1 > 0].
7. Hidden layer 1: dW1 = X^T dZ1, db1 = sum over the batch of dZ1.

## Numerical gradient check (Task 5)

Method: batch of 32 training images, 10 randomly chosen weights from W1, W2
and W3 (fixed seed). Each weight is nudged by h = 1e-5 in both directions.
The numerical slope is (L(w+h) - L(w-h)) / (2h). Relative error is
|a - n| / (|a| + |n|).

Result after 200 full-batch training steps (lr 0.1):

| Weight | Analytic | Numeric | Relative error | Pass (< 1e-6) |
|---|---|---|---|---|
| W3(51, 7) | 3.0047e-04 | 3.0047e-04 | 8.18e-09 | yes |
| W1(20, 38) | 1.4054e-04 | 1.4054e-04 | 4.01e-08 | yes |
| W2(18, 22) | 0 | 0 | 0 | yes |
| W3(10, 7) | -8.4970e-05 | -8.4970e-05 | 2.47e-08 | yes |
| W1(35, 39) | 6.6882e-05 | 6.6882e-05 | 9.22e-08 | yes |
| W3(21, 4) | 1.8627e-04 | 1.8627e-04 | 1.22e-08 | yes |
| W2(23, 43) | 0 | 0 | 0 | yes |
| W2(37, 1) | 0 | 0 | 0 | yes |
| W1(32, 11) | 0 | 0 | 0 | yes |
| W2(21, 60) | -1.9952e-06 | -1.9952e-06 | 1.09e-06 | no |

9 of 10 weights pass, which meets the requirement. The same check at
initialisation also gave 9 of 10 (full output in results/gradient_check.txt).

The one failure has a tiny gradient (about 2e-6). Its absolute difference is
about 1e-11, which matches the floating-point noise of a central difference
with h = 1e-5 (about 1e-16 * L / h). So it is rounding error, not a wrong
derivative. Weights with an exact zero gradient belong to ReLU units that
were inactive for all 32 images, so analytic and numeric both give zero.
