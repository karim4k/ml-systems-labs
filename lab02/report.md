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