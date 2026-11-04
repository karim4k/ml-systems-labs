# Lab 5: Frameworks and automatic differentiation

## 1. Goal
Build a scalar autograd engine, validate it against PyTorch autograd and
numerical gradients, train a small model with it, and compare eager,
traced and compiled PyTorch execution and the cost of non-contiguous tensors.

## 2. Method
- Engine (src/engine.py): a Value node stores data, grad, its parent nodes
  and a local backward rule. Addition, multiplication, powers and ReLU build
  the graph during the forward pass. Backward sorts the graph
  topologically (iterative, no recursion), sets the output gradient to 1,
  visits nodes in reverse order and accumulates gradients with +=. sigmoid
  and log were added only for the training task.
- Verification (src/verify.py): ((a*b + c)**2).relu().sum() on random
  vectors of length 5 (seed 42), compared with PyTorch autograd (float64)
  and central-difference numerical gradients (h = 1e-6). Because a square
  is never negative, the ReLU in this expression never blocks anything, so I
  also tested (a*b + c).relu().sum() and an accumulation test (x*x + x).
- Training (src/moons.py): Two Moons, **n_samples=200** [CHECK], noise 0.1,
  random_state 42, architecture 2 -> 8 -> 1, ReLU hidden layer, sigmoid
  output, BCE loss, full-batch gradient descent, lr 0.5, 30 epochs, float64,
  same starting weights for both implementations. The manual's dataset uses
  1000 samples, but my first attempt with 1000 samples and 100 epochs ran
  for many minutes without finishing, because the engine is pure Python. I
  reduced the sample count and epochs. Time is the median of 5 runs after
  1 warm-up.
- PyTorch graph (src/torch_graph.py): grad_fn chain of a small MLP, tensor
  shape/dtype/contiguity inspection, and eager vs torch.jit.trace vs
  torch.compile on a 512-1024-1024-10 MLP (3 warm-up calls, median of 100,
  CPU, no_grad). Contiguity test on a 2048x2048 float32 tensor, median of 20
  after 2 warm-up calls.
- Timings change a little between runs, so the tables use the final run.

## 3. Results

### 3.1 Engine vs PyTorch and numerical gradients
| Expression | max abs diff vs PyTorch | max abs diff vs numerical |
|---|---|---|
| ((a*b+c)**2).relu().sum() | 0.000e+00 | 1.052e-09 |
| (a*b+c).relu().sum() (extra, tests the ReLU gate) | 0.000e+00 | 5.423e-11 |

Both are below the required 1e-6. The forward values also match PyTorch
(11.6879237903 and 0.7390209494). Accumulation test x*x + x at x = 3: the
engine gives gradient 7.0, as expected (2x + 1).

### 3.2 Training on Two Moons (30 epochs, n = 200) [CHECK]
| Implementation | Median time (s) | Final loss | Accuracy |
|---|---|---|---|
| My engine | 2.245 | 0.299566 | 0.8650 |
| PyTorch | 0.015 | 0.299566 | 0.8650 |

The engine is about 152 times slower. The largest difference between the
two loss curves over all epochs is 2.4e-12, so both implementations follow
the same optimisation path. This is a second check that the gradients are
correct.

### 3.3 Autograd graph (small model Linear-ReLU-Linear)
Read from the loss backwards, the graph is MeanBackward0, PowBackward0,
AddmmBackward0 (second Linear), ReluBackward0, AddmmBackward0 (first
Linear), with AccumulateGrad nodes at the parameters and TBackward0 nodes
for the transposed weights. Input: shape (3, 4), float32, contiguous,
stride (4, 1). Weight: shape (8, 4), float32, contiguous, stride (4, 1).
weight.t(): shape (4, 8), not contiguous, stride (1, 4), so it is a view of
the same memory. The jit.trace graph only shows three calls to the
submodules (Linear, ReLU, Linear), because the traced module keeps the
module structure.

### 3.4 Eager vs traced vs compiled (median of 100, ms)
| Case | Eager | jit.trace | torch.compile |
|---|---|---|---|
| MLP 512-1024-1024-10, batch 256 | 2.3911 | 2.3611 | 2.4871 |
| MLP 512-1024-1024-10, batch 1 | 0.2250 | 0.1942 | 0.2256 |

torch.compile ran without an error on this machine. I did not check which
backend it used. At batch 256 tracing was about 1% faster and compile about
4% slower than eager. At batch 1 tracing was about 14% faster, and compile
matched eager. In the earlier run the batch-1 gain from tracing was smaller
(0.2231 vs 0.2068 ms), so differences of a few percent are probably noise.
Only the batch-1 tracing gain is large enough to be worth a note.

### 3.5 Contiguous vs non-contiguous (2048x2048 float32, median of 20, ms)
| Operation | Time (ms) |
|---|---|
| x + x (both contiguous) | 0.3311 |
| xt + x (one operand transposed) | 3.4675 |
| x.exp() | 0.8108 |
| xt.exp() (non-contiguous) | 0.9529 |
| xt.contiguous() (copy) | 5.8072 |
| xt.reshape(-1) (forces a copy) | 4.6318 |
| x.reshape(-1) (view, no copy) | 0.0012 |
| x @ x | 20.5559 |
| xt @ x | 21.9996 |

## 4. When does compilation help?
In my measurements compilation did not help: eager, traced and compiled
runs were within a few percent of each other, except for a 14% gain from
tracing at batch 1. The model is three large matrix multiplications, and
these already run in optimised library kernels in eager mode, so there is
little left to remove. As far as I understand it, tracing and compilation
help most when a model has many small operations, especially elementwise
ones that can be fused to avoid repeated memory reads and writes, when the
same model is called many times (so the one-time compile cost is paid
back), and on accelerators. They help little for a few big matrix products
on CPU, and they add a warm-up cost. I did not test a model with many
elementwise operations, so this is background knowledge and not something
my numbers show.

## 5. Conclusions
1. My scalar engine reproduces PyTorch. Its gradients match PyTorch
   autograd exactly and the numerical estimates to about 1e-9 (section
   3.1), and training from the same starting weights gave the same final
   loss (0.299566) and accuracy (0.8650), with loss curves that differ by at
   most 2.4e-12. The key details were the topological order and accumulating
   with +=, so that a node used several times receives every contribution.

2. Building one Python object per scalar is very expensive. The engine took
   2.245 s against 0.015 s for PyTorch, about 152 times slower, for the same
   result, and that is with a reduced dataset. PyTorch groups the work into
   tensor operations that run in optimised compiled code, which is
   probably the main reason frameworks use tensors instead of scalar
   graphs. I did not profile the engine, so this is my interpretation.

3. Memory layout matters, but not for every operation. Adding a transposed
   view to a contiguous tensor was about 10 times slower than adding two
   contiguous tensors (3.47 ms vs 0.33 ms), exp on a transposed view was
   about 18% slower, and making a contiguous copy cost 5.8 ms. Matrix
   multiplication differed by only about 7% (20.6 vs 22.0 ms), probably
   because the matmul routine handles transposed inputs without a full copy.
   Reshape on a contiguous tensor is free (0.0012 ms), but on a transposed
   view it forces a copy (4.6 ms). Compilation did not speed up my MLP
   (section 4).