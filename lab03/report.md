# Lab 3: Architectures and their costs

## 1. Goal
Implement comparable MLP, CNN and patch-attention models in PyTorch on
Fashion-MNIST, compare parameters, FLOPs, latency and accuracy, and decide
which is most cost-effective.

## 2. Method
- Data: Fashion-MNIST, Normalize((0.2860,), (0.3530,)), 55k train / 5k
  validation / 10k test, batch size 128, seed 42.
- Models exactly as specified in the manual. The attention model has no
  positional embedding, because the manual does not mention one.
- Parameters: torchinfo and sum of p.numel().
- FLOPs: by hand (multiply-add = 2 FLOPs) and with thop (reports MACs).
- Latency: 1 warm-up, median of 100 runs (batch 1) and 20 runs (batch 128),
  CPU, 14 torch threads.
- Training: Adam, lr 1e-3, batch 128, 5 epochs, CPU.
- Note: the manual targets 100k-150k parameters, but the specified
  architectures give 235,146 (MLP), 20,490 (CNN) and 70,794 (attention).
  I built them as specified.

## 3. Results

| Model | Params | Test acc | Train time (s) | Latency b=1 (ms) | Latency b=128 (ms) | thop FLOPs |
|---|---|---|---|---|---|---|
| MLP | 235,146 | 0.8805 | 57.8 | 0.218 | 0.552 | 469,504 |
| CNN | 20,490 | 0.8912 | 66.2 | 0.273 | 2.776 | 2,063,488 |
| Attention | 70,794 | 0.8390 | 105.3 | 1.071 | 3.084 | 101,632 |

Hand-calculated FLOPs (one layer of each type):

| Layer | Formula | FLOPs |
|---|---|---|
| MLP Linear 784->256 | 2 x 784 x 256 | 401,408 |
| CNN Conv2d(1,16,3), 28x28 output | 2 x 3 x 3 x 1 x 16 x 28 x 28 | 225,792 |
| Attention QKV projections | 2 x 16 x 64 x 3 x 64 | 393,216 |
| Attention scores | 2 x 16^2 x 64 | 32,768 |

thop gap: thop counts MACs, so its Linear and Conv numbers are half the
hand FLOPs (they agree after multiplying by 2). For attention, thop total
(50,816 MACs) equals only the input projection plus the head, so it counted
nothing inside the encoder layers. My hand estimate is about 2.33 million
FLOPs for the whole model, roughly 23 times thop's figure.

Plots: results/accuracy_vs_cost.png (accuracy vs parameters and latency).

## 4. Conclusions
## 4. Conclusions
1. The CNN gave the best test accuracy (0.8912) with the fewest parameters
   (20,490), about 11 times fewer than the MLP (235,146), which reached
   0.8805. The attention model was lowest at 0.8390 after 5 epochs, but its
   validation accuracy was still rising, so more training might narrow the
   gap. Convolution reuses the same small kernel at every image position,
   so it gets good accuracy from few weights.

2. Parameter count does not predict latency. The CNN has 11 times fewer
   parameters than the MLP but is about 5 times slower at batch 128
   (2.776 ms vs 0.552 ms), because it performs about 4 times more FLOPs
   (thop: 2.06 M vs 0.47 M) on large activation maps. The MLP is the
   fastest at both batch sizes. Attention is the slowest at batch 1
   (1.071 ms), which fits many small operations (unfold, softmax,
   LayerNorm) where overhead dominates. Overall, the CNN is the most
   cost-effective on accuracy per parameter, and the MLP is best if
   latency matters most.

3. thop gave a misleading cost for attention. It reported 101,632 FLOPs,
   which covers only the input projection and the final layer, so it
   counted nothing inside the encoder layers. My hand estimate is about
   2.33 million FLOPs, roughly 23 times higher, which puts attention's
   real compute near the CNN's. Profiling tools therefore need to be
   checked against hand calculations, especially for attention layers.
   Also, the manual's 100k-150k parameter target did not match the
   specified architectures, so I reported the actual counts.
