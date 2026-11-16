# Individual Project Proposal

**Student:** Karim Kallel  **ID:** 70592

## 1. Problem
Classify small grayscale clothing images (10 classes) on a resource-limited
device, such as an embedded camera or a phone. A large model is accurate but
too big and slow for the device, so the goal is to compress it while keeping
accuracy close to the original.

## 2. Dataset
Fashion-MNIST (torchvision), as in Labs 3 and 6.
- Source: Zalando Research. License: MIT (I will confirm on the dataset page
  before submitting the data card).
- Size: 70,000 images of 28x28 pixels; 55k train / 5k validation / 10k test.
- Known biases: only 10 product categories, clean studio-style images on a
  plain background, so it does not represent real camera conditions.

## 3. Success metric
- Test accuracy (10k official test set), compared with the teacher and with
  my Lab 3 CNN (0.8912). Target: student accuracy within 1.5 points of the
  teacher and at least 0.90.
- Secondary: macro F1 per class.

## 4. Deployment constraints
- Latency <= 50 ms per single image (batch 1, CPU, median of 100 runs after
  warm-up).
- Model size <= 5 MB (saved file on disk).

## 5. Baselines and planned optimization
- Baselines: (a) my Lab 3 CNN (20,490 parameters, about 80 KB); (b) a larger
  teacher (a ResNet-style network, about 11 M parameters, which is too big for
  the 5 MB limit in float32).
- Optimization: knowledge distillation from the teacher into a small student
  network, followed by post-training int8 quantization. If time allows, I
  will add magnitude pruning and compare it with distillation alone.
- Evaluation: a table of accuracy, F1, file size and latency for the
  teacher, student, quantized student and (optionally) pruned student.

## 6. Plan and risks
- Reproducible YAML-driven pipeline with MLflow tracking (as built in Lab 6),
  fixed seed 42, pinned library versions.
- Risks: quantization may reduce accuracy; CPU-only training of the teacher
  may be slow, so I will limit its epochs; a pure int8 speed-up is not
  guaranteed on my CPU, so I will measure it and not assume it.