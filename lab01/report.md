# Lab 1: Environment and first system measurements

## 1. Goal
Set up a reproducible Python 3.11.8 environment with pinned dependencies,
train two baseline models on Breast Cancer Wisconsin, measure their system
cost, and decide which deployment targets (Cloud, Edge, Mobile, TinyML) each
model could run on.

## 2. Method
- Environment: venv with requirements.txt. pyarrow is pinned to 15.0.2
  instead of 16.1.0, because mlflow 2.14.1 requires pyarrow<16.
- Data: load_breast_cancer, 80/20 stratified split, random_state=42.
- Models: LogisticRegression(max_iter=1000) and
  RandomForestClassifier(n_estimators=100), seed 42.
- Training time: 1 warm-up, then median of 5 runs.
- Inference latency: 1 warm-up, then median of 100 single-sample predictions.
- Model size: saved with joblib.dump. One file per model
  (model_logreg.joblib, model_random_forest.joblib) instead of a single
  model.joblib, because there are two models.
- Memory: psutil RSS of the whole Python process during inference.
- Timings vary between runs on a laptop, so the table uses the final run.

## 3. Results

| Model | Test accuracy | Train time, median (s) | Latency, median (ms) | Size (bytes / KB) | Process RSS (MB) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9561 | 0.2821 | 0.0318 | 1055 / 1.03 | 104.9 |
| Random Forest | 0.9561 | 0.1094 | 1.4484 | 324329 / 316.73 | 106.0 |

Deployment fit:

| Target | Budget | Logistic Regression | Random Forest |
|---|---|---|---|
| Cloud | mem >= 1 GB, <= 100 ms, <= 500 MB | Fits | Fits |
| Edge | mem 256-1024 MB, <= 50 ms, <= 50 MB | Fits | Fits |
| Mobile | mem 64-256 MB, <= 20 ms, <= 10 MB | Fits | Fits |
| TinyML | mem <= 256 KB, <= 10 ms, <= 100 KB | Model fits (1 KB, 0.03 ms), but the 105 MB Python process does not | Does not fit (317 KB > 100 KB) |

The RSS is the whole Python process, not just the model.

## 4. Conclusions
1. Both models reached the same test accuracy of 0.9561, so accuracy
could not decide between them. The choice came down to system cost.
The Random Forest was about 307 times larger (317 KB vs 1 KB) and
about 45 times slower at single-sample inference (1.45 ms vs 0.03 ms).

2. Both models fit the Cloud, Edge and Mobile budgets with a lot of
   room to spare. TinyML is the only target that separates them.
   Random Forest is ruled out because 317 KB exceeds the 100 KB limit.
   Logistic Regression is the only candidate, since the model itself is
   1 KB and 0.03 ms. However, my measured 105 MB is the whole Python
   process, so it would only work on TinyML after being reimplemented
   outside Python, for example in C.

3. Logistic Regression trained more slowly than Random Forest
   (0.28 s vs 0.11 s), which I did not expect. It most likely happened
   because the lbfgs solver reached the 1000-iteration limit on unscaled
   features and raised a ConvergenceWarning. Timings also changed a lot
   between runs, so a single run would be misleading, which is why the
   lab requires warm-up and medians.