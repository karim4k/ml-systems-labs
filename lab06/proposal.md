
Lead scoring for phone sales: predict which customers will say yes to a call
# Individual Project Proposal

**Student:** Karim Kallel  **ID:** 70592

## 1. Problem
Phone sales teams call many customers and only a small share say yes.
The project builds a lead-scoring model that predicts, before a call,
whether a customer will subscribe to the offer, so that agents call the
most promising leads first. It must run cheaply inside a CRM or a
call-center tool, so the model has to be small and fast. I have sold by
phone and face to face in my own business, so I know this process.

## 2. Dataset
UCI Bank Marketing (Moro, Cortez and Rita, 2014), phone campaigns of a
Portuguese bank.
- Source: UCI Machine Learning Repository. License: I believe CC BY 4.0,
  and I will confirm it on the dataset page.
- Size: about 45,000 rows, 16 input features plus the yes/no outcome.
- Known issues: strong class imbalance (about 11% yes); one portuguese
  bank and 2008-2010 only; the `duration` feature is known only after the
  call and will be removed to avoid leakage; age and marital status are
  sensitive attributes.
- Approval requested for this dataset (not in the default list).

## 3. Success metric
- Primary: F1 and PR-AUC on the positive class, on a held-out test set
  with a stratified split before any fitting.
- Business metric: share of all "yes" customers found in the top 20% of
  scored leads (lift), compared with calling at random.
- Target: the compressed model loses at most 2 points of PR-AUC against
  the teacher.

## 4. Deployment constraints
- Latency <= 50 ms per customer (batch 1, CPU, median of 100 runs after
  warm-up).
- Model size <= 5 MB (file on disk).

## 5. Baselines and planned optimization
- Baselines: logistic regression and gradient-boosted trees.
- Teacher: a large ensemble (several large MLPs or boosted models) whose
  total size is above 5 MB.
- Student: a small MLP under 5 MB.
- Optimization: (1) knowledge distillation from the teacher to the
  student, (2) magnitude pruning with fine-tuning, (3) post-training
  int8 quantization, compared as an ablation.
- Final table: PR-AUC, F1, lift, file size and latency for each model.

## 6. Plan and risks
- Reuse the Lab 4 leakage-free pipeline (split first, fit on train only,
  duplicates, validation report, SHA256 hash, data card) and the Lab 6
  YAML + MLflow training pipeline. Seed 42, pinned versions.
- Risk 1: tabular models are already small, so size and latency limits
  may be met without compression; I will report that honestly and use
  the teacher/student gap to show the benefit.
- Risk 2: class imbalance, so I will use class weights or resampling on
  train only, and judge by PR-AUC, not accuracy.
- Risk 3: int8 speed-up is not guaranteed on my CPU, so I will measure it.