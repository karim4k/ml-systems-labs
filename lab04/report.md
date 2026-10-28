# Lab 4: Data pipeline and quality

## 1. Problem definition
Task: predict whether a person's annual income exceeds 50K USD from census
records (UCI Adult, 1994 US census, OpenML version 2).

- Inputs: 14 features (6 numeric: age, fnlwgt, education-num, capital-gain,
  capital-loss, hours-per-week; 8 categorical: workclass, education,
  marital-status, occupation, relationship, race, sex, native-country).
- Output: income, with two classes (<=50K and >50K). The classes are
  imbalanced, so accuracy alone is misleading.
- Success metrics: F1 on the >50K class and accuracy, both on a held-out
  test set.
- Constraints: no leakage (the split happens before any fitting, and all
  statistics come from train only); the pipeline is reproducible (fixed seed
  42, pinned versions, one command, config file, SHA256 hash of the output).
- Risks: race and sex are sensitive attributes, and the data is from 1994,
  so it may not reflect today's population.