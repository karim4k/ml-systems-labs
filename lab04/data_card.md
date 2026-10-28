# Data card: UCI Adult (OpenML v2)

- Source: UCI Machine Learning Repository, Adult / Census Income, via
  OpenML (data id for name=adult, version=2).
- Origin: 1994 US Census database (extracted by Barry Becker).
- License: check the OpenML/UCI page before submitting. I believe UCI
  lists it as CC BY 4.0, but I have not verified it.
- Size: 48,842 rows, 15 columns (14 features + income). Split 70/30:
  34,189 train, 14,653 test (before dropping 24 train duplicates).
- Missing values: workclass 2,799, occupation 2,809, native-country 857.
- Class balance: 76.07% <=50K, 23.93% >50K.
- Known biases: 1994 US data only; sex and race are included and income
  differs strongly between groups; fnlwgt are census sampling weights;
  the >50K label is a threshold on a skewed variable.
- Privacy: census-derived and de-identified, but combinations of age,
  country and occupation could still be re-identifying; use for coursework only.
- Reproducibility: python lab04/src/run_all.py; seed 42; processed
  dataset SHA256 in results/processed_sha256.txt.