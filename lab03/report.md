# Lab 4: Data pipeline and quality

## 1. Problem definition
Task: predict whether a person's annual income exceeds 50K USD from census
records (UCI Adult, 1994 US census, OpenML version 2).

- Inputs: 14 features (6 numeric: age, fnlwgt, education-num, capital-gain,
  capital-loss, hours-per-week; 8 categorical: workclass, education,
  marital-status, occupation, relationship, race, sex, native-country).
- Output: income, with two classes (<=50K and >50K). The classes are
  imbalanced (76% vs 24%), so accuracy alone is misleading.
- Success metrics: F1 on the >50K class and accuracy, both on a held-out
  test set.
- Constraints: no leakage (the split happens before any fitting, and all
  statistics come from train only); the pipeline is reproducible (fixed seed
  42, pinned versions, one command, config file, SHA256 hash of the output).
- Risks: race and sex are sensitive attributes, and the data is from 1994,
  so it may not reflect today's population.

## 2. Ingestion
The raw data is read with fetch_openml(name='adult', version=2, as_frame=True)
without any changes. Rows: 48,842 and columns: 15, both matching the
expected values (logged in results/ingestion_log.txt). The target column is
named `class` in the raw data and is renamed to `income` only for validation.

## 3. Validation (report only, no rows dropped)
| Check | Result |
|---|---|
| Columns present | all 15, none missing or extra |
| Types | all match the schema |
| Ranges | 0 violations (age 17-90, hours 1-99, gain 0-99999, loss 0-4356) |
| Allowed categories | 0 violations |
| Missing values | workclass 2,799; occupation 2,809; native-country 857 |
| Exact duplicate rows | 52 |

OpenML stores '?' as NaN, so the '?' count is 0 and the missing values
appear as NaN. Class balance: 76.07% <=50K, 23.93% >50K. Full output is in
results/validation_report.txt and results/validation_report.csv.

## 4. Cleaning and leakage control
- The data is split 70/30, stratified on income, with seed 42, before any
  fitting: 34,189 train rows and 14,653 test rows.
- Imputation (median for numeric, mode for categorical), clipping to the
  1st/99th percentile, scaling and one-hot encoding are all fitted on train
  only. The test set only goes through transform.
- Exact duplicates were dropped from train only (24 rows). The test set was
  left untouched (14,653 rows).
- The fitted pipeline is saved to results/preprocessor.joblib.

## 5. Derived features
Two features were added: capital_net = capital-gain - capital-loss, and
hours_per_age = hours-per-week / (age - 16). Both were evaluated with a
LogisticRegression baseline on the same split.

| Setting | Accuracy | F1 (>50K) |
|---|---|---|
| Without derived features | 0.8540 | 0.6601 |
| With capital_net and hours_per_age | 0.8578 | 0.6696 |

The gain is +0.0038 accuracy and +0.0095 F1. This comes from one split and
one seed, so it may be noise.

## 6. Label-quality check
- Duplicate-conflict audit: 54 groups of rows with identical features, of
  which 5 groups have conflicting income labels (10 rows, about 0.02% of
  the data).
- A random sample of 50 rows (results/sample_50_rows.csv) was checked
  against the data dictionary. The automatic checks (ranges, and consistency
  of education with education-num) found no impossible values. I also read
 

## 7. Reproducibility
- One command: python lab04/src/run_all.py
- Settings are in lab04/config.yaml (seed 42, split 70/30, clip 1/99).
- SHA256 of the processed dataset (results/processed.csv):
  1de6b204b6524f63be6ada1a1faf9788580ba623821cf022a5d75860e951e7a9
  (also stored in results/processed_sha256.txt and
  results/config_with_hash.yaml).
- The data card is in lab04/data_card.md.

## 8. Conclusions
1. The raw Adult data was already clean by the schema checks: 48,842 rows,
   15 columns, correct types, and zero range or category violations. The
   real quality problem was missing values, which OpenML stores as NaN
   (workclass 2,799, occupation 2,809, native-country 857). Because the
   validation step only reports problems and never drops rows, these issues
   stayed visible instead of being silently removed.

2. Doing the stratified 70/30 split first, and fitting the imputer, clipper,
   scaler and encoder on train only, keeps the test set honest. Duplicates
   were dropped from train only (24 rows), so the 14,653 test rows were left
   untouched. The two derived features gave a small gain, from 0.8540 to
   0.8578 accuracy and from 0.6601 to 0.6696 F1. This comes from one split
   and one seed, so I treat it as a modest improvement, not a proven one.

3. The label audit found only 5 groups of identical feature rows with
   conflicting income labels (10 rows, about 0.02% of the data), so label
   noise from this source is very small. The 50-row sample had no impossible
   values. The pipeline is reproducible with one command and a SHA256 hash of
   the processed data. However, the data comes from 1994 and includes
   sensitive attributes (sex and race), so it should not be used to make
   real decisions about people.