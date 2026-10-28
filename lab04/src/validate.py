import os
import pandas as pd

RESULTS = os.path.join("lab04", "results")

INT_COLS = ["age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country", "income"]
RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}

ALLOWED = {
    "workclass": {"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                  "Local-gov", "State-gov", "Without-pay", "Never-worked"},
    "education": {"Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                  "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                  "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"},
    "marital-status": {"Married-civ-spouse", "Divorced", "Never-married",
                       "Separated", "Widowed", "Married-spouse-absent",
                       "Married-AF-spouse"},
    "occupation": {"Tech-support", "Craft-repair", "Other-service", "Sales",
                   "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                   "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                   "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"},
    "relationship": {"Wife", "Own-child", "Husband", "Not-in-family",
                     "Other-relative", "Unmarried"},
    "race": {"White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"},
    "sex": {"Female", "Male"},
    "native-country": {
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
        "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
        "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
        "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
        "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
        "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"},
    "income": {"<=50K", ">50K"},
}


def run(df):
    """Report-only checks. Never modifies or drops rows."""
    os.makedirs(RESULTS, exist_ok=True)
    rows = []

    def add(check, column, count, note=""):
        rows.append({"check": check, "column": column,
                     "count": int(count), "note": note})

    # Columns present
    expected = INT_COLS + CAT_COLS
    miss = sorted(set(expected) - set(df.columns))
    extra = sorted(set(df.columns) - set(expected))
    add("columns_present", "all", len(miss) + len(extra),
        f"missing={miss} extra={extra}")

    # Types
    for c in INT_COLS:
        if c not in df.columns:
            continue
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s)
        whole = bool(numeric and (s.dropna() % 1 == 0).all())
        add("type_int", c, 0 if whole else 1,
            f"dtype={s.dtype}, whole_numbers={whole}")
    for c in CAT_COLS:
        if c not in df.columns:
            continue
        ok = isinstance(df[c].dtype, pd.CategoricalDtype)
        add("type_category", c, 0 if ok else 1, f"dtype={df[c].dtype}")

    # Ranges
    for c, (lo, hi) in RANGES.items():
        bad = ((df[c] < lo) | (df[c] > hi)).sum()
        add("range", c, bad,
            f"allowed {lo}-{hi}, observed min={df[c].min()} max={df[c].max()}")

    # Missing values: NaN or '?' (reported, not dropped)
    for c in df.columns:
        n_nan = int(df[c].isna().sum())
        n_q = int((df[c].astype(str).str.strip() == "?").sum())
        if n_nan or n_q:
            add("missing", c, n_nan + n_q, f"NaN={n_nan}, '?'={n_q}")

    # Allowed categories ('?' and NaN count as missing, not as violations)
    for c, allowed in ALLOWED.items():
        if c not in df.columns:
            continue
        vals = df[c].dropna().astype(str).str.strip()
        bad = vals[~vals.isin(allowed | {"?"})]
        note = ""
        if len(bad):
            note = "unexpected: " + str(bad.value_counts().to_dict())
        add("allowed_categories", c, len(bad), note)

    # Exact duplicates (reported only)
    add("exact_duplicate_rows", "all", df.duplicated().sum(),
        "all 15 columns identical; rows are NOT dropped here")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "validation_report.csv"), index=False)

    lines = ["VALIDATION REPORT (report only, no rows dropped)",
             f"Rows checked: {len(df)}", ""]
    for r in rows:
        lines.append(f"[{r['check']}] {r['column']}: count={r['count']}  {r['note']}")
    lines += ["", "Income class balance:",
              df["income"].value_counts(normalize=True).round(4).to_string()]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "validation_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)