import os
from sklearn.datasets import fetch_openml

RESULTS = os.path.join("lab04", "results")


def load_raw():
    """Read the raw Adult data. No cleaning, no dropping, no value changes."""
    os.makedirs(RESULTS, exist_ok=True)
    bunch = fetch_openml(name="adult", version=2, as_frame=True)
    df = bunch.frame.copy()
    n_rows, n_cols = df.shape
    lines = [
        f"Rows: {n_rows} (expected 48842) -> {'OK' if n_rows == 48842 else 'MISMATCH'}",
        f"Columns: {n_cols} (expected 15) -> {'OK' if n_cols == 15 else 'MISMATCH'}",
        f"Column names: {list(df.columns)}",
        "Dtypes:",
        df.dtypes.to_string(),
    ]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "ingestion_log.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return df


def to_schema(df):
    """Only renames the target column ('class' -> 'income') to match the manual."""
    return df.rename(columns={"class": "income"})