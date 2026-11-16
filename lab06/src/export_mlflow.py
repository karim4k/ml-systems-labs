from pathlib import Path
import mlflow

mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
df = mlflow.search_runs(search_all_experiments=True)
df.to_csv("lab06/results/mlflow_runs.csv", index=False)
print(df[["tags.mlflow.runName", "metrics.test_acc",
          "metrics.peak_rss_mb"]].to_string())