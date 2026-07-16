import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")

rows = []

for file in sorted(RESULTS_DIR.glob("gat_metrics_seed_*.json")):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test = data["final_metrics"]["test"]

    rows.append({
        "seed": data.get("seed", file.stem.split("_")[-1]),
        "best_epoch": data["best_epoch"],
        "threshold": test["threshold"],
        "accuracy": test["accuracy"],
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "roc_auc": test["roc_auc"],
        "pr_auc": test["pr_auc"],
    })

df = pd.DataFrame(rows)

print("\n================ PER SEED ================\n")
print(df.to_string(index=False))

print("\n================ MEAN ± STD ================\n")

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

summary = []

for m in metrics:
    summary.append({
        "Metric": m,
        "Mean": df[m].mean(),
        "Std": df[m].std(),
        "Min": df[m].min(),
        "Max": df[m].max(),
    })

summary_df = pd.DataFrame(summary)

pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print(summary_df.to_string(index=False))

print("\n================ BEST SEED ================\n")

best = df.loc[df["f1"].idxmax()]

print(best)