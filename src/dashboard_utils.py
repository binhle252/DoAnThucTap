import json
from pathlib import Path


RESULT_PATH = Path("results/gat_metrics.json")


def load_metrics():

    if not RESULT_PATH.exists():
        return None

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    test = data["final_metrics"]["test"]

    return {
        "accuracy": test["accuracy"],
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "roc_auc": test["roc_auc"],
        "pr_auc": test["pr_auc"],
        "threshold": data["threshold_selection"]["threshold"],
        "best_epoch": data["best_epoch"],
        "history": data["history"],
        "selection_policy": data["selection_policy"],
        "threshold_selection": data["threshold_selection"],
        "final_metrics": data["final_metrics"],
    }