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
    }