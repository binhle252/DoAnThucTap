from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_graph_features(arrays_path: Path) -> dict:
    arrays = np.load(arrays_path)
    return {
        "x": arrays["edge_attr"],
        "y": arrays["y"],
        "train_mask": arrays["train_mask"],
        "val_mask": arrays["val_mask"],
        "test_mask": arrays["test_mask"],
    }


def tune_threshold(y_true: np.ndarray, positive_probs: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, positive_probs)
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1_scores))
    threshold = 1.0 if best_idx >= len(thresholds) else float(thresholds[best_idx])

    return {
        "threshold": threshold,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1_scores[best_idx]),
    }


def compute_metrics(
    y_true: np.ndarray,
    positive_probs: np.ndarray,
    threshold: float,
) -> dict:
    preds = (positive_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, positive_probs))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_true, positive_probs))
    except ValueError:
        pr_auc = None

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def predict_positive_probs(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]

    scores = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def build_models(random_state: int) -> dict:
    return {
        "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=30,
            max_depth=4,
            min_samples_leaf=20,
            min_samples_split=40,
            max_features=0.3,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.0005,
            batch_size=1024,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=80,
            random_state=random_state,
        ),
    }


def save_confusion_matrix(results_dir: Path, model_name: str, split: str, metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    rows = [
        "actual,predicted,count",
        f"Benign,Benign,{cm['tn']}",
        f"Benign,Malicious,{cm['fp']}",
        f"Malicious,Benign,{cm['fn']}",
        f"Malicious,Malicious,{cm['tp']}",
    ]
    path = results_dir / f"confusion_matrix_{model_name}_{split}.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_gat_row(results_dir: Path) -> dict | None:
    metrics_path = results_dir / "gat_metrics_seed_45.json"
    if not metrics_path.exists():
        return None

    gat = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = gat["final_metrics"]["test"]
    selected_by = gat.get("selection_policy", {}).get(
        "best_model_selected_by",
        "validation_f1",
    )
    return {
        "model": "GAT",
        "threshold": test["threshold"],
        "accuracy": test["accuracy"],
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "roc_auc": test["roc_auc"],
        "pr_auc": test["pr_auc"],
        "selected_by": selected_by,
    }


def train_baselines(args: argparse.Namespace) -> dict:
    data = load_graph_features(args.arrays_path)
    x = data["x"]
    y = data["y"]

    x_train, y_train = x[data["train_mask"]], y[data["train_mask"]]
    x_val, y_val = x[data["val_mask"]], y[data["val_mask"]]
    x_test, y_test = x[data["test_mask"]], y[data["test_mask"]]

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": {
            "source": str(args.arrays_path),
            "feature_source": "edge_attr_from_graph_arrays",
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "feature_dim": int(x.shape[1]),
        },
        "selection_policy": {
            "classification_threshold_selected_by": "validation_f1_precision_recall_curve",
            "test_set_usage": "reported_once_after_threshold_selection",
        },
        "models": {},
    }

    rows = []
    for model_name, model in build_models(args.random_state).items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)

        val_probs = predict_positive_probs(model, x_val)
        threshold_info = tune_threshold(y_val, val_probs)
        threshold = threshold_info["threshold"]

        split_metrics = {}
        for split, split_x, split_y in [
            ("train", x_train, y_train),
            ("val", x_val, y_val),
            ("test", x_test, y_test),
        ]:
            probs = predict_positive_probs(model, split_x)
            metrics = compute_metrics(split_y, probs, threshold)
            split_metrics[split] = metrics
            save_confusion_matrix(args.results_dir, model_name, split, metrics)

        model_path = args.model_dir / f"{model_name.lower()}_baseline.joblib"
        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "threshold_selection": threshold_info,
            },
            model_path,
        )

        test = split_metrics["test"]
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "accuracy": test["accuracy"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "roc_auc": test["roc_auc"],
                "pr_auc": test["pr_auc"],
                "selected_by": "validation_f1",
            }
        )

        results["models"][model_name] = {
            "model_path": str(model_path),
            "threshold_selection": threshold_info,
            "metrics": split_metrics,
        }
        print(
            f"{model_name}: "
            f"test_acc={test['accuracy']:.4f}, "
            f"test_f1={test['f1']:.4f}, "
            f"threshold={threshold:.4f}"
        )

    gat_row = load_gat_row(args.results_dir)
    if gat_row is not None:
        rows.append(gat_row)

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
    comparison_path = args.results_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    results["comparison_path"] = str(comparison_path)
    metrics_path = args.results_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train graphless baseline models.")
    parser.add_argument(
        "--arrays-path",
        type=Path,
        default=root / "data" / "processed" / "graph_arrays.npz",
    )
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_baselines(args)
    comparison = pd.read_csv(results["comparison_path"])
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
