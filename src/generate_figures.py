from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def save_model_comparison(results_dir: Path, figures_dir: Path) -> None:
    comparison = pd.read_csv(results_dir / "model_comparison.csv")
    comparison = comparison.sort_values("f1", ascending=True)

    plt.figure(figsize=(9, 5))
    bars = plt.barh(comparison["model"], comparison["f1"], color="#2f6f73")
    plt.xlim(0, 1.03)
    plt.xlabel("F1-score")
    plt.title("Model Comparison on Test Set")
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_f1.png", dpi=180)
    plt.close()


def save_training_curve(results_dir: Path, figures_dir: Path) -> None:
    metrics = json.loads((results_dir / "gat_metrics.json").read_text(encoding="utf-8"))
    history = pd.DataFrame(metrics["history"])

    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_f1"], label="Train F1", linewidth=2)
    plt.plot(history["epoch"], history["val_f1"], label="Validation F1", linewidth=2)
    if "test_f1" in history:
        plt.plot(history["epoch"], history["test_f1"], label="Test F1", linewidth=1.5, linestyle="--")
    plt.axvline(metrics["best_epoch"], color="#9b2226", linestyle=":", label="Best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.ylim(0.55, 1.02)
    plt.title("GAT Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "gat_training_curve.png", dpi=180)
    plt.close()


def save_gat_confusion_matrix(results_dir: Path, figures_dir: Path) -> None:
    metrics = json.loads((results_dir / "gat_metrics.json").read_text(encoding="utf-8"))
    cm = metrics["final_metrics"]["test"]["confusion_matrix"]
    matrix = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual Benign", "Actual Malicious"],
        columns=["Predicted Benign", "Predicted Malicious"],
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("GAT Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(figures_dir / "gat_confusion_matrix_test.png", dpi=180)
    plt.close()


def save_metric_table(results_dir: Path, figures_dir: Path) -> None:
    comparison = pd.read_csv(results_dir / "model_comparison.csv")
    columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    table = comparison[columns].sort_values("f1", ascending=False)
    table.to_csv(figures_dir / "model_comparison_table.csv", index=False)


def generate_figures(args: argparse.Namespace) -> None:
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    save_model_comparison(args.results_dir, args.figures_dir)
    save_training_curve(args.results_dir, args.figures_dir)
    save_gat_confusion_matrix(args.results_dir, args.figures_dir)
    save_metric_table(args.results_dir, args.figures_dir)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Generate figures for report and slides.")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--figures-dir", type=Path, default=root / "results" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figures(args)
    print(f"Saved figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
