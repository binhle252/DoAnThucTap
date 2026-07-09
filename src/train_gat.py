from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GATEdgeClassifier(nn.Module):
    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        hidden_channels: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels=node_in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_in_channels,
        )
        self.gat2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_in_channels,
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_in_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        message_edge_index: torch.Tensor,
        message_edge_attr: torch.Tensor,
        classify_edge_index: torch.Tensor | None = None,
        classify_edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if classify_edge_index is None:
            classify_edge_index = message_edge_index
        if classify_edge_attr is None:
            classify_edge_attr = message_edge_attr

        h = self.gat1(x, message_edge_index, message_edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, message_edge_index, message_edge_attr)

        src_nodes, dst_nodes = classify_edge_index
        edge_repr = torch.cat([h[src_nodes], h[dst_nodes], classify_edge_attr], dim=1)
        return self.edge_classifier(edge_repr)


def load_graph_data(arrays_path: Path, device: torch.device) -> Data:
    arrays = np.load(arrays_path)
    data = Data(
        x=torch.tensor(arrays["x"], dtype=torch.float32),
        edge_index=torch.tensor(arrays["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(arrays["edge_attr"], dtype=torch.float32),
        y=torch.tensor(arrays["y"], dtype=torch.long),
    )
    data.train_mask = torch.tensor(arrays["train_mask"], dtype=torch.bool)
    data.val_mask = torch.tensor(arrays["val_mask"], dtype=torch.bool)
    data.test_mask = torch.tensor(arrays["test_mask"], dtype=torch.bool)
    return data.to(device)


def get_message_edges(data: Data, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "all":
        return data.edge_index, data.edge_attr
    if mode == "train":
        return data.edge_index[:, data.train_mask], data.edge_attr[data.train_mask]
    raise ValueError(f"Unknown message passing edge mode: {mode}")


def positive_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def compute_metrics(
    positive_probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> dict:
    probs = positive_probs[mask].detach().cpu().numpy()
    true = labels[mask].detach().cpu().numpy()
    preds = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        preds,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(true, preds, labels=[0, 1]).ravel()

    try:
        roc_auc = float(roc_auc_score(true, probs))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(true, probs))
    except ValueError:
        pr_auc = None

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(true, preds)),
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


def tune_threshold(
    positive_probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    probs = positive_probs[mask].detach().cpu().numpy()
    true = labels[mask].detach().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(true, probs)
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1_scores))

    threshold = 1.0 if best_idx >= len(thresholds) else float(thresholds[best_idx])
    return {
        "threshold": threshold,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1_scores[best_idx]),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    message_edge_index: torch.Tensor,
    message_edge_attr: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    logits = model(
        data.x,
        message_edge_index,
        message_edge_attr,
        data.edge_index,
        data.edge_attr,
    )
    positive_probs = positive_probs_from_logits(logits)
    return {
        "train": compute_metrics(positive_probs, data.y, data.train_mask, threshold),
        "val": compute_metrics(positive_probs, data.y, data.val_mask, threshold),
        "test": compute_metrics(positive_probs, data.y, data.test_mask, threshold),
    }


@torch.no_grad()
def evaluate_with_threshold_tuning(
    model: nn.Module,
    data: Data,
    message_edge_index: torch.Tensor,
    message_edge_attr: torch.Tensor,
) -> dict:
    model.eval()
    logits = model(
        data.x,
        message_edge_index,
        message_edge_attr,
        data.edge_index,
        data.edge_attr,
    )
    positive_probs = positive_probs_from_logits(logits)
    threshold_info = tune_threshold(positive_probs, data.y, data.val_mask)
    threshold = threshold_info["threshold"]

    return {
        "threshold_selection": {
            "selected_on": "validation",
            **threshold_info,
        },
        "metrics": {
            "train": compute_metrics(positive_probs, data.y, data.train_mask, threshold),
            "val": compute_metrics(positive_probs, data.y, data.val_mask, threshold),
            "test": compute_metrics(positive_probs, data.y, data.test_mask, threshold),
        },
    }


def flatten_metrics(metrics: dict) -> dict:
    flat = {}
    for split, values in metrics.items():
        for key, value in values.items():
            if key == "confusion_matrix":
                for cm_key, cm_value in value.items():
                    flat[f"{split}_{cm_key}"] = cm_value
            else:
                flat[f"{split}_{key}"] = value
    return flat


def generalization_gap(metrics: dict) -> dict:
    return {
        key: float(metrics["train"][key] - metrics["test"][key])
        for key in ["accuracy", "precision", "recall", "f1"]
    }


def save_confusion_matrices(results_dir: Path, metrics: dict) -> None:
    for split, values in metrics.items():
        cm = values["confusion_matrix"]
        rows = [
            "actual,predicted,count",
            f"Benign,Benign,{cm['tn']}",
            f"Benign,Malicious,{cm['fp']}",
            f"Malicious,Benign,{cm['fn']}",
            f"Malicious,Malicious,{cm['tp']}",
        ]
        (results_dir / f"confusion_matrix_{split}.csv").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )


def train(args: argparse.Namespace) -> dict:
    set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    selection_metric = getattr(args, "selection_metric", "val_f1")

    data = load_graph_data(args.arrays_path, device)
    message_passing_edges = getattr(args, "message_passing_edges", "all")
    message_edge_index, message_edge_attr = get_message_edges(data, message_passing_edges)
    model = GATEdgeClassifier(
        node_in_channels=data.x.shape[1],
        edge_in_channels=data.edge_attr.shape[1],
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best_selection_score = -1.0
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(
            data.x,
            message_edge_index,
            message_edge_attr,
            data.edge_index,
            data.edge_attr,
        )
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        metrics = evaluate(
            model,
            data,
            message_edge_index,
            message_edge_attr,
            threshold=0.5,
        )
        with torch.no_grad():
            model.eval()
            selection_logits = model(
                data.x,
                message_edge_index,
                message_edge_attr,
                data.edge_index,
                data.edge_attr,
            )
            selection_probs = positive_probs_from_logits(selection_logits)
            val_threshold_info = tune_threshold(selection_probs, data.y, data.val_mask)

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "val_tuned_threshold": float(val_threshold_info["threshold"]),
            "val_tuned_precision": float(val_threshold_info["precision"]),
            "val_tuned_recall": float(val_threshold_info["recall"]),
            "val_tuned_f1": float(val_threshold_info["f1"]),
            **flatten_metrics(metrics),
        }
        history.append(row)

        if selection_metric not in row:
            raise ValueError(f"Unknown selection metric: {selection_metric}")
        selection_score = row[selection_metric]

        if selection_score > best_selection_score + args.min_delta:
            best_selection_score = selection_score
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_f1={row['val_f1']:.4f} | "
            f"val_tuned_f1={row['val_tuned_f1']:.4f} | "
            f"best_epoch={best_epoch:03d}"
        )

        if args.early_stopping and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state_dict)
    default_threshold_metrics = evaluate(
        model,
        data,
        message_edge_index,
        message_edge_attr,
        threshold=0.5,
    )
    tuned_evaluation = evaluate_with_threshold_tuning(
        model,
        data,
        message_edge_index,
        message_edge_attr,
    )
    final_metrics = tuned_evaluation["metrics"]
    threshold_selection = tuned_evaluation["threshold_selection"]
    gap = generalization_gap(final_metrics)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrices(args.results_dir, final_metrics)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "node_in_channels": int(data.x.shape[1]),
            "edge_in_channels": int(data.edge_attr.shape[1]),
            "hidden_channels": int(args.hidden_channels),
            "heads": int(args.heads),
            "dropout": float(args.dropout),
            "message_passing_edges": message_passing_edges,
            "selection_metric": selection_metric,
        },
        "threshold": float(threshold_selection["threshold"]),
        "best_epoch": int(best_epoch),
        "final_metrics": final_metrics,
    }
    model_path = args.model_dir / "gat_edge_classifier.pt"
    torch.save(checkpoint, model_path)

    results = {
        "device": str(device),
        "requested_epochs": int(args.epochs),
        "completed_epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "selection_policy": {
            "best_model_selected_by": selection_metric,
            "classification_threshold_selected_by": "validation_f1_precision_recall_curve",
            "test_set_usage": "reported_once_after_model_and_threshold_selection",
            "message_passing_edges": message_passing_edges,
            "selection_score": float(best_selection_score),
        },
        "threshold_selection": threshold_selection,
        "history": history,
        "default_threshold_metrics": default_threshold_metrics,
        "final_metrics": final_metrics,
        "generalization_gap_train_minus_test": gap,
        "model_path": str(model_path),
    }
    metrics_path = args.results_dir / "gat_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train a GAT edge classifier.")
    parser.add_argument(
        "--arrays-path",
        type=Path,
        default=root / "data" / "processed" / "graph_arrays.npz",
    )
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument(
        "--selection-metric",
        choices=["val_f1", "val_tuned_f1", "val_pr_auc", "val_roc_auc"],
        default="val_f1",
        help="Validation metric used for best-epoch selection.",
    )
    parser.add_argument(
        "--message-passing-edges",
        choices=["all", "train"],
        default="all",
        help="Use all edges for transductive GAT message passing, or only train edges for stricter evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train(args)
    print(json.dumps(results["final_metrics"], indent=2))


if __name__ == "__main__":
    main()
