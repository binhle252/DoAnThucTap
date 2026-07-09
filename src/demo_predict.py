from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

try:
    from src.build_graph_arrays import ip_to_features
    from src.preprocess_binary import CATEGORICAL_COLUMNS, IP_COLUMNS, NUMERIC_COLUMNS
    from src.train_gat import GATEdgeClassifier
except ModuleNotFoundError:
    from build_graph_arrays import ip_to_features
    from preprocess_binary import CATEGORICAL_COLUMNS, IP_COLUMNS, NUMERIC_COLUMNS
    from train_gat import GATEdgeClassifier


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_flow(flow: dict) -> pd.DataFrame:
    df = pd.DataFrame([flow])
    df = df.replace("-", np.nan)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("missing").astype(str)

    for col in IP_COLUMNS:
        df[col] = df[col].fillna("unknown").astype(str)

    return df


def load_node_mapping(path: Path) -> dict[str, int]:
    mapping_df = pd.read_csv(path)
    return dict(zip(mapping_df["ip"].astype(str), mapping_df["node_id"].astype(int)))


def append_new_flow_to_graph(
    arrays_path: Path,
    node_mapping: dict[str, int],
    flow_df: pd.DataFrame,
    edge_attr: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = np.load(arrays_path)
    x = arrays["x"].astype(np.float32)
    edge_index = arrays["edge_index"].astype(np.int64)
    old_edge_attr = arrays["edge_attr"].astype(np.float32)

    src_ip = str(flow_df.iloc[0][IP_COLUMNS[0]])
    dst_ip = str(flow_df.iloc[0][IP_COLUMNS[1]])

    for ip_value in [src_ip, dst_ip]:
        if ip_value not in node_mapping:
            node_mapping[ip_value] = len(node_mapping)
            ip_features = np.asarray(ip_to_features(ip_value), dtype=np.float32)
            if x.shape[1] > len(ip_features):
                padding = np.zeros(x.shape[1] - len(ip_features), dtype=np.float32)
                node_features = np.concatenate([ip_features, padding])
            else:
                node_features = ip_features[: x.shape[1]]
            x = np.vstack([x, node_features])

    new_edge_index = np.asarray(
        [[node_mapping[src_ip]], [node_mapping[dst_ip]]],
        dtype=np.int64,
    )

    edge_index = np.hstack([edge_index, new_edge_index])
    edge_attr = np.vstack([old_edge_attr, edge_attr.astype(np.float32)])

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(edge_attr, dtype=torch.float32),
    )


def predict_flow(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint["config"]
    threshold = float(checkpoint.get("threshold", 0.5))

    model = GATEdgeClassifier(
        node_in_channels=config["node_in_channels"],
        edge_in_channels=config["edge_in_channels"],
        hidden_channels=config["hidden_channels"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    flow = {
        "ts": args.ts,
        "id.orig_h": args.src_ip,
        "id.orig_p": args.src_port,
        "id.resp_h": args.dst_ip,
        "id.resp_p": args.dst_port,
        "proto": args.proto,
        "service": args.service,
        "duration": args.duration,
        "orig_bytes": args.orig_bytes,
        "resp_bytes": args.resp_bytes,
        "conn_state": args.conn_state,
        "missed_bytes": args.missed_bytes,
        "history": args.history,
        "orig_pkts": args.orig_pkts,
        "orig_ip_bytes": args.orig_ip_bytes,
        "resp_pkts": args.resp_pkts,
        "resp_ip_bytes": args.resp_ip_bytes,
    }
    flow_df = clean_flow(flow)

    preprocessor = joblib.load(args.preprocessor_path)
    new_edge_attr = preprocessor.transform(flow_df)
    node_mapping = load_node_mapping(args.node_mapping_path)

    x, edge_index, edge_attr = append_new_flow_to_graph(
        arrays_path=args.arrays_path,
        node_mapping=node_mapping,
        flow_df=flow_df,
        edge_attr=new_edge_attr,
    )

    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        probs = torch.softmax(logits[-1], dim=0).detach().cpu().numpy()

    malicious_probability = float(probs[1])
    benign_probability = float(probs[0])
    predicted_class = int(malicious_probability >= threshold)

    return {
        "predicted_class": predicted_class,
        "predicted_label": "Malicious" if predicted_class == 1 else "Benign",
        "threshold": threshold,
        "benign_probability": benign_probability,
        "malicious_probability": malicious_probability,
        "input_flow": flow,
    }


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Predict one new IoT-23 traffic flow.")
    parser.add_argument("--arrays-path", type=Path, default=root / "data" / "processed" / "graph_arrays.npz")
    parser.add_argument("--node-mapping-path", type=Path, default=root / "data" / "processed" / "node_mapping.csv")
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=root / "data" / "processed" / "edge_preprocessor.joblib",
    )
    parser.add_argument("--model-path", type=Path, default=root / "models" / "gat_edge_classifier.pt")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--ts", type=float, default=1536227023.384673)
    parser.add_argument("--src-ip", default="192.168.100.111")
    parser.add_argument("--src-port", type=float, default=17576.0)
    parser.add_argument("--dst-ip", default="78.1.220.212")
    parser.add_argument("--dst-port", type=float, default=8081.0)
    parser.add_argument("--proto", default="tcp")
    parser.add_argument("--service", default="missing")
    parser.add_argument("--duration", type=float, default=0.000003)
    parser.add_argument("--orig-bytes", type=float, default=0.0)
    parser.add_argument("--resp-bytes", type=float, default=0.0)
    parser.add_argument("--conn-state", default="S0")
    parser.add_argument("--missed-bytes", type=float, default=0.0)
    parser.add_argument("--history", default="S")
    parser.add_argument("--orig-pkts", type=float, default=2.0)
    parser.add_argument("--orig-ip-bytes", type=float, default=80.0)
    parser.add_argument("--resp-pkts", type=float, default=0.0)
    parser.add_argument("--resp-ip-bytes", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_flow(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
