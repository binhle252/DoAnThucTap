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


def clean_flow(data: pd.DataFrame | dict) -> pd.DataFrame:

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame([data])

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
    graph,
    node_mapping: dict[str, int],
    flow_df: pd.DataFrame,
    edge_attr: np.ndarray,
):
    x = graph["x"].copy()
    edge_index = graph["edge_index"].copy()
    old_edge_attr = graph["edge_attr"].copy()

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

def load_model(model_path: Path, cpu: bool = False):

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    config = checkpoint["config"]

    threshold = float(
        checkpoint.get("threshold", 0.5)
    )

    model = GATEdgeClassifier(
        node_in_channels=config["node_in_channels"],
        edge_in_channels=config["edge_in_channels"],
        hidden_channels=config["hidden_channels"],
        heads=config["heads"],
        layers=config.get("layers", 2),
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return (
        model,
        device,
        threshold,
        config,
    )

def predict_flow(
        flow,
        model,
        device,
        threshold,
        preprocessor,
        node_mapping,
        graph,
    ) -> dict:    

    import time

    t0 = time.perf_counter()

    flow_df = clean_flow(flow)

    print("clean:", time.perf_counter() - t0)

    t1 = time.perf_counter()

    new_edge_attr = preprocessor.transform(flow_df)

    print("preprocess:", time.perf_counter() - t1)

    t2 = time.perf_counter()

    x, edge_index, edge_attr = append_new_flow_to_graph(
        graph=graph,
        node_mapping=node_mapping,
        flow_df=flow_df,
        edge_attr=new_edge_attr,
    )

    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    print("append:", time.perf_counter() - t2)

    t3 = time.perf_counter()

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        probs = torch.softmax(logits[-1], dim=0).detach().cpu().numpy()

    malicious_probability = float(probs[1])
    benign_probability = float(probs[0])
    predicted_class = int(malicious_probability >= threshold)

    print("model:", time.perf_counter() - t3)
    print(device)

    return {
        "predicted_class": predicted_class,
        "predicted_label": "Malicious" if predicted_class == 1 else "Benign",
        "threshold": threshold,
        "benign_probability": benign_probability,
        "malicious_probability": malicious_probability,
        "input_flow": flow,
    }


def predict_dataframe(
    df: pd.DataFrame,
    model_path: Path,
    arrays_path: Path,
    node_mapping_path: Path,
    preprocessor_path: Path,
    cpu: bool = False,
) -> pd.DataFrame:

    model, device, threshold, _ = load_model(
        model_path,
        cpu,
    )

    preprocessor = joblib.load(preprocessor_path)

    node_mapping = load_node_mapping(node_mapping_path)

    results = []

    for _, row in df.iterrows():

        result = predict_flow(
            flow=row.to_dict(),
            model=model,
            device=device,
            threshold=threshold,
            preprocessor=preprocessor,
            node_mapping=node_mapping.copy(),
            arrays_path=arrays_path,
        )

        results.append(result)

    output = df.copy()

    output["Prediction"] = [
        r["predicted_label"]
        for r in results
    ]

    output["Probability"] = [
        r["malicious_probability"]
        for r in results
    ]

    print("1. Load model")
    model, device, threshold, _ = load_model(
        model_path,
        cpu,
    )

    print("2. Load preprocessor")
    preprocessor = joblib.load(preprocessor_path)

    print("3. Load node mapping")
    node_mapping = load_node_mapping(node_mapping_path)

    print("4. Start prediction")

    results = []

    for i, (_, row) in enumerate(df.iterrows()):

        if i % 100 == 0:
            print(f"Processing {i}/{len(df)}")

        result = predict_flow(
            flow=row.to_dict(),
            model=model,
            device=device,
            threshold=threshold,
            preprocessor=preprocessor,
            node_mapping=node_mapping.copy(),
            arrays_path=arrays_path,
        )

        results.append(result)

    print("5. Finish")

    return output

def load_graph(arrays_path):

    arrays = np.load(arrays_path)

    return {
        "x": arrays["x"].astype(np.float32),
        "edge_index": arrays["edge_index"].astype(np.int64),
        "edge_attr": arrays["edge_attr"].astype(np.float32),
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
