from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.preprocess_binary import (
        CATEGORICAL_COLUMNS,
        IP_COLUMNS,
        NUMERIC_COLUMNS,
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
    )
except ModuleNotFoundError:
    from preprocess_binary import (
        CATEGORICAL_COLUMNS,
        IP_COLUMNS,
        NUMERIC_COLUMNS,
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
    )


SPLITS = ["train", "val", "test"]
IP_NODE_FEATURE_NAMES = [
    "ipv4_octet_1",
    "ipv4_octet_2",
    "ipv4_octet_3",
    "ipv4_octet_4",
    "is_private",
    "is_global",
    "is_multicast",
    "is_loopback",
]
TRAIN_NODE_STAT_FEATURE_NAMES = [
    "train_scaled_log1p_out_degree",
    "train_scaled_log1p_in_degree",
    "train_scaled_log1p_total_degree",
    "train_scaled_log1p_unique_out_peers",
    "train_scaled_log1p_unique_in_peers",
    "train_scaled_log1p_sent_bytes",
    "train_scaled_log1p_received_bytes",
    "train_scaled_log1p_sent_packets",
    "train_scaled_log1p_received_packets",
    "train_scaled_log1p_total_duration",
    "train_scaled_log1p_mean_duration",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_splits(processed_dir: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = processed_dir / f"{split}.csv"
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fit_edge_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    preprocessor.fit(train_df)
    return preprocessor


def get_edge_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def build_node_mapping(df: pd.DataFrame) -> dict[str, int]:
    ips = pd.concat([df[IP_COLUMNS[0]], df[IP_COLUMNS[1]]], ignore_index=True)
    unique_ips = sorted(ips.astype(str).unique())
    return {ip: idx for idx, ip in enumerate(unique_ips)}


def ip_to_features(ip_value: str) -> list[float]:
    try:
        ip_obj = ipaddress.ip_address(ip_value)
    except ValueError:
        return [0.0] * 8

    if ip_obj.version == 4:
        octets = [part / 255.0 for part in ip_obj.packed]
    else:
        octets = [0.0, 0.0, 0.0, 0.0]

    return [
        *octets,
        float(ip_obj.is_private),
        float(ip_obj.is_global),
        float(ip_obj.is_multicast),
        float(ip_obj.is_loopback),
    ]


def build_ip_node_features(node_mapping: dict[str, int]) -> np.ndarray:
    x = np.zeros((len(node_mapping), 8), dtype=np.float32)
    for ip_value, node_id in node_mapping.items():
        x[node_id] = np.asarray(ip_to_features(ip_value), dtype=np.float32)
    return x


def map_ips_to_node_ids(values: pd.Series, node_mapping: dict[str, int]) -> np.ndarray:
    return values.astype(str).map(node_mapping).to_numpy(dtype=np.int64)


def build_train_node_stat_features(
    train_df: pd.DataFrame,
    node_mapping: dict[str, int],
) -> np.ndarray:
    n_nodes = len(node_mapping)
    out_degree = np.zeros(n_nodes, dtype=np.float64)
    in_degree = np.zeros(n_nodes, dtype=np.float64)
    unique_out_peers = np.zeros(n_nodes, dtype=np.float64)
    unique_in_peers = np.zeros(n_nodes, dtype=np.float64)
    sent_bytes = np.zeros(n_nodes, dtype=np.float64)
    received_bytes = np.zeros(n_nodes, dtype=np.float64)
    sent_packets = np.zeros(n_nodes, dtype=np.float64)
    received_packets = np.zeros(n_nodes, dtype=np.float64)
    duration_sum = np.zeros(n_nodes, dtype=np.float64)

    src = map_ips_to_node_ids(train_df[IP_COLUMNS[0]], node_mapping)
    dst = map_ips_to_node_ids(train_df[IP_COLUMNS[1]], node_mapping)

    orig_bytes = train_df["orig_bytes"].to_numpy(dtype=np.float64)
    resp_bytes = train_df["resp_bytes"].to_numpy(dtype=np.float64)
    orig_pkts = train_df["orig_pkts"].to_numpy(dtype=np.float64)
    resp_pkts = train_df["resp_pkts"].to_numpy(dtype=np.float64)
    duration = train_df["duration"].to_numpy(dtype=np.float64)

    np.add.at(out_degree, src, 1.0)
    np.add.at(in_degree, dst, 1.0)

    np.add.at(sent_bytes, src, orig_bytes)
    np.add.at(sent_bytes, dst, resp_bytes)
    np.add.at(received_bytes, src, resp_bytes)
    np.add.at(received_bytes, dst, orig_bytes)

    np.add.at(sent_packets, src, orig_pkts)
    np.add.at(sent_packets, dst, resp_pkts)
    np.add.at(received_packets, src, resp_pkts)
    np.add.at(received_packets, dst, orig_pkts)

    np.add.at(duration_sum, src, duration)
    np.add.at(duration_sum, dst, duration)

    out_peer_counts = train_df.groupby(IP_COLUMNS[0])[IP_COLUMNS[1]].nunique()
    in_peer_counts = train_df.groupby(IP_COLUMNS[1])[IP_COLUMNS[0]].nunique()
    for ip_value, count in out_peer_counts.items():
        unique_out_peers[node_mapping[str(ip_value)]] = float(count)
    for ip_value, count in in_peer_counts.items():
        unique_in_peers[node_mapping[str(ip_value)]] = float(count)

    total_degree = out_degree + in_degree
    mean_duration = np.divide(
        duration_sum,
        total_degree,
        out=np.zeros_like(duration_sum),
        where=total_degree > 0,
    )

    stats = np.column_stack(
        [
            out_degree,
            in_degree,
            total_degree,
            unique_out_peers,
            unique_in_peers,
            sent_bytes,
            received_bytes,
            sent_packets,
            received_packets,
            duration_sum,
            mean_duration,
        ]
    )
    log_stats = np.log1p(np.maximum(stats, 0.0))
    mean = log_stats.mean(axis=0, keepdims=True)
    std = log_stats.std(axis=0, keepdims=True)
    scaled_stats = (log_stats - mean) / np.where(std < 1e-12, 1.0, std)
    return scaled_stats.astype(np.float32)


def build_node_features(
    node_mapping: dict[str, int],
    train_df: pd.DataFrame,
    node_feature_mode: str,
) -> np.ndarray:

    ip_features = build_ip_node_features(node_mapping)

    train_stats = build_train_node_stat_features(
        train_df,
        node_mapping,
    )

    if node_feature_mode == "ip":
        return ip_features

    elif node_feature_mode == "stats":
        return train_stats

    elif node_feature_mode == "ip_stats":
        return np.hstack(
            [
                ip_features,
                train_stats,
            ]
        ).astype(np.float32)

    else:
        raise ValueError(
            f"Unknown node feature mode: {node_feature_mode}"
        )


def build_edge_index(df: pd.DataFrame, node_mapping: dict[str, int]) -> np.ndarray:
    src = df[IP_COLUMNS[0]].astype(str).map(node_mapping).to_numpy(dtype=np.int64)
    dst = df[IP_COLUMNS[1]].astype(str).map(node_mapping).to_numpy(dtype=np.int64)
    return np.vstack([src, dst])


def build_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        f"{split}_mask": (df["split"].to_numpy() == split)
        for split in SPLITS
    }


def build_graph_arrays(
    processed_dir: Path,
    output_dir: Path,
    node_feature_mode: str = "ip",
) -> dict:
    df = load_splits(processed_dir)
    train_df = df[df["split"] == "train"]

    preprocessor = fit_edge_preprocessor(train_df)
    edge_attr = preprocessor.transform(df).astype(np.float32)
    edge_feature_names = get_edge_feature_names(preprocessor)

    node_mapping = build_node_mapping(df)
    x = build_node_features(node_mapping, train_df, node_feature_mode)
    node_feature_names = IP_NODE_FEATURE_NAMES
    if node_feature_mode == "ip_stats":
        node_feature_names = IP_NODE_FEATURE_NAMES + TRAIN_NODE_STAT_FEATURE_NAMES
    edge_index = build_edge_index(df, node_mapping)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    masks = build_masks(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "graph_arrays.npz"
    np.savez_compressed(
        arrays_path,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        **masks,
    )

    mapping_df = (
        pd.DataFrame(
            {
                "ip": list(node_mapping.keys()),
                "node_id": list(node_mapping.values()),
            }
        )
        .sort_values("node_id")
    )

    mapping_df.to_csv(
        output_dir / "node_mapping.csv",
        index=False,
    )

    node_mapping_path = output_dir / "node_mapping.csv"
    pd.DataFrame(
        {
            "ip": list(node_mapping.keys()),
            "node_id": list(node_mapping.values()),
        }
    ).to_csv(node_mapping_path, index=False)

    preprocessor_path = output_dir / "edge_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "task": "binary_edge_classification",
        "graph_design": {
            "node": "IP address",
            "edge": "traffic flow from id.orig_h to id.resp_h",
            "label": TARGET_COLUMN,
        },
        "model_feature_policy": {
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "excluded_columns": [TIMESTAMP_COLUMN],
            "note": "Timestamp is kept for splitting/metadata but is not used as a model feature.",
        },
        "node_feature_policy": {
            "mode": node_feature_mode,
            "ip_features": IP_NODE_FEATURE_NAMES,
            "train_only_stat_features": (
                TRAIN_NODE_STAT_FEATURE_NAMES if node_feature_mode == "ip_stats" else []
            ),
            "note": "Train-only node statistics are computed only from train edges, use no labels, then log1p-scaled and standardized across nodes.",
        },
        "num_nodes": int(x.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_dim": int(edge_attr.shape[1]),
        "node_feature_names": node_feature_names,
        "edge_feature_names": edge_feature_names,
        "class_counts": {
            str(key): int(value)
            for key, value in pd.Series(y).value_counts().sort_index().items()
        },
        "split_counts": {
            split: int((df["split"] == split).sum())
            for split in SPLITS
        },
        "paths": {
            "arrays": str(arrays_path),
            "node_mapping": str(node_mapping_path),
            "edge_preprocessor": str(preprocessor_path),
        },
    }

    metadata_path = output_dir / "graph_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["paths"]["metadata"] = str(metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build graph-ready arrays for PyG.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "processed",
    )
    parser.add_argument(
        "--node-feature-mode",
        choices=[
            "ip",
            "stats",
            "ip_stats",
        ],
        default="ip",
        help="Use only IP-derived node features, or add train-only node statistics for ablation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_graph_arrays(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        node_feature_mode=args.node_feature_mode,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
