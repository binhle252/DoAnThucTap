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
    )
except ModuleNotFoundError:
    from preprocess_binary import (
        CATEGORICAL_COLUMNS,
        IP_COLUMNS,
        NUMERIC_COLUMNS,
        TARGET_COLUMN,
    )


SPLITS = ["train", "val", "test"]


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


def build_node_features(node_mapping: dict[str, int]) -> np.ndarray:
    x = np.zeros((len(node_mapping), 8), dtype=np.float32)
    for ip_value, node_id in node_mapping.items():
        x[node_id] = np.asarray(ip_to_features(ip_value), dtype=np.float32)
    return x


def build_edge_index(df: pd.DataFrame, node_mapping: dict[str, int]) -> np.ndarray:
    src = df[IP_COLUMNS[0]].astype(str).map(node_mapping).to_numpy(dtype=np.int64)
    dst = df[IP_COLUMNS[1]].astype(str).map(node_mapping).to_numpy(dtype=np.int64)
    return np.vstack([src, dst])


def build_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        f"{split}_mask": (df["split"].to_numpy() == split)
        for split in SPLITS
    }


def build_graph_arrays(processed_dir: Path, output_dir: Path) -> dict:
    df = load_splits(processed_dir)
    train_df = df[df["split"] == "train"]

    preprocessor = fit_edge_preprocessor(train_df)
    edge_attr = preprocessor.transform(df).astype(np.float32)
    edge_feature_names = get_edge_feature_names(preprocessor)

    node_mapping = build_node_mapping(df)
    x = build_node_features(node_mapping)
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
        "num_nodes": int(x.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_dim": int(edge_attr.shape[1]),
        "node_feature_names": [
            "ipv4_octet_1",
            "ipv4_octet_2",
            "ipv4_octet_3",
            "ipv4_octet_4",
            "is_private",
            "is_global",
            "is_multicast",
            "is_loopback",
        ],
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_graph_arrays(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
