from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_COLUMNS = [
    "ts",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "label",
]

IP_COLUMNS = ["id.orig_h", "id.resp_h"]
CATEGORICAL_COLUMNS = ["proto", "service", "conn_state", "history"]
NUMERIC_COLUMNS = [
    "ts",
    "id.orig_p",
    "id.resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]

TARGET_COLUMN = "binary_label"
RANDOM_STATE = 42


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    df = chunk.copy()
    df = df.replace("-", np.nan)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("missing").astype(str)

    for col in IP_COLUMNS:
        df[col] = df[col].fillna("unknown").astype(str)

    df["label"] = df["label"].astype(str)
    df[TARGET_COLUMN] = (df["label"] != "Benign").astype(int)
    return df


def update_reservoir(
    reservoir: pd.DataFrame,
    candidates: pd.DataFrame,
    target_size: int,
) -> pd.DataFrame:
    if candidates.empty:
        return reservoir

    combined = pd.concat([reservoir, candidates], ignore_index=True)
    return combined.nsmallest(target_size, "_sample_key").reset_index(drop=True)


def make_binary_sample(
    data_path: Path,
    rows_per_class: int,
    chunk_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(random_state)

    benign_reservoir = pd.DataFrame()
    attack_reservoir = pd.DataFrame()
    total_rows_seen = 0

    for chunk in pd.read_csv(
        data_path,
        usecols=RAW_COLUMNS,
        chunksize=chunk_size,
        dtype=str,
    ):
        chunk = clean_chunk(chunk)
        total_rows_seen += len(chunk)
        chunk["_sample_key"] = rng.random(len(chunk))

        benign_candidates = chunk[chunk[TARGET_COLUMN] == 0]
        attack_candidates = chunk[chunk[TARGET_COLUMN] == 1]

        benign_reservoir = update_reservoir(
            benign_reservoir,
            benign_candidates,
            rows_per_class,
        )
        attack_reservoir = update_reservoir(
            attack_reservoir,
            attack_candidates,
            rows_per_class,
        )

    sample = pd.concat([benign_reservoir, attack_reservoir], ignore_index=True)
    sample = sample.drop(columns=["_sample_key"])
    sample = sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    summary = {
        "total_rows_seen": int(total_rows_seen),
        "requested_rows_per_class": int(rows_per_class),
        "actual_rows": int(len(sample)),
        "binary_label_counts": {
            str(key): int(value)
            for key, value in sample[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "raw_label_counts": {
            str(key): int(value)
            for key, value in sample["label"].value_counts().items()
        },
    }
    return sample, summary


def save_splits(
    sample: pd.DataFrame,
    output_dir: Path,
    random_state: int,
) -> dict:
    train_df, temp_df = train_test_split(
        sample,
        test_size=0.30,
        random_state=random_state,
        stratify=sample[TARGET_COLUMN],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df[TARGET_COLUMN],
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "iot23_binary_sample.csv"
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    sample.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "full": str(full_path),
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "split_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }


def write_metadata(output_dir: Path, summary: dict, paths: dict, args: argparse.Namespace) -> None:
    metadata = {
        "task": "binary_flow_classification",
        "label_mapping": {
            "Benign": 0,
            "Malicious": 1,
        },
        "source_file": str(args.data_path),
        "target_column": TARGET_COLUMN,
        "raw_label_column": "label",
        "ip_columns": IP_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "random_state": int(args.random_state),
        "chunk_size": int(args.chunk_size),
        "summary": summary,
        "paths": paths,
    }

    metadata_path = output_dir / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Create a balanced binary IoT-23 sample.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root / "data" / "iot23_combined_new.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "processed",
    )
    parser.add_argument("--rows-per-class", type=int, default=50_000)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample, summary = make_binary_sample(
        data_path=args.data_path,
        rows_per_class=args.rows_per_class,
        chunk_size=args.chunk_size,
        random_state=args.random_state,
    )
    paths = save_splits(
        sample=sample,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
    write_metadata(args.output_dir, summary, paths, args)

    print(json.dumps({"summary": summary, "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
