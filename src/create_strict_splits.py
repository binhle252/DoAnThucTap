from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from src.preprocess_binary import TARGET_COLUMN
except ModuleNotFoundError:
    from preprocess_binary import TARGET_COLUMN


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def split_counts(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "binary_label_counts": {
            str(key): int(value)
            for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "raw_label_counts": {
            str(key): int(value)
            for key, value in df["label"].value_counts().items()
        },
        "ts_min": float(df["ts"].min()) if len(df) else None,
        "ts_max": float(df["ts"].max()) if len(df) else None,
    }


def sample_from_pool(pool: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    if n_rows <= 0:
        return pool.iloc[:0].copy()
    if len(pool) <= n_rows:
        return pool.copy()
    return pool.sample(n=n_rows, random_state=random_state).sort_values("ts")


def create_class_time_split(
    df: pd.DataFrame,
    rows_per_class: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    test_parts = []

    for label_value, class_df in df.groupby(TARGET_COLUMN):
        class_df = class_df.sort_values("ts").reset_index(drop=True)
        n = len(class_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_pool = class_df.iloc[:train_end]
        val_pool = class_df.iloc[train_end:val_end]
        test_pool = class_df.iloc[val_end:]

        if rows_per_class is None:
            train_n = len(train_pool)
            val_n = len(val_pool)
            test_n = len(test_pool)
        else:
            train_n = int(rows_per_class * 0.70)
            val_n = int(rows_per_class * 0.15)
            test_n = rows_per_class - train_n - val_n

        offset = int(label_value) * 1000
        train_parts.append(sample_from_pool(train_pool, train_n, random_state + offset + 1))
        val_parts.append(sample_from_pool(val_pool, val_n, random_state + offset + 2))
        test_parts.append(sample_from_pool(test_pool, test_n, random_state + offset + 3))

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=random_state)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=random_state)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=random_state)
    return train_df, val_df, test_df


def create_chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def save_split_files(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    paths = {
        "full": output_dir / "iot23_binary_sample.csv",
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }
    full_df.to_csv(paths["full"], index=False)
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    return {key: str(value) for key, value in paths.items()}


def create_strict_splits(args: argparse.Namespace) -> dict:
    df = pd.read_csv(args.input_path)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0.0)

    if args.strategy == "class_time":
        train_df, val_df, test_df = create_class_time_split(
            df=df,
            rows_per_class=args.rows_per_class,
            random_state=args.random_state,
        )
    elif args.strategy == "chronological":
        train_df, val_df, test_df = create_chronological_split(df)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    paths = save_split_files(train_df, val_df, test_df, args.output_dir)
    metadata = {
        "strategy": args.strategy,
        "source": str(args.input_path),
        "rows_per_class": args.rows_per_class,
        "random_state": args.random_state,
        "strict_split_note": (
            "class_time keeps each class balanced while preserving temporal order inside each class"
            if args.strategy == "class_time"
            else "chronological uses global timestamp order and may have strong class drift"
        ),
        "splits": {
            "train": split_counts(train_df),
            "val": split_counts(val_df),
            "test": split_counts(test_df),
        },
        "paths": paths,
    }
    (args.output_dir / "strict_split_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Create stricter time-aware train/val/test splits.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=root / "data" / "processed" / "iot23_binary_sample.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "strict_time_balanced",
    )
    parser.add_argument(
        "--strategy",
        choices=["class_time", "chronological"],
        default="class_time",
    )
    parser.add_argument("--rows-per-class", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = create_strict_splits(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
