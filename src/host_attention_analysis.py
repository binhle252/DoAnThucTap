from pathlib import Path

import pandas as pd


def analyze_host_attention(
    attention_path: Path,
    output_path: Path,
):
    # Đọc attention đã map sang IP
    attention = pd.read_csv(attention_path)

    # Thống kê theo Source IP
    host_summary = (
        attention.groupby("src_ip")
        .agg(
            connection_count=("attention_mean", "count"),
            mean_attention=("attention_mean", "mean"),
            max_attention=("attention_mean", "max"),
            min_attention=("attention_mean", "min"),
            std_attention=("attention_mean", "std"),
        )
        .reset_index()
    )

    # Sắp xếp giảm dần theo Attention trung bình
    host_summary = host_summary.sort_values(
        by="mean_attention",
        ascending=False,
    )

    # Lưu kết quả
    host_summary.to_csv(output_path, index=False)

    print("\n===== TOP 20 HOSTS =====\n")
    print(host_summary.head(20))


if __name__ == "__main__":
    analyze_host_attention(
        Path("results/attention_with_ip.csv"),
        Path("results/host_attention_summary.csv"),
    )