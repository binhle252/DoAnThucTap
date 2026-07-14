from pathlib import Path

import pandas as pd

def export_attention(
    attention_path: Path,
    mapping_path: Path,
    output_path: Path,
):
    attention = pd.read_csv(attention_path)
    mapping = pd.read_csv(mapping_path)
    id_to_ip = dict(zip(mapping["node_id"], mapping["ip"]))
    attention["src_ip"] = attention["src_node"].map(id_to_ip)
    attention["dst_ip"] = attention["dst_node"].map(id_to_ip)
    attention = attention[
        attention["src_node"] != attention["dst_node"]
    ]
    attention = attention[
        ~attention["src_ip"].str.startswith("ff02::", na=False)
    ]   

    attention = attention[
        ~attention["dst_ip"].str.startswith("ff02::", na=False)
    ]

    attention = attention[
        attention["src_ip"] != "255.255.255.255"
    ]

    attention = attention[
        attention["dst_ip"] != "255.255.255.255"
    ]

    attention = attention[
        attention["src_ip"] != "0.0.0.0"
    ]

    attention = attention[
        attention["dst_ip"] != "0.0.0.0"
    ]
    attention = attention.sort_values(
        "attention_mean",
        ascending=False,
    )
    top20 = attention.head(20)

    top20.to_csv(
        "results/top20_attention.csv",
        index=False,
    )

    print(top20[
        [
            "src_ip",
            "dst_ip",
            "attention_mean",
        ]
    ])

    attention.to_csv(
        output_path,
        index=False,
    )

if __name__ == "__main__":
    export_attention(
        Path("results/attention.csv"),
        Path("data/processed/node_mapping.csv"),
        Path("results/attention_with_ip.csv"),
    )