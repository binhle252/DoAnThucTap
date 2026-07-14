from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap


def visualize_graph(
    attention_file: Path,
    output_file: Path,
    top_edges: int = 30,
):
    # =====================
    # Load data
    # =====================
    df = pd.read_csv(attention_file)

    df = (
        df.sort_values(
            "attention_mean",
            ascending=False,
        )
        .head(top_edges)
    )

    # =====================
    # Build Graph
    # =====================
    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(
            row["src_ip"],
            row["dst_ip"],
            weight=row["attention_mean"],
        )

    # =====================
    # Layout
    # =====================
    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.2,
        iterations=300,
    )

    fig, ax = plt.subplots(
        figsize=(16,12),
        facecolor="#f5f5f5"
    )

    ax.set_facecolor("#f5f5f5")

    # =====================
    # Degree
    # =====================
    degree = dict(G.degree())

    # Node size
    node_sizes = [
        350 + degree[node] * 120
        for node in G.nodes()
    ]

    # Node color
    node_colors = []

    for node in G.nodes():

        if degree[node] >= 6:
            node_colors.append("#d73027")      # đỏ

        elif degree[node] >= 3:
            node_colors.append("#fc8d59")      # cam

        else:
            node_colors.append("#91bfdb")      # xanh

    # =====================
    # Edge style
    # =====================
    edge_weights = [
        G[u][v]["weight"]
        for u, v in G.edges()
    ]

    edge_width = [
        1 + w * 6
        for w in edge_weights
    ]

    norm = mcolors.Normalize(
        vmin=min(edge_weights),
        vmax=max(edge_weights),
    )

    reds = plt.cm.Reds

    new_reds = LinearSegmentedColormap.from_list(
        "new_reds",
        reds(np.linspace(0.30, 1, 256))
    )

    edge_colors = edge_weights

    # =====================
    # Draw
    # =====================
    

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_width,
        edge_color=edge_colors,
        edge_cmap=new_reds,
        edge_vmin=min(edge_weights),
        edge_vmax=max(edge_weights),
        arrows=True,
        arrowsize=16,
        alpha=0.85,
        ax=ax,
    )

    # =====================
    # Label only top hubs
    # =====================
    top_nodes = sorted(
        degree,
        key=degree.get,
        reverse=True,
    )[:8]

    labels = {
        node: node
        for node in top_nodes
    }

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=9,
        font_weight="bold",
        ax=ax,
    )

    # =====================
    # Color bar
    # =====================
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Reds,
        norm=norm,
    )

    sm.set_array([])

    fig.colorbar(
        sm,
        ax=ax,
        label="Attention Weight",
        shrink=0.75,
    )

    ax.set_title(
        "Top 30 Network Connections by GAT Attention",
        fontsize=18,
        fontweight="bold",
    )

    ax.axis("off")
    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":

    visualize_graph(
        Path("results/attention_with_ip.csv"),
        Path("results/graph_attention.png"),
        top_edges=30,
    )