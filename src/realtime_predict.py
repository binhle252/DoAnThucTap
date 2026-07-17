from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from build_graph_arrays import (
    build_edge_index,
    build_node_features,
    build_node_mapping,
)

from train_gat import GATEdgeClassifier

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "gat_edge_classifier_seed_42.pt"

PREPROCESSOR_PATH = ROOT / "data" / "processed" / "edge_preprocessor.joblib"

NODE_MAPPING_PATH = ROOT / "data" / "processed" / "node_mapping.csv"

GRAPH_PATH = ROOT / "data" / "processed" / "graph_arrays.npz"

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    config = checkpoint["config"]

    model = GATEdgeClassifier(
        node_in_channels=config["node_in_channels"],
        edge_in_channels=config["edge_in_channels"],
        hidden_channels=config["hidden_channels"],
        heads=config["heads"],
        layers=config["layers"],
        dropout=config["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return (
        model,
        checkpoint["threshold"],
    )

def load_preprocessor():

    return joblib.load(
        PREPROCESSOR_PATH
    )

def load_graph_assets():

    graph_npz = np.load(GRAPH_PATH)

    graph = {
        "x": graph_npz["x"],
        "edge_attr": graph_npz["edge_attr"],
        "edge_index": graph_npz["edge_index"],
        "y": graph_npz["y"],
        "train_mask": graph_npz["train_mask"],
        "val_mask": graph_npz["val_mask"],
        "test_mask": graph_npz["test_mask"],
    }

    node_mapping = pd.read_csv(
        NODE_MAPPING_PATH
    )

    return graph, node_mapping

def preprocess_flow(
    flow_df: pd.DataFrame,
):

    preprocessor = load_preprocessor()

    edge_attr = preprocessor.transform(
        flow_df
    ).astype(np.float32)

    return edge_attr

class RealtimeGraph:

    def __init__(self):

        self.graph, self.node_mapping = load_graph_assets()

        self.preprocessor = load_preprocessor()

        self.model, self.threshold = load_model()

        self.flow_history = pd.read_csv(
            ROOT / "data" / "processed" / "iot23_binary_sample.csv"
        )

        self.ip_to_node = dict(
            zip(
                self.node_mapping["ip"],
                self.node_mapping["node_id"],
            )
        )

        print("Realtime engine initialized.")

    def get_node_id(self, ip):

        if ip in self.ip_to_node:
            return self.ip_to_node[ip], False

        new_id = len(self.ip_to_node)

        self.ip_to_node[ip] = new_id

        self.node_mapping.loc[len(self.node_mapping)] = [
            ip,
            new_id,
        ]

        return new_id, True

    def add_flow(self, flow_df):

        self.flow_history = pd.concat(
            [
                self.flow_history,
                flow_df,
            ],
            ignore_index=True,
        )

        edge_attr = self.preprocessor.transform(
            flow_df
        ).astype(np.float32)

        src, new_src = self.get_node_id(
            flow_df.iloc[0]["id.orig_h"]
        )

        dst, new_dst = self.get_node_id(
            flow_df.iloc[0]["id.resp_h"]
        )

        if new_src:

            self.graph["x"] = np.vstack([
                self.graph["x"],
                self.create_new_node_feature(
                    flow_df.iloc[0]["id.orig_h"]
                ),
            ])

        if new_dst:

            self.graph["x"] = np.vstack([
                self.graph["x"],
                self.create_new_node_feature(
                    flow_df.iloc[0]["id.resp_h"]
                ),
            ])

        new_edge = np.array(
            [
                [src],
                [dst],
            ],
            dtype=np.int64,
        )

        self.graph["edge_index"] = np.hstack(
            [
                self.graph["edge_index"],
                new_edge,
            ]
        )

        self.graph["edge_attr"] = np.vstack(
            [
                self.graph["edge_attr"],
                edge_attr,
            ]
        )

        print(
            f"Graph Updated | Nodes={len(self.graph['x'])} "
            f"Edges={len(self.graph['edge_attr'])}"
        )

    def create_new_node_feature(self, ip):

        ip_parts = ip.split(".")

        feature = np.zeros(
            self.graph["x"].shape[1],
            dtype=np.float32,
        )

        if len(ip_parts) == 4:

            feature[0] = int(ip_parts[0])
            feature[1] = int(ip_parts[1])
            feature[2] = int(ip_parts[2])
            feature[3] = int(ip_parts[3])

        feature[4] = 1

        return feature.reshape(1, -1)

    def predict_last_flow(self):

        from torch_geometric.data import Data

        data = Data(
            x=torch.tensor(
                self.graph["x"],
                dtype=torch.float,
            ),
            edge_index=torch.tensor(
                self.graph["edge_index"],
                dtype=torch.long,
            ),
            edge_attr=torch.tensor(
                self.graph["edge_attr"],
                dtype=torch.float,
            ),
        )

        with torch.no_grad():

            logits = self.model(
                data.x,
                data.edge_index,
                data.edge_attr,
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

        probability = probs[-1].item()

        prediction = (
            "MALICIOUS"
            if probability >= self.threshold
            else "BENIGN"
        )

        print()
        print(f"Probability : {probability:.6f}")
        print(f"Threshold   : {self.threshold:.6f}")
        print(f"Prediction  : {prediction}")

        return probability, prediction

if __name__ == "__main__":

    engine = RealtimeGraph()

    print()

    print("Node feature:", engine.graph["x"].shape)
    print("Edge feature:", engine.graph["edge_attr"].shape)
    print("Edge index:", engine.graph["edge_index"].shape)

    print()

    print(engine.node_mapping.head())

    print("\n========== START REALTIME ==========\n")

    test_df = pd.read_csv(
        ROOT / "data" / "processed" / "test.csv"
    ).head(30)

    for i, (_, row) in enumerate(test_df.iterrows(), start=1):

        print(f"\nFlow {i}")

        flow = pd.DataFrame([row])

        engine.add_flow(flow)

        engine.predict_last_flow()

        print("-" * 60)
    

