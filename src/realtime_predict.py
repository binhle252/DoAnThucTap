from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.build_graph_arrays import ip_to_features

from src.train_gat import GATEdgeClassifier

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "gat_edge_classifier_seed_42.pt"

PREPROCESSOR_PATH = ROOT / "data" / "processed" / "edge_preprocessor.joblib"

NODE_MAPPING_PATH = ROOT / "data" / "processed" / "node_mapping.csv"

GRAPH_PATH = ROOT / "data" / "processed" / "graph_arrays.npz"

NODE_SCALER_PATH = (
    ROOT /
    "data" /
    "processed" /
    "node_stat_scaler.joblib"
)

RAW_NODE_STATS_PATH = (
    ROOT /
    "data" /
    "processed" /
    "initial_node_stats.npy"
)

def load_node_scaler():

    return joblib.load(
        NODE_SCALER_PATH
    )

def load_raw_node_stats():

    return np.load(
        RAW_NODE_STATS_PATH
    )

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

class NodeStatisticsManager:

    OUT_DEGREE = 0
    IN_DEGREE = 1
    TOTAL_DEGREE = 2
    UNIQUE_OUT = 3
    UNIQUE_IN = 4
    SENT_BYTES = 5
    RECEIVED_BYTES = 6
    SENT_PKTS = 7
    RECEIVED_PKTS = 8
    DURATION = 9
    MEAN_DURATION = 10
    AVG_PKT_SIZE = 11

    def __init__(
        self,
        raw_stats,
        scaler,
    ):
        self.raw_stats = raw_stats.copy()
        self.scaler = scaler

        self.out_neighbors = [
            set()
            for _ in range(len(self.raw_stats))
        ]

        self.in_neighbors = [
            set()
            for _ in range(len(self.raw_stats))
        ]

    def add_node(self):

        self.raw_stats = np.vstack(
            [
                self.raw_stats,
                np.zeros(
                    (
                        1,
                        self.raw_stats.shape[1],
                    ),
                    dtype=np.float32,
                ),
            ]
        )

    def _update_derived_features(
        self,
        node_id,
    ):

        total_degree = self.raw_stats[
            node_id,
            self.TOTAL_DEGREE,
        ]

        duration_sum = self.raw_stats[
            node_id,
            self.DURATION,
        ]

        sent_bytes = self.raw_stats[
            node_id,
            self.SENT_BYTES,
        ]

        received_bytes = self.raw_stats[
            node_id,
            self.RECEIVED_BYTES,
        ]

        sent_pkts = self.raw_stats[
            node_id,
            self.SENT_PKTS,
        ]

        received_pkts = self.raw_stats[
            node_id,
            self.RECEIVED_PKTS,
        ]

        if total_degree > 0:

            self.raw_stats[
                node_id,
                self.MEAN_DURATION,
            ] = duration_sum / total_degree

        else:

            self.raw_stats[
                node_id,
                self.MEAN_DURATION,
            ] = 0.0

        total_packets = sent_pkts + received_pkts

        if total_packets > 0:
            self.raw_stats[
                node_id,
                self.AVG_PKT_SIZE,
            ] = (
                sent_bytes +
                received_bytes
            ) / total_packets

        else:
            self.raw_stats[
                node_id,
                self.AVG_PKT_SIZE,
            ] = 0.0

    def update_flow(
        self,
        src,
        dst,
        orig_bytes,
        resp_bytes,
        orig_pkts,
        resp_pkts,
        duration,
    ):
        self.raw_stats[src, self.OUT_DEGREE] += 1
        self.raw_stats[src, self.TOTAL_DEGREE] += 1

        self.raw_stats[src, self.SENT_BYTES] += orig_bytes
        self.raw_stats[src, self.RECEIVED_BYTES] += resp_bytes

        self.raw_stats[src, self.SENT_PKTS] += orig_pkts
        self.raw_stats[src, self.RECEIVED_PKTS] += resp_pkts

        self.raw_stats[src, self.DURATION] += duration
        self.raw_stats[dst, self.IN_DEGREE] += 1
        self.raw_stats[dst, self.TOTAL_DEGREE] += 1

        self.raw_stats[dst, self.SENT_BYTES] += resp_bytes
        self.raw_stats[dst, self.RECEIVED_BYTES] += orig_bytes

        self.raw_stats[dst, self.SENT_PKTS] += resp_pkts
        self.raw_stats[dst, self.RECEIVED_PKTS] += orig_pkts

        self.raw_stats[dst, self.DURATION] += duration
            
        self._update_derived_features(src)
        self._update_derived_features(dst)

    def get_scaled_features(
        self,
        node_id,
    ):

        raw = self.raw_stats[node_id]

        log_stats = np.log1p(
            np.maximum(
                raw,
                0.0,
            )
        )

        scaled = self.scaler.transform(
            log_stats.reshape(1, -1)
        )

        return scaled.astype(
            np.float32
        )[0]

class RealtimeGraph:

    def __init__(self):


        self.graph, self.node_mapping = load_graph_assets()

        self.preprocessor = load_preprocessor()

        self.model, self.threshold = load_model()

        self.node_scaler = load_node_scaler()

        self.node_stats = NodeStatisticsManager(
            load_raw_node_stats(),
            self.node_scaler,
        )

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

        self.node_stats.add_node()

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

        self.node_stats.update_flow(
            src,
            dst,
            float(flow_df.iloc[0]["orig_bytes"]),
            float(flow_df.iloc[0]["resp_bytes"]),
            float(flow_df.iloc[0]["orig_pkts"]),
            float(flow_df.iloc[0]["resp_pkts"]),
            float(flow_df.iloc[0]["duration"]),
        )

        src_stats = self.node_stats.get_scaled_features(
            src
        )

        dst_stats = self.node_stats.get_scaled_features(
            dst
        )

        self.graph["x"][src,8:] = src_stats

        self.graph["x"][dst,8:] = dst_stats

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
    
    def create_new_node_feature(
        self,
        ip,
    ):
        ip_feature = np.asarray(
            ip_to_features(ip),
            dtype=np.float32,
        )

        stat_feature = np.zeros(
            12,
            dtype=np.float32,
        )

        feature = np.concatenate(
            [
                ip_feature,
                stat_feature,
            ]
        )

        return feature.reshape(1,-1)  

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
    

