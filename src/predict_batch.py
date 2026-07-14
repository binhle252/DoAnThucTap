from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.build_graph_arrays import (
    build_edge_index,
    build_ip_node_features,
    build_node_mapping,
)
from src.demo_predict import clean_flow, load_model

def predict_batch_dataframe(
    df: pd.DataFrame,
    model_path: Path,
    preprocessor_path: Path,
    cpu: bool = False,
) -> pd.DataFrame:
    model, device, threshold, _ = load_model(
        model_path,
        cpu,
    )

    df = clean_flow(df)

    preprocessor = joblib.load(preprocessor_path)