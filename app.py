import streamlit as st
from src.dashboard_utils import load_metrics
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
from src.demo_predict import (
    predict_flow,
    load_model,
    load_node_mapping,
)
import joblib

st.set_page_config(
    page_title="IDS using GAT",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path("models/gat_edge_classifier.pt")
PREPROCESSOR_PATH = Path("data/processed/edge_preprocessor.joblib")
NODE_MAPPING_PATH = Path("data/processed/node_mapping.csv")
ARRAYS_PATH = Path("data/processed/graph_arrays.npz")

@st.cache_resource
def load_resources():

    model, device, threshold, _ = load_model(
        MODEL_PATH,
        cpu=False,
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH,
    )

    node_mapping = load_node_mapping(
        NODE_MAPPING_PATH,
    )

    return (
        model,
        device,
        threshold,
        preprocessor,
        node_mapping,
    )

st.title("🛡️ Intrusion Detection System using Graph Attention Network")

st.markdown(
"""
Hệ thống phát hiện hành vi độc hại trong mạng máy tính
sử dụng Graph Attention Network (GAT)
với tập dữ liệu IoT-23.
"""
)

left, right = st.columns([2,1])

with left:

    st.header("📂 Upload Traffic")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        import pandas as pd

        df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded.")

        st.subheader("📊 Dataset Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Number of Flows",
            len(df),
        )

        col2.metric(
            "Number of Features",
            df.shape[1],
        )

        col3.metric(
            "Memory Usage",
            f"{df.memory_usage(deep=True).sum()/1024/1024:.2f} MB",
        )

        st.subheader("Uploaded Traffic")

        display_columns = [
            "ts",
            "id.orig_h",
            "id.resp_h",
            "proto",
            "conn_state",
            "label",
        ]

        st.dataframe(
            df[display_columns],
            height=350,
            width="stretch",
        )

        st.subheader("Traffic Selection")

        selected_row = st.number_input(
            "Select Row",
            min_value=0,
            max_value=len(df)-1,
            value=0,
        )
        selected_flow = df.iloc[selected_row]
        st.subheader("Selected Flow")

        st.json(selected_flow.to_dict())
        predict_clicked = st.button(
            "🛡 Predict Selected Flow",
            type="primary",
        )

        if predict_clicked:

            model, device, threshold, preprocessor, node_mapping = load_resources()

            result = predict_flow(
                flow=selected_flow.to_dict(),
                model=model,
                device=device,
                threshold=threshold,
                preprocessor=preprocessor,
                node_mapping=node_mapping.copy(),
                arrays_path=ARRAYS_PATH,
            )

            st.divider()

            st.header("🚨 Detection Result")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            confidence = max(
                result["malicious_probability"],
                result["benign_probability"],
            )

            c1.metric(
                "Prediction",
                result["predicted_label"],
            )

            c2.metric(
                "Malicious Probability",
                f"{result['malicious_probability']*100:.2f}%"
            )

            c3.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            c4.metric(
                "Threshold",
                f"{result['threshold']:.2f}"
            )

            if result["predicted_class"] == 1:

                st.error(
                    "⚠ Potential malicious traffic detected."
                )

            else:

                st.success(
                    "✓ Traffic classified as benign."
                )
            
            st.subheader("Malicious Score")

            st.progress(
                float(result["malicious_probability"])
            )

            st.write(
                f"{result['malicious_probability']*100:.2f}%"
            )

with right:

    st.header("📈 Model Performance")

    metrics = load_metrics()

    if metrics:

        st.metric(
            "Accuracy",
            f"{metrics['accuracy']*100:.2f}%"
        )

        st.metric(
            "Precision",
            f"{metrics['precision']*100:.2f}%"
        )

        st.metric(
            "Recall",
            f"{metrics['recall']*100:.2f}%"
        )

        st.metric(
            "F1-score",
            f"{metrics['f1']*100:.2f}%"
        )

        st.divider()

        st.subheader("⚙ Model Configuration")

        st.write("**Architecture:** Graph Attention Network")

        st.write("**Heads:** 4")

        st.write("**Hidden Channels:** 64")

        st.write("**Layers:** 2")

        st.write("**Node Features:** IP + Statistics")

        st.write("**Edge Features:** 54")

    else:

        st.warning("No trained model found.")
        st.divider()

        st.header("🔄 Detection Pipeline")

        st.markdown("""
        **Traffic Flow**

        ⬇

        **Graph Construction**

        ⬇

        **Graph Attention Network**

        ⬇

        **Edge Classification**

        ⬇

        **Malicious / Benign**
        """)    