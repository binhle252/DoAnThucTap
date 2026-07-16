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

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

MODEL_PATH = Path("models/gat_edge_classifier_seed_45.pt")
PREPROCESSOR_PATH = Path("data/processed/edge_preprocessor.joblib")
NODE_MAPPING_PATH = Path("data/processed/node_mapping.csv")
ARRAYS_PATH = Path("data/processed/graph_arrays.npz")

@st.cache_resource
def load_resources():

    model, device, threshold, config = load_model(
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
        config,
        preprocessor,
        node_mapping,
    )
model, device, threshold, config, preprocessor, node_mapping = load_resources()
metrics = load_metrics()
st.title("🛡️ Intrusion Detection System")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Model", "GAT")
c2.metric("Best Epoch", metrics["best_epoch"])
c3.metric("Threshold", f"{threshold:.3f}")
c4.metric("Dataset", "IoT-23")

st.divider()

left, right = st.columns(
    [2.2,1],
    gap="large",
)

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

        st.subheader("📈 Traffic Statistics")

        if "label" in df.columns:

            benign = (df["label"] == "Benign").sum()
            attack = len(df) - benign

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Benign",
                benign,
            )

            c2.metric(
                "Attack",
                attack,
            )

            c3.metric(
                "Attack %",
                f"{attack / len(df) * 100:.2f}%"
            )
        
        if "proto" in df.columns:

            proto = df["proto"].value_counts()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "TCP",
                int(proto.get("tcp", 0))
            )

            c2.metric(
                "UDP",
                int(proto.get("udp", 0))
            )

            c3.metric(
                "ICMP",
                int(proto.get("icmp", 0))
            )

        c1, c2 = st.columns(2)

        c1.metric(
            "Unique Source IP",
            df["id.orig_h"].nunique()
        )

        c2.metric(
            "Unique Destination IP",
            df["id.resp_h"].nunique()
        )

        st.subheader("Uploaded Traffic")

        st.caption(
            "Preview of uploaded network traffic flows."
        )

        display_columns = [
            "ts",
            "id.orig_h",
            "id.resp_h",
            "proto",
            "conn_state",
            "label",
        ]

        protocol = st.selectbox(
            "Protocol",
            ["All"] + sorted(df["proto"].unique().tolist())
        )

        display_df = df

        if protocol != "All":
            display_df = display_df[
                display_df["proto"] == protocol
            ]

        search_ip = st.text_input(
            "Search IP"
        )

        if search_ip:

            display_df = display_df[
                display_df["id.orig_h"].astype(str).str.contains(search_ip)
                |
                display_df["id.resp_h"].astype(str).str.contains(search_ip)
            ]

        st.dataframe(
            display_df[display_columns],
            height=260,
            use_container_width=True,
        )

        st.subheader("🔍 Select Flow")

        selected_index = st.selectbox(
            "Choose one flow",
            display_df.index,
        )

        selected_flow = display_df.loc[selected_index]

        st.subheader("📄 Flow Information")

        c1, c2 = st.columns(2)

        c1.metric(
            "Source",
            selected_flow["id.orig_h"],
        )

        c2.metric(
            "Destination",
            selected_flow["id.resp_h"],
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Protocol",
            selected_flow["proto"],
        )

        c2.metric(
            "State",
            selected_flow["conn_state"],
        )

        if "label" in selected_flow.index:
            c3.metric(
                "Ground Truth",
                selected_flow["label"],
            )

        flow_info = pd.DataFrame(
            {
                "Field": selected_flow.index.astype(str),
                "Value": selected_flow.astype(str).values,
            }
        )

        st.dataframe(
            flow_info,
            use_container_width=True
        )

        predict_clicked = st.button(
            "🚀 Analyze Selected Flow",
            type="primary",
        )

        if "history" not in st.session_state:
            st.session_state.history = []

        if predict_clicked:

            model, device, threshold, config, preprocessor, node_mapping = load_resources()

            result = predict_flow(
                flow=selected_flow.to_dict(),
                model=model,
                device=device,
                threshold=threshold,
                preprocessor=preprocessor,
                node_mapping=node_mapping.copy(),
                arrays_path=ARRAYS_PATH,
            )

            st.session_state.history.append(
                {
                    "Source": selected_flow["id.orig_h"],
                    "Destination": selected_flow["id.resp_h"],
                    "Protocol": selected_flow["proto"],
                    "Prediction": result["predicted_label"],
                    "Probability": round(
                        result["malicious_probability"],
                        4,
                    ),
                }
            )

            st.divider()

            st.header("🚨 Detection Result")

            c1,c2,c3 = st.columns(3)
            c4,c5,c6 = st.columns(3)

            confidence = max(
                result["malicious_probability"],
                result["benign_probability"],
            )

            score = result["malicious_probability"]

            if score >= 0.90:
                risk = "🔴 Critical"

            elif score >= 0.70:
                risk = "🟠 High"

            elif score >= 0.40:
                risk = "🟡 Medium"

            else:
                risk = "🟢 Low"

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

            c5.metric(
                "Benign Probability",
                f"{result['benign_probability']*100:.2f}%"
            )

            c6.metric(
                "Risk Level",
                risk,
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

            st.subheader("📝 Explanation")

            if result["predicted_class"] == 1:

                st.info(
                    f"""
            The Graph Attention Network classified this flow as **Malicious**.

            Reason:

            • Malicious probability = {result['malicious_probability']*100:.2f}%

            • Decision threshold = {result['threshold']*100:.2f}%

            Since the probability exceeds the threshold,
            the flow is classified as an attack.
            """
                )

            else:

                st.info(
                    f"""
            The Graph Attention Network classified this flow as **Benign**.

            Reason:

            • Malicious probability = {result['malicious_probability']*100:.2f}%

            • Decision threshold = {result['threshold']*100:.2f}%

            Since the probability is below the threshold,
            the flow is considered normal.
            """
                )
            
            if result["predicted_class"] == 1:

                st.warning("""
            Recommended actions

            • Inspect source IP

            • Inspect destination IP

            • Check firewall logs

            • Monitor similar traffic

            • Block connection if necessary
            """)
            else:

                st.success("""
            Recommended actions

            • No immediate action required

            • Continue monitoring

            • Store this traffic for future analysis
            """)
            
            st.divider()

            st.subheader("Prediction History")
            history_df = pd.DataFrame(
                st.session_state.history
            )

            st.dataframe(
                history_df,
                use_container_width=True
            )
            if len(history_df):

                c1,c2,c3 = st.columns(3)

                c1.metric(
                    "Analyzed",
                    len(history_df)
                )

                c2.metric(
                    "Attack",
                    (
                        history_df["Prediction"]
                        == "Malicious"
                    ).sum()
                )

                c3.metric(
                    "Benign",
                    (
                        history_df["Prediction"]
                        == "Benign"
                    ).sum()
                )

            csv = history_df.to_csv(
                index=False,
            ).encode("utf-8")

            st.download_button(
                "⬇ Download Prediction Report",
                csv,
                file_name="prediction_report.csv",
                mime="text/csv",
            )

            if st.button(
                "🗑 Clear History"
            ):

                st.session_state.history = []

                st.rerun()

            st.subheader("Graph Attention Analysis")

            attention_image = Path(
                "results/graph_attention.png"
            )

            if attention_image.exists():

                st.subheader("Attention Graph")

                st.image(
                    attention_image,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Attention graph not found."
                )

            attention_csv = Path(
                "results/attention_with_ip.csv"
            )

            if attention_csv.exists():

                import pandas as pd

                attention_df = pd.read_csv(
                    attention_csv
                )
            
            top_attention = attention_df.sort_values(
                "attention_mean",
                ascending=False,
            ).head(10)

            st.subheader(
                "Top Attention Connections"
            )

            st.dataframe(
                top_attention[
                    [
                        "src_ip",
                        "dst_ip",
                        "attention_mean",
                    ]
                ],
                use_container_width=True
            )

            st.subheader(
                "Attention Distribution"
            )

            fig, ax = plt.subplots(
                figsize=(7,3)
            )

            ax.hist(
                attention_df["attention_mean"],
                bins=30,
            )

            ax.set_xlabel(
                "Attention"
            )

            ax.set_ylabel(
                "Connections"
            )

            st.pyplot(fig)

            st.write(
                f"{result['malicious_probability']*100:.2f}%"
            )

            st.write("Selected index:", selected_index)

            st.write("Source:", selected_flow["id.orig_h"])

            st.write("Destination:", selected_flow["id.resp_h"])

with right:

    st.header("📈 Model Performance")

    metrics = load_metrics()

    if metrics:

        c1, c2 = st.columns(2)

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy']*100:.2f}%"
        )

        c2.metric(
            "Precision",
            f"{metrics['precision']*100:.2f}%"
        )

        c3, c4 = st.columns(2)

        c3.metric(
            "Recall",
            f"{metrics['recall']*100:.2f}%"
        )

        c4.metric(
            "F1-score",
            f"{metrics['f1']*100:.2f}%"
        )

        st.divider()

        st.subheader("Training")

        c1, c2 = st.columns(2)

        c1.metric(
            "Best Epoch",
            metrics["best_epoch"],
        )

        c2.metric(
            "Threshold",
            f"{metrics['threshold']:.3f}",
        )

        c3, c4 = st.columns(2)

        c3.metric(
            "ROC AUC",
            f"{metrics['roc_auc']:.3f}",
        )

        c4.metric(
            "PR AUC",
            f"{metrics['pr_auc']:.3f}",
        )

        st.divider()

        st.subheader("📈 Training History")
        history = metrics["history"]

        epochs = [
            h["epoch"]
            for h in history
        ]

        loss = [
            h["loss"]
            for h in history
        ]

        val_f1 = [
            h["val_f1"]
            for h in history
        ]

        fig, ax = plt.subplots(figsize=(6,3))

        ax.plot(
            epochs,
            loss,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6,3))

        ax.plot(
            epochs,
            val_f1,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation F1")

        st.pyplot(fig)

        st.divider()

        st.subheader("⚙ Model Configuration")

        config_df = pd.DataFrame({
            "Parameter":[
                "Heads",
                "Layers",
                "Hidden",
                "Threshold",
                "Node Features",
                "Edge Features",
            ],
            "Value":[
                config["heads"],
                config["layers"],
                config["hidden_channels"],
                f"{threshold:.3f}",
                config["node_in_channels"],
                config["edge_in_channels"],
            ]
        })

        st.dataframe(
            config_df,
            hide_index=True,
            use_container_width=True,
        )

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