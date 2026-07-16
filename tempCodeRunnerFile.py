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