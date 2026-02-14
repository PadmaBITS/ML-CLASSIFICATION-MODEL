import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "XGBoost (Ensemble)": "xgboost_ensemble.joblib",
}


@st.cache_resource
def load_model(model_name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


@st.cache_data
def load_metadata():
    with (MODEL_DIR / "dataset_metadata.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


@st.cache_data
def load_benchmark_metrics():
    return pd.read_csv(MODEL_DIR / "metrics.csv")


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def render_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="ML Assignment 2 - Classifier App", layout="wide")
    st.title("ML Assignment 2: Classification Model Explorer")

    metadata = load_metadata()
    metrics_df = load_benchmark_metrics()

    st.markdown(
        f"""
        **Dataset:** {metadata['dataset_name']}  
        **Task:** {metadata['task_type']}  
        **Shape:** {metadata['num_instances']} rows, {metadata['num_features']} features
        """
    )

    st.subheader("Benchmark Metrics on Holdout Test Set")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Model Selection")
    selected_model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    model = load_model(selected_model_name)

    st.subheader("Dataset Upload (CSV)")
    st.caption("Upload test data CSV. Include the target column 'target' for evaluation metrics.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded successfully.")
    else:
        df = pd.read_csv(DATA_DIR / "test_data.csv")
        st.info("No file uploaded. Using bundled test_data.csv for demo.")

    st.write("Preview of input data")
    st.dataframe(df.head(), use_container_width=True)

    feature_cols = metadata["feature_columns"]
    target_col = metadata["target_column"]

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        st.error(
            "Uploaded CSV is missing required feature columns. "
            f"Missing: {missing_features[:8]}{'...' if len(missing_features) > 8 else ''}"
        )
        st.stop()

    X = df[feature_cols]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    out_df = df.copy()
    out_df["prediction"] = y_pred

    st.subheader("Predictions")
    st.dataframe(out_df.head(20), use_container_width=True)

    if target_col in df.columns:
        y_true = df[target_col]
        metrics = compute_metrics(y_true, y_pred, y_prob)

        st.subheader("Evaluation Metrics")
        cols = st.columns(6)
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            cols[idx].metric(metric_name, f"{metric_value:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        render_confusion_matrix(cm)

        st.subheader("Classification Report")
        cls_report = classification_report(y_true, y_pred, output_dict=False)
        st.code(cls_report)
    else:
        st.warning("No 'target' column detected, so evaluation metrics are skipped.")


if __name__ == "__main__":
    main()
