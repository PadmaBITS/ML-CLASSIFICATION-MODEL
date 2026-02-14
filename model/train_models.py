from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def build_models():
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "kNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    y = raw.target.copy()

    full_df = X.copy()
    full_df["target"] = y
    full_df.to_csv(DATA_DIR / "full_dataset.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(DATA_DIR / "test_data.csv", index=False)

    summary = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "scikit-learn (UCI original source)",
        "task_type": "Binary Classification",
        "num_instances": int(full_df.shape[0]),
        "num_features": int(X.shape[1]),
        "target_column": "target",
        "target_mapping": {"0": raw.target_names[0], "1": raw.target_names[1]},
        "feature_columns": list(X.columns),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / "dataset_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics_rows = []
    report_dump = {}

    models = build_models()

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics_row = {"ML Model Name": model_name, **metrics}
        metrics_rows.append(metrics_row)

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

        report_dump[model_name] = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(MODEL_DIR / "metrics.csv", index=False)

    with (MODEL_DIR / "evaluation_reports.json").open("w", encoding="utf-8") as fp:
        json.dump(report_dump, fp, indent=2)

    print("Training complete. Files generated:")
    print(f"- {MODEL_DIR / 'metrics.csv'}")
    print(f"- {MODEL_DIR / 'evaluation_reports.json'}")
    print(f"- {DATA_DIR / 'test_data.csv'}")


if __name__ == "__main__":
    main()
