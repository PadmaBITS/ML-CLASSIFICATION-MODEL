# ML Assignment 2 - Multi-Model Classification with Streamlit

## a) Problem statement
Build and evaluate six machine learning classification models on one dataset, compare model performance using required metrics, and deploy an interactive Streamlit application to demonstrate predictions and evaluation.

## b) Dataset description
- Dataset: **Breast Cancer Wisconsin (Diagnostic)**
- Source: UCI dataset (loaded via `sklearn.datasets.load_breast_cancer`)
- Task type: Binary classification (`target`: 0 = malignant, 1 = benign)
- Number of instances: **569**
- Number of features: **30**
- Meets assignment constraints: at least 12 features and at least 500 instances

## c) Models used and evaluation
The following six models were trained on the same train/test split (`random_state=42`, stratified split, 80/20):

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison table (required metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9934 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| XGBoost (Ensemble) | 0.9561 | 0.9947 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer in this run, with highest Accuracy/F1/MCC and very strong AUC. |
| Decision Tree | Fast and interpretable but comparatively lower scores, indicating likely underfitting with current depth settings. |
| kNN | Very high recall and strong overall metrics, but may be sensitive to feature scaling and dataset size. |
| Naive Bayes | Good baseline with strong AUC despite simpler independence assumptions. |
| Random Forest (Ensemble) | Stable ensemble performance and good AUC, but slightly below Logistic Regression in this dataset split. |
| XGBoost (Ensemble) | Strong ensemble model with high AUC/recall and competitive overall performance. |

## Project structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- full_dataset.csv
|   |-- test_data.csv
|-- model/
|   |-- train_models.py
|   |-- dataset_metadata.json
|   |-- metrics.csv
|   |-- evaluation_reports.json
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest_ensemble.joblib
|   |-- xgboost_ensemble.joblib
```
