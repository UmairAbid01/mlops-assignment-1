import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

dataset = load_iris()
features, labels = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

def log_with_mlflow(model_name, model, params, y_true, y_pred, model_path):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, model_name)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=dataset.target_names,
                    yticklabels=dataset.target_names)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        cm_path = f"../results/{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

    return [acc, prec, rec, f1]


log_reg = LogisticRegression(solver="liblinear", C=0.8, max_iter=300, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

log_scores = log_with_mlflow(
    "Logistic Regression",
    log_reg,
    {"solver": "liblinear", "C": 0.8, "max_iter": 300},
    y_test,
    log_reg_pred,
    "../models/log_reg_model.pkl"
)

rf = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_split=3, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_scores = log_with_mlflow(
    "Random Forest",
    rf,
    {"n_estimators": 120, "max_depth": 6, "min_samples_split": 3},
    y_test,
    rf_pred,
    "../models/random_forest_model.pkl"
)

svm = SVC(kernel="poly", degree=3, C=2.0, probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

svm_scores = log_with_mlflow(
    "SVM (Poly Kernel)",
    svm,
    {"kernel": "poly", "degree": 3, "C": 2.0},
    y_test,
    svm_pred,
    "../models/svm_poly_model.pkl"
)

all_results = [
    ["Logistic Regression", *log_scores],
    ["Random Forest", *rf_scores],
    ["SVM (Poly Kernel)", *svm_scores]
]

results_df = pd.DataFrame(all_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
results_df.to_csv("../results/model_results.csv", index=False)

print("\n=== Model Performance Comparison ===\n")
print(results_df)
print("\nResults saved to results/model_results.csv")
print("\nYou can now open MLflow UI with: mlflow ui")
