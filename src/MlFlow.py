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
mlflow.set_tracking_uri("file:../mlruns")   # ensures logging in project root
mlflow.set_experiment("Iris Classification")  # all runs grouped here

# ===============================
# Load dataset
# ===============================
dataset = load_iris()
features, labels = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

# ===============================
# Helper function for MLflow logging
# ===============================
def log_with_mlflow(model_name, model, params, y_true, y_pred, model_path):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Save locally
        joblib.dump(model, model_path)

        # ✅ Log model in MLflow (artifact_path must be simple)
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name.replace(" ", "_")
        )

        # Confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=dataset.target_names,
                    yticklabels=dataset.target_names)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        os.makedirs("../results", exist_ok=True)
        cm_path = f"../results/{model_name.replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

    return [acc, prec, rec, f1]


# ===============================
# Train and log models
# ===============================

# Logistic Regression
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

# Random Forest
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

# SVM
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

# ===============================
# Save results comparison
# ===============================
all_results = [
    ["Logistic Regression", *log_scores],
    ["Random Forest", *rf_scores],
    ["SVM (Poly Kernel)", *svm_scores]
]

results_df = pd.DataFrame(all_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
os.makedirs("../results", exist_ok=True)
results_df.to_csv("../results/model_results.csv", index=False)

print("\n=== Model Performance Comparison ===\n")
print(results_df)
print("\nResults saved to ../results/model_results.csv")
print("\nYou can now open MLflow UI with: mlflow ui")
