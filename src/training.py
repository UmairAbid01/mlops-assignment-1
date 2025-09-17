import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# ================================
# Load and prepare dataset
# ================================
dataset = load_iris()
features, labels = dataset.data, dataset.target

# Split into training and test sets (30% test for variation)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

# Ensure output folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ================================
# Logistic Regression Model
# ================================
log_model = LogisticRegression(
    solver="liblinear",      # changed solver
    C=0.8,                   # changed regularization strength
    max_iter=300,            # increased iterations
    random_state=42
)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

log_scores = [
    accuracy_score(y_test, log_pred),
    precision_score(y_test, log_pred, average="macro"),
    recall_score(y_test, log_pred, average="macro"),
    f1_score(y_test, log_pred, average="macro")
]

joblib.dump(log_model, "../models/log_reg_model.pkl")

# ================================
# Random Forest Model
# ================================
rf_model = RandomForestClassifier(
    n_estimators=120,         # more trees
    max_depth=6,              # restrict depth
    min_samples_split=3,      # stricter split rule
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_scores = [
    accuracy_score(y_test, rf_pred),
    precision_score(y_test, rf_pred, average="macro"),
    recall_score(y_test, rf_pred, average="macro"),
    f1_score(y_test, rf_pred, average="macro")
]

joblib.dump(rf_model, "../models/random_forest_model.pkl")

# ================================
# Support Vector Machine Model
# ================================
svm_model = SVC(
    kernel="poly",           # polynomial kernel instead of rbf
    degree=3,                # polynomial degree
    C=2.0,                   # margin hardness
    probability=True,
    random_state=42
)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

svm_scores = [
    accuracy_score(y_test, svm_pred),
    precision_score(y_test, svm_pred, average="macro"),
    recall_score(y_test, svm_pred, average="macro"),
    f1_score(y_test, svm_pred, average="macro")
]

joblib.dump(svm_model, "../models/svm_poly_model.pkl")

# ================================
# Compile and Save Results
# ================================
all_results = [
    ["Logistic Regression", *log_scores],
    ["Random Forest", *rf_scores],
    ["SVM (Poly Kernel)", *svm_scores]
]

results_df = pd.DataFrame(
    all_results, 
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\n=== Model Performance Comparison ===\n")
print(results_df)

results_df.to_csv("../results/model_results.csv", index=False)
print("\nResults saved to results/model_results.csv")
