import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ---------------- LOAD DATA ----------------
print("Loading dataset...")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data_path = os.path.join(BASE_DIR, "data", "processed_train.csv")

df = pd.read_csv(data_path)

# ---------------- FEATURE ENGINEERING ----------------
print("Creating features...")

df["AGE"] = abs(df["DAYS_BIRTH"]) / 365

df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(
    lambda x: abs(x) / 365 if x < 0 else 0
)

# ---------------- SELECT FEATURES ----------------
features = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AGE",
    "YEARS_EMPLOYED"
]

target = "TARGET"

X = df[features]
y = df[target]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- LOAD MODEL ----------------
print("Loading trained model...")

model_path = os.path.join(BASE_DIR, "models", "credit_risk_model.pkl")
model = joblib.load(model_path)

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc_auc)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# ---------------- SAVE METRICS ----------------
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": roc_auc
}

metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.pkl")
joblib.dump(metrics, metrics_path)

print("\nMetrics saved successfully!")
print(metrics_path)