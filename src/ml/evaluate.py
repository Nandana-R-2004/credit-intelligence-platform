import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay
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
print("Making predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)

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

# ---------------- ROC CURVE ----------------
print("\nGenerating ROC Curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2,
         label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Risk Prediction")
plt.legend(loc="lower right")
plt.grid(True)

roc_path = os.path.join(BASE_DIR, "models", "roc_curve.png")
plt.savefig(roc_path, bbox_inches="tight")
plt.close()

print(f"ROC Curve saved at: {roc_path}")

# ---------------- CONFUSION MATRIX PLOT ----------------
print("Generating Confusion Matrix Plot...")

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

cm_path = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()

print(f"Confusion Matrix saved at: {cm_path}")

print("\nEvaluation completed successfully!")