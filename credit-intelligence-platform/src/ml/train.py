import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ---------------- LOAD DATA ----------------
print("Loading dataset...")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data_path = os.path.join(BASE_DIR, "data", "processed_train.csv")

df = pd.read_csv(data_path)

# ---------------- FEATURE ENGINEERING ----------------
print("Creating user-friendly features...")

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

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- MODEL ----------------
print("Training XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "credit_risk_model.pkl")
joblib.dump(model, model_path)

print("\nModel saved at:", model_path)