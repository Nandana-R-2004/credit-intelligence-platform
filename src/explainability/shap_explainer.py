import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

model_path = os.path.join(BASE_DIR, "models", "credit_risk_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "processed_train.csv")

print("Loading model...")
model = joblib.load(model_path)

print("Loading data...")
df = pd.read_csv(data_path)

# Create same features used during training
df["AGE"] = abs(df["DAYS_BIRTH"]) / 365

df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(
    lambda x: abs(x) / 365 if x < 0 else 0
)

features = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AGE",
    "YEARS_EMPLOYED"
]

# Take only 100 rows
X = df[features].sample(100, random_state=42)

print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X)

print("Generating summary plot...")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)

save_path = os.path.join(BASE_DIR, "models", "shap_summary.png")
plt.savefig(save_path, bbox_inches="tight")
plt.close()

print(f"SHAP figure saved at: {save_path}")