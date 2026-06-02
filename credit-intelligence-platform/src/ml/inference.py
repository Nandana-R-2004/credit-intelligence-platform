import pandas as pd
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

model = joblib.load(os.path.join(BASE_DIR, "models", "credit_risk_model.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed_train.csv"))

# Take a real sample row
sample = df.drop(columns=["TARGET"]).iloc[0:1]

prob = model.predict_proba(sample)[0][1]
pred = model.predict(sample)[0]

print("\n=== INFERENCE RESULT ===")
print("Default Probability:", prob)
print("Prediction:", pred)
print("Risk:", "HIGH" if prob > 0.5 else "LOW")