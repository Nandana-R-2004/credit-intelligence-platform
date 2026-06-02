import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def shap_explain():

    # -----------------------------
    # FIX BASE PATH (IMPORTANT)
    # -----------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    model_path = os.path.join(BASE_DIR, "models", "credit_risk_model.pkl")
    data_path = os.path.join(BASE_DIR, "data", "processed_train.csv")

    print("Loading model and data...")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.drop(columns=["TARGET"])

    print("Creating SHAP explainer...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # -----------------------------
    # GLOBAL EXPLANATION
    # -----------------------------
    shap.summary_plot(shap_values, X)

    # -----------------------------
    # SINGLE PREDICTION
    # -----------------------------
    print("Explaining first prediction...")

    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X.iloc[0],
        matplotlib=True
    )

    plt.show()


if __name__ == "__main__":
    shap_explain()