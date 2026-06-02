import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

def feature_importance():

    model = joblib.load("models/credit_risk_model.pkl")

    df = pd.read_csv("data/processed_train.csv")

    X = df.drop(columns=["TARGET"])

    importances = model.feature_importances_
    feature_names = X.columns

    # Create dataframe
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(15)

    print("\nTop 15 Important Features:\n")
    print(feat_df)

    # Plot
    plt.figure(figsize=(10,6))
    plt.barh(feat_df["Feature"], feat_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance (XGBoost)")
    plt.show()


if __name__ == "__main__":
    feature_importance()