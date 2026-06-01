import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


def preprocess_data():

    # ==============================
    # FIXED: absolute project path
    # ==============================
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    input_path = os.path.join(BASE_DIR, "data", "application_train.csv")
    output_path = os.path.join(BASE_DIR, "data", "processed_train.csv")

    print("Loading Dataset...")

    # ==============================
    # CHECK FILE EXISTS (IMPORTANT)
    # ==============================
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at: {input_path}")

    df = pd.read_csv(input_path)

    print("Original Shape:", df.shape)

    # ==============================
    # COLUMNS
    # ==============================
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    print("Handling Missing Values...")

    # ==============================
    # MISSING VALUES
    # ==============================
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Encoding Categorical Features...")

    # ==============================
    # ENCODING
    # ==============================
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print("Final Missing Values:", df.isnull().sum().sum())
    print("Processed Shape:", df.shape)

    # ==============================
    # SAVE OUTPUT
    # ==============================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Saved:", output_path)

    return df, encoders


if __name__ == "__main__":
    preprocess_data()