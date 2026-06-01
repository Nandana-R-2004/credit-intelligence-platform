import pandas as pd

df = pd.read_csv("data/application_train.csv")

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("\nFirst 5 Rows")
print(df.head())

print("\nTarget Distribution")
print(df["TARGET"].value_counts())

print("\nMissing Values")
print(df.isnull().sum().sort_values(ascending=False).head(20))