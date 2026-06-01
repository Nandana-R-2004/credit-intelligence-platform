import pandas as pd

def run_query(df, sql):

    sql = sql.lower()

    if "target = 1" in sql:
        result = df[df["TARGET"] == 1]
        print("High Risk Customers:", len(result))
        return result

    elif "target = 0" in sql:
        result = df[df["TARGET"] == 0]
        print("Low Risk Customers:", len(result))
        return result

    elif "income" in sql:
        result = df[df["AMT_INCOME_TOTAL"] > 200000]
        print("High Income Customers:", len(result))
        return result

    else:
        print("Default result returned")
        return df.head(5)


if __name__ == "__main__":
    df = pd.read_csv("data/processed_train.csv")
    print(run_query(df, "SELECT * FROM loans WHERE TARGET = 1"))