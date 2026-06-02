import pandas as pd

def load_data():
    train_df = pd.read_csv("data/application_train.csv")
    test_df = pd.read_csv("data/application_test.csv")

    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data()

    print("Training Shape:", train_df.shape)
    print("Testing Shape:", test_df.shape)