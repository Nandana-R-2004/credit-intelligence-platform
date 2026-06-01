import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/application_train.csv")

sns.countplot(x="TARGET", data=df)
plt.title("Loan Default Distribution")
plt.savefig("documents/target_distribution.png")
plt.show()