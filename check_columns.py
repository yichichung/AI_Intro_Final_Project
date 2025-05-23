import pandas as pd

df = pd.read_csv("data/participants.csv", nrows=3)
print("📋 欄位名稱如下：")
print(df.columns.tolist())
