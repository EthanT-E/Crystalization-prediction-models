import pandas as pd

# df = pd.read_csv("data/tm_scaled.csv")
df = pd.read_csv('data/raw_tm_desc.csv')
target_df = pd.read_csv("data/Dataset_Tm.csv")
print('read_data')

df["target"] = target_df["Tm"]
print('targeted appended')
corr = df.corr(method='spearman')['target']
print('corr calced')
corr = corr.drop(columns=["target"])
# corr = corr[corr > 0.5]
print(corr.describe())
