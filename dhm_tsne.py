import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost import XGBRegressor
import numpy as np


df = pd.read_csv("data/dhm_raw_desc.csv")

scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df)
scaled_df.columns = df.columns
std = scaled_df.std(axis=0)
low_std = std[std < 0.02]

df = df.drop(columns=low_std.index.tolist())

# Dropped columns with > 0.9 pairwise corr
drop_col = [
    'MolWt', 'HeavyAtomMolWt', 'MaxAbsEStateIndex', 'NumValenceElectrons',
    'ExactMolWt', 'Chi0', 'Chi0n', 'Chi0v',
    'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
    'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n',
    'Chi4v', 'AvgIpc', 'Kappa1', 'fr_benzene',
    'SMR_VSA7', 'MaxPartialCharge', 'MinPartialCharge', 'fr_phos_ester',
    'fr_COO2', 'Kappa2', 'fr_Al_OH_noTert', 'FpDensityMorgan2',
    'LabuteASA', 'fr_Ar_OH', 'fr_phenol_noOrthoHbond', 'NumAromaticRings',
    'NOCount', 'SlogP_VSA12', 'VSA_EState6', 'Phi',
    'fr_Ar_NH', 'HeavyAtomCount', 'NumAmideBonds', 'SMR_VSA2',
    'NHOHCount'
]
df = df.drop(columns=drop_col)
print(df.columns.tolist())

# corr = df.corr(method='spearman')
# corr = corr.abs()
#
# corr_df = pd.DataFrame(corr)
# corr_df.columns = df.columns
# corr_df = corr_df >= 0.9
# value_counts = []
# for element in corr_df.columns:
#     value_counts.append(corr_df[element].value_counts()[True])
#
# high_corr_bool = pd.DataFrame([value_counts])
# high_corr_bool.columns = corr_df.columns
# high_corr_bool = high_corr_bool.T
# high_corr_bool = high_corr_bool[high_corr_bool > 1]
# high_corr_bool = high_corr_bool.dropna()
# print(high_corr_bool)
df.to_csv('data/tsne_dhm_desc.csv', index=False)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


pca = PCA(n_components=43)
pca_desc = pca.fit_transform(df_scaled)

tnse = TSNE(perplexity=9)
X_tsne = tnse.fit_transform(pca_desc)


tnse_df = pd.DataFrame(X_tsne)

target_df = pd.read_csv('data/Dataset_dHm.csv')
target = target_df['dHm']
X_train, X_test, Y_train, Y_test = train_test_split(X_tsne, target)
model = XGBRegressor()
model.fit(X=X_train, y=Y_train)
pred = model.predict(X_test)

print(f"MAE {mean_absolute_error(Y_test, pred)} RMSE {
      root_mean_squared_error(Y_test, pred)}")

df_scaled = pd.DataFrame(df_scaled)
# df_scaled.to_csv('data/tsne_dhm_desc.csv')
