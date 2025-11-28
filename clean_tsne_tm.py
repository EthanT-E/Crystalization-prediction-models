import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("data/raw_tm_desc.csv")

# Removing low variance
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
scaled_df = pd.DataFrame(data_scaled)
scaled_df.columns = df.columns
std = scaled_df.std(axis=0)
std_low = std[std < 0.01]
drop_cols = std_low.index.tolist()
drop_cols.extend(
    ['MaxAbsEStateIndex', 'MolWt', 'HeavyAtomMolWt',
     'NumValenceElectrons', 'NumAmideBonds', 'fr_phenol_noOrthoHbond',
     'fr_Nhpyrrole', 'VSA_EState10', 'VSA_EState6', 'fr_benzene',
     'NHOHCount', 'HeavyAtomCount', 'SMR_VSA2',
     'Kappa3', 'fr_Ar_N', 'NOCount',
     'SlogP_VSA6', 'fr_C_O_noCOO', 'fr_Al_OH_noTert',
     'ExactMolWt', 'Chi0', 'LabuteASA', 'Kappa1', 'Kappa2',
     'MaxPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
     'FpDensityMorgan1', 'FpDensityMorgan2',
     'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
     'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
     'NumUnspecifiedAtomStereoCenters',
     'Chi3v', 'Chi4n', 'Chi4v', 'fr_COO2'])
df = df.drop(columns=drop_cols)

# removing corr above 0.9
corr = df.corr(method='spearman')
corr = corr.abs()
corr_df = pd.DataFrame(corr)
corr_df.columns = df.columns

corr_df = corr_df >= 0.9
value_counts = []
for element in corr_df.columns:
    value_counts.append(corr_df[element].value_counts()[True])

high_corr_bool = pd.DataFrame([value_counts])
high_corr_bool.columns = corr_df.columns
high_corr_bool = high_corr_bool.T
high_corr_bool = high_corr_bool[high_corr_bool > 1]
high_corr_bool = high_corr_bool.dropna()
print(high_corr_bool)

df.to_csv('tsne_cleaned_desc_Tm.csv')
