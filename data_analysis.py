import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


df = pd.read_csv("data/rdkit_unscaled.csv")
target_df = pd.read_csv("data/Dataset_dHm.csv")
df["target"] = target_df["dHm"]

corr = df.corr(method='spearman')["target"]
filt_corr = corr[abs(corr) > 0.5]
filt_corr = filt_corr.drop("target")
print(filt_corr)
filt_df = df[filt_corr.index.to_list()]
filt_df = filt_df.drop(
    columns=["HeavyAtomMolWt", "ExactMolWt", "Kappa1",
             "Kappa2", 'Chi0', 'Chi0n', 'Chi0v', 'Chi1',
             'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
             'Chi3v', 'Chi4v', 'HeavyAtomCount', 'LabuteASA',
             'Ipc', 'MolWt', 'MolMR', 'NumValenceElectrons', 'NumRotatableBonds', 'Phi'])

x_train, x_test, y_train, y_test = train_test_split(
    filt_df, target_df["dHm"], test_size=0.3, train_size=0.7)
model = XGBRegressor(max_depth=12)
model.fit(x_train, y_train)
pred = model.predict(x_test)
results_df = pd.DataFrame()
results_df["true"] = y_test
results_df["predict"] = pred
results_df["residuals"] = results_df["true"] - results_df["predict"]

print(f"Mean squared error: {root_mean_squared_error(y_test, pred)}")
print(f"Mean absoute error: {mean_absolute_error(y_test, pred)}")
results_df["residuals"].plot(style=".")
plt.show()
plt.scatter(x=results_df["true"], y=results_df["predict"])
plt.show()
# sns.heatmap(filt_df.corr(method="spearman"), annot=True)
# plt.show()
