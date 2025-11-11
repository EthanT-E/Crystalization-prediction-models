import pandas as pd


df = pd.read_csv("data/rdkit_unscaled.csv")
target_df = pd.read_csv("data/Dataset_dHm.csv")
df["target"] = target_df["dHm"]

corr = df.corr(method='spearman')["target"]
filt_corr = corr[abs(corr) > 0.5]
filt_corr = filt_corr.drop("target")
filt_df = df[filt_corr.index.to_list()]
filt_df = filt_df.drop(
    columns=["HeavyAtomMolWt", "ExactMolWt", "Kappa1",
             "Kappa2", 'Chi0', 'Chi0n', 'Chi0v', 'Chi1',
             'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
             'Chi3v', 'Chi4v', 'HeavyAtomCount', 'LabuteASA',
             'Ipc', 'MolWt', 'MolMR', 'NumValenceElectrons',
             'NumRotatableBonds', 'Phi'])
filt_df.to_csv("data/filt_desc.csv", index=False)
