import pandas as pd
import numpy as np
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import Chem
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import xgboost as xgb
import sys


def Get_descriptor_df(Path_to_Smiles_list, Path_to_desc_list):

    good_desc = pd.read_csv('descriptor/dhm_desc.csv')
    good_desc = good_desc.columns.to_list()
    calculator = MolecularDescriptorCalculator(good_desc)
    supplier = Chem.SmilesMolSupplier(
        Path_to_Smiles_list, delimiter=",", smilesColumn=0)

    # Getting molecules from input file
    calced_desc_dic = []
    for mol in supplier:
        calced_desc_dic.append(calculator.CalcDescriptors(mol))
    # Calculate descriptors of inputs
    calced_desc_df = pd.DataFrame(calced_desc_dic)
    print("Descriptors calculated")
    return calced_desc_df


def main(Path_to_Smiles_list='data/Smiles_dHm.csv', Path_to_desc_list='descriptor/dhm_desc.csv', plotflag=0):
    target_df = pd.read_csv('data/Dataset_dHm.csv')
    # desc_df = pd.read_csv('data/tsne_dhm_desc.
    # desc_df = Get_descriptor_df(
    #    'data/Smiles_dHm.csv', 'descriptor/dhm_desc.csv')
    desc_df = Get_descriptor_df(
        Path_to_Smiles_list, Path_to_desc_list)
    x_train, x_test, y_train, y_test = train_test_split(
        desc_df, target_df["dHm"], test_size=0.3, train_size=0.7)

    model = xgb.XGBRegressor(max_depth=9,
                             learning_rate=0.03751292045151882,
                             n_estimators=392,
                             subsample=0.6288953935334737,
                             colsample_bytree=0.6103729030584606,
                             min_child_weight=8,
                             gamma=4.531330426603764,
                             alpha=0.9572092266603516,
                             tree_method='hist'
                             )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(f"SMRE: {root_mean_squared_error(y_test, pred)}, MAE {
          mean_absolute_error(y_test, pred)}")
    if plotflag == 1:
        residuals = y_test - pred
        plt.scatter(x=np.arange(0, len(y_test)), y=residuals)
        plt.show()
    download = input("Download Model? [yes/exit] ")
    if download == "yes":
        model.save_model('models/dHm_model.json')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        if (int(sys.argv[1]) != 1):
            print(f"Usage: {
                  sys.argv[0]}, <PLOTFLAG>\nPlot flag is by default 0 if you want to print residuals set to 1")
        else:
            main(1)
    else:
        print(f"Usage: {
            sys.argv[0]}, <PLOTFLAG>\nPlot flag is by default 0 if you want to print residuals set to 1")
