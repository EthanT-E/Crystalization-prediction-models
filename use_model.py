from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import Chem
import xgboost as xgb
import pandas as pd
import re
import sys


def main(Path_to_inputs, Output_File_Name='dHm_predicted.csv'):
    # Get the used descriptors
    good_desc = pd.read_csv('descriptor/dhm_desc.csv')
    good_desc = good_desc.columns.to_list()

    # RDKit descriptor calculator made
    calculator = MolecularDescriptorCalculator(
        good_desc)  # only calcs the given descs
    # Get the input set
    # supplier = Chem.SmilesMolSupplier(
    #     'data/input_set.csv', delimiter=",", smilesColumn=0)
    supplier = Chem.SmilesMolSupplier(
        Path_to_inputs, delimiter=",", smilesColumn=0)

    # Getting molecules from input file
    input_dic = []
    for mol in supplier:
        input_dic.append(calculator.CalcDescriptors(mol))

    # Calculate descriptors of inputs
    calced_desc_df = pd.DataFrame(input_dic)
    # calced_desc_df.to_csv("data/input_set_desc_unscaled.csv")

    # load model
    model = xgb.XGBRegressor()
    model.load_model('models/dHm_model_tsne.json')
    print('model loaded')

    # Set model column names
    calced_desc_df.columns = model.get_booster().feature_names
    predicted = model.predict(calced_desc_df)
    print('prediction made')

    # output_df = pd.read_csv("data/input_set.csv")
    output_df = pd.read_csv(Path_to_inputs)
    output_df = output_df["NAME"]

    predict_df = pd.DataFrame([output_df, predicted])
    predict_df = predict_df.T
    predict_df.columns = ["Name", "Predicted dHm / kJ mol-1"]
    output_path = 'output/'+Output_File_Name
    predict_df.to_csv(output_path, index=False)
    print('Output written\nFin')


if __name__ == '__main__':
    print(f"Usage: {
          sys.argv[0]} <PATH_TO_INPUT_FILE> <NAME_OF_OUTPUT_FILE>\nBy Defualt output file is called dHm results")
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        if (sys.argv[2].find('.csv') == -1 or sys.argv[2] == -1):
            output_file_name = sys.argv[2] + '.csv'
        else:
            output_file_name = sys.argv[2]
        main(sys.argv[1], output_file_name)
    else:
        print("Please input input file path")
