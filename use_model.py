from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import Chem
import xgboost as xgb
import pandas as pd

# Get the used descriptors
good_desc = pd.read_csv('descriptor/dhm_descriptors.csv')
good_desc = good_desc.columns.to_list()

# RDKit descriptor calculator made
calculator = MolecularDescriptorCalculator(
    good_desc)  # only calcs the given descs
# Get the input set
supplier = Chem.SmilesMolSupplier(
    'data/input_set.csv', delimiter=",", smilesColumn=0)

# Getting molecules from input file
input_dic = []
for mol in supplier:
    input_dic.append(calculator.CalcDescriptors(mol))

# Calculate descriptors of inputs
calced_desc_df = pd.DataFrame(input_dic)

# load model
model = xgb.XGBRegressor()
model.load_model('models/dHm_model.json')

# Set model column names
calced_desc_df.columns = model.get_booster().feature_names
predicted = model.predict(calced_desc_df)

output_df = pd.read_csv("data/input_set.csv")
output_df = output_df["NAME"]

predict_df = pd.DataFrame([output_df, predicted])
predict_df = predict_df.T
predict_df.columns = ["Name", "Predicted dHm / kJ mol-1"]
predict_df.to_csv('output/dHm.csv', index=False)
