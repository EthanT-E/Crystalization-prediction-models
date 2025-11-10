from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

df = pd.read_csv("data/Dataset_dHm.csv")
df = pd.DataFrame(df["SMILES"])
df["names"] = "unkown"
df.to_csv("data/Smiles_dHm.csv", index=False)

supplier = Chem.SmilesMolSupplier(
    'data/Smiles_dHm.csv', delimiter=",", smilesColumn=0)
desc = []
for mol in supplier:
    desc.append(Descriptors.CalcMolDescriptors(mol))

desc_df = pd.DataFrame(desc)
desc_df.to_csv("data/test.csv")
