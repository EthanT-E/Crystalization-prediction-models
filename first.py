from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import pandas as pd

df = pd.read_csv("data/Dataset_dHm.csv")
smiles_list = df["SMILES"].tolist()
mol_list = []
desc_list = []

for smiles_str in smiles_list:
    mol = Chem.MolFromSmiles(smiles_str)
    desc_list.append(Descriptors.CalcMolDescriptors(mol))

df = pd.DataFrame(desc_list)
df.to_csv("data/rdkit_unscaled.csv")
# for smiles_str in range(0, 10):
#     print(smiles_list[smiles_str])
#     mol = Chem.MolFromSmiles(smiles_list[smiles_str])
#     desc_list.append(Descriptors.CalcMolDescriptors(mol))
# for i in range(0, len(smiles_list)):
#     mol_list.append(Chem.MolFromSmiles(smiles_list[i]))
#
