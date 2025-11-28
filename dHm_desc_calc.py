from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

supplier = Chem.SmilesMolSupplier('data/Smiles_dHm.csv', delimiter=',')

desc = []
for mol in supplier:
    desc.append(Descriptors.CalcMolDescriptors(mol))

desc_df = pd.DataFrame(desc)
desc_df.to_csv('data/dhm_raw_desc.csv', index=False)
