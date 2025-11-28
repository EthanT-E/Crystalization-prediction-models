from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd

supplier = Chem.SmilesMolSupplier(
    "data/Dataset_Tm.csv", delimiter=",", smilesColumn=0)

mol_num = len(supplier)
slice = mol_num//100
outer_arr = np.arange(0, 100)
inner_arr = np.arange(0, 2540)
for outer in outer_arr:
    desc = []
    index = (outer*100) + inner_arr
    for inner in inner_arr:
        i = int(index[inner])
        desc.append(Descriptors.CalcMolDescriptors(supplier[i]))
    df = pd.DataFrame(desc)
    if outer != 0:
        df.to_csv("data/raw_tm_desc.csv", index=False, header=False, mode='a')
    else:
        df.to_csv("data/raw_tm_desc.csv", index=False)
    print(f"{outer+1}%")
