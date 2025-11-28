import pandas

df = pandas.read_csv("../data/raw_tm_desc.csv")
df = df.drop(columns=['SMR_VSA8', 'SlogP_VSA9',
                      'fr_diazo', 'fr_phos_ester', 'fr_prisulfonamd',
                      'Chi1v', 'Chi0v', 'Chi1', 'Ipc', 'Kappa1',
                      'MolMR', 'NumValenceElectrons', 'Chi0', 'Chi0n',
                      'LabuteASA', 'HeavyAtomCount', 'Chi1n', 'Chi2n',
                      'MolWt', 'ExactMolWt', 'HeavyAtomMolWt', 'Chi3n',
                      'Chi2v', 'Chi3v', 'Chi4n', 'Chi4v', 'Kappa2',
                      'Phi', 'FpDensityMorgan2', 'Kappa3',
                      'MaxAbsEStateIndex', 'fr_amide',
                      'fr_nitrile', 'fr_phenol_noOrthoHbond', 'fr_benzene',
                      'NumUnspecifiedAtomStereoCenters', 'NumHDonors',
                      'NumAromaticHeterocycles',
                      'fr_Nhpyrrole', 'VSA_EState6', 'SlogP_VSA6', 'TPSA',
                      'fr_Al_OH_noTert', 'SlogP_VSA12', 'fr_COO',
                      'fr_C_O_noCOO', 'MinAbsPartialCharge', 'MinPartialCharge'])
df.to_csv("clean_PCA_desc.csv", index=False)
