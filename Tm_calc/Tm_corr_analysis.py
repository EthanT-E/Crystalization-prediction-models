import pandas as pd
from seaborn import heatmap
import matplotlib.pyplot as plt

corr = pd.read_csv('./corr_Tm.csv', index_col=0)
nan = corr.isna().all()
nan = nan[nan == True]
nan_drop_list = nan.index.tolist()
'''
5 cols had Nans so they will be removed
    SMR_VSA8
    SlogP_VSA9
    fr_diazo
    fr_phos_ester
    fr_prisulfonamd
'''

list_of_removed = []
# for i in range(0, 25):
#     high_corr = corr > 0.9
#
#     col_names = high_corr.columns.tolist()
#     value_counts = []
#
#     for element in col_names:
#         value_counts.append(high_corr[element].value_counts()[True])
#
#     high_corr_bool = pd.DataFrame([value_counts])
#     high_corr_bool.columns = col_names
#     high_corr_bool = high_corr_bool.T
#     high_corr_bool = high_corr_bool[high_corr_bool > 1]
#     high_corr_bool = high_corr_bool.dropna()
#     max = high_corr_bool.idxmax()
#     list_of_removed.append(max.values[0])
#     corr = corr.drop(columns=max.values[0])
#
# print(list_of_removed)
# print(high_corr_bool.describe())
# print(high_corr_bool.idxmax())
heatmap(corr, annot=True, xticklabels=True, yticklabels=True)
plt.show()

high_corr = corr > 0.9
col_names = high_corr.columns.tolist()
value_counts = []
for element in col_names:
    value_counts.append(high_corr[element].value_counts()[True])
high_corr_bool = pd.DataFrame([value_counts])
high_corr_bool.columns = col_names
high_corr_bool = high_corr_bool.T
high_corr_bool = high_corr_bool[high_corr_bool > 1]
high_corr_bool = high_corr_bool.dropna()
print(high_corr_bool)
'''
Dropped:              'Chi0', 'Chi0n',
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
                      'fr_C_O_noCOO', 'MinAbsPartialCharge', 'MinPartialCharge'
for PCA
'''
