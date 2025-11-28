from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def main(n_components: int):
    df = pd.read_csv("./clean_PCA_desc.csv")

    bin = df.isna().all()

    bin = bin[bin == True]

    scaler = StandardScaler()

    df_scaled = scaler.fit_transform(df)
    bin = type(df_scaled)
    df_df = pd.DataFrame(df_scaled)
    df_bin = df_df.isnull().all()

    pca_var = PCA(n_components=n_components)
    pca_var.fit(df_scaled)
    var = pca_var.explained_variance_ratio_
    df_var = pd.DataFrame(
        {'PC': np.arange(1, n_components+1), 'Var': var, 'Cum var': var.cumsum()})

    plt.bar(df_var['PC'], df_var['Cum var'])
    plt.bar(df_var['PC'], df_var['Var'])
    plt.xlabel('Number of PC')
    plt.ylabel('Cumulative variance')
    plt.title('PC vs variance')
    plt.plot(np.arange(1, n_components+1),
             np.full(n_components, 0.7), color='red', linestyle='--')
    plt.show()

    plt.plot(df_var['PC'], var, 'o-')
    plt.show()
    print(df_var.tail(10))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'USAGE: {sys.argv[0]} <n_components>')
    else:
        main(int(sys.argv[1]))
