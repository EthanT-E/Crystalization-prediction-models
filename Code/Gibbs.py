import pandas as pd
import sys


def main(Inputs_Data_Set_Path, Predicted_Path, Cas_File_Path=None):
    df_smiles = pd.read_csv(Inputs_Data_Set_Path)
    df = pd.read_csv(Predicted_Path)
    df["min dHm"] = df['Predicted dHm / kJ mol-1'] - 9.9323/2
    df["max dHm"] = df['Predicted dHm / kJ mol-1'] + 9.9323/2
    df["dGc / kJ mol-1"] = -0.225*df['Predicted dHm / kJ mol-1']
    print("Gibbs calculated")
    df["dGc error ±"] = 0.225*(df["max dHm"] - df["min dHm"])/2
    df = df.drop(columns=["min dHm", "max dHm"])
    if (Cas_File_Path is not None):
        cas_df = pd.read_csv(Cas_File_Path)
        df["CAS"] = cas_df["CAS"]
        # df["Price 100mg / £"] = cas_df["Price(100 mg)/£"]
    df["SMILES"] = df_smiles["SMILES"]
    df = df.sort_values(by=["dGc / kJ mol-1"])
    df = df.round(5)
    df.to_csv("output/results.csv", index=False)
    print("Output written\nFinished")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Command prompt parameters wrong")
        print(f"Usage: {
              sys.argv[0]} <INPUT_DATA_PATH>, <PREDICTED_DHM_PATH> <CAS_FILE_PATH/OPTIONAL>")
        exit(1)
