import pandas as pd

df = pd.read_csv('output/dHm.csv')
df["dGc / kJ mol-1"] = -0.225*df['Predicted dHm / kJ mol-1']  # app
df.to_csv("output/results.csv", index=False)
