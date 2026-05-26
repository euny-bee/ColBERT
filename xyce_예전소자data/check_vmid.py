import pandas as pd
df = pd.read_csv("tmp_v4v5_m1d0.cir.csv")
df.columns = ["v4","vsn","vmid","iv5"]
df["x"] = df["v4"] - 1.0
print(df[["v4","x","vsn","vmid","iv5"]].iloc[::5].to_string())
