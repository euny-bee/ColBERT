import pandas as pd
import numpy as np

vth_list = [0.03, 0.53, 1.03, 2.03, 3.03]
V1 = 1.0

def get_finals(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]
    finals = []
    for vth, (i0, i1) in zip(vth_list, zip(reset_idx, reset_idx[1:])):
        seg = df.iloc[i0:i1]
        seg = seg[seg['TIME'] > 0]
        finals.append(seg['V(SN)'].values[-1])
    return finals

f0 = get_finals('phase2_only_sharptail_IC0.cir.csv')
f4 = get_finals('phase2_only_sharptail_IC4.cir.csv')

print(f"{'Vth':>6} {'예측(-V1+Vth)':>15} {'SN_final(0V시작)':>18} {'SN_final(4V시작)':>18}")
for vth, a, b in zip(vth_list, f0, f4):
    pred = -V1+vth
    print(f"{vth:6.2f} {pred:15.4f} {a:18.4f} {b:18.4f}")
