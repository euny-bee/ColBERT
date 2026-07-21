import pandas as pd
import numpy as np

df = pd.read_csv('phase12_dual_die4_vcomp_scaled.cir.csv')
df = df.loc[:, ~df.columns.duplicated()]

vth_list = [0.03, 0.53, 1.03, 2.03, 3.03]
V1 = 1.0
reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]

print(f"{'Vth':>6} {'예측(-V1+Vth)':>15} {'실제 SN_final':>15} {'실제 SNB_final':>16} {'settle_us(SN)':>14}")
for vth, (i0, i1) in zip(vth_list, zip(reset_idx, reset_idx[1:])):
    seg = df.iloc[i0:i1]
    seg = seg[seg['TIME'] > 0.2e-6]
    sn = seg['V(SN)'].values
    snb = seg['V(SNB)'].values
    t = seg['TIME'].values
    sn_final = sn[-1]; snb_final = snb[-1]
    pred = -V1 + vth
    tol = max(0.01*abs(sn_final-sn[0]), 1e-4)
    idx = np.where(np.abs(sn - sn_final) <= tol)[0]
    t_settle = t[idx[0]]*1e6 if len(idx) else float('nan')
    print(f"{vth:6.2f} {pred:15.4f} {sn_final:15.4f} {snb_final:16.4f} {t_settle:14.3f}")
