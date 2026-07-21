import pandas as pd
import numpy as np

df = pd.read_csv('phase12_dual_die4.cir.csv')
df = df.loc[:, ~df.columns.duplicated()]

vth_list = [0.03, 0.53, 1.03, 2.03, 3.03]
reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]

for vth, (i0, i1) in zip(vth_list, zip(reset_idx, reset_idx[1:])):
    seg = df.iloc[i0:i1]
    seg = seg[seg['TIME'] > 0.2e-6]
    t = seg['TIME'].values
    sn = seg['V(SN)'].values
    snb = seg['V(SNB)'].values
    sn_final = sn[-1]
    snb_final = snb[-1]
    # find first time within 1% of final value (and staying there)
    tol_sn = 0.01 * abs(sn_final - sn[0]) if abs(sn_final-sn[0])>1e-9 else 1e-6
    tol_snb = 0.01 * abs(snb_final - snb[0]) if abs(snb_final-snb[0])>1e-9 else 1e-6
    idx_sn = np.where(np.abs(sn - sn_final) <= max(tol_sn,1e-4))[0]
    idx_snb = np.where(np.abs(snb - snb_final) <= max(tol_snb,1e-4))[0]
    t_settle_sn = t[idx_sn[0]] if len(idx_sn) else None
    t_settle_snb = t[idx_snb[0]] if len(idx_snb) else None
    print(f"Vth={vth:.2f}V  SN: {sn[0]:.4f}->{sn_final:.4f}  settle@{t_settle_sn*1e6 if t_settle_sn else None:.3f}us | "
          f"SNB: {snb[0]:.4f}->{snb_final:.4f}  settle@{t_settle_snb*1e6 if t_settle_snb else None:.3f}us")
