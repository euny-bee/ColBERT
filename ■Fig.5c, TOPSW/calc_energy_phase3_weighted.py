"""Phase3 search 에너지 -- Option A/C, 실제 q2 diff(Q-C) 분포로 가중평균."""
import pandas as pd
import numpy as np

SENSE_US = 1.0
src_v = {'I(VVCOMP)': 'V(VCOMPN)', 'I(VVREAD)': 'V(VREADN)',
         'I(VBL_L)': 'V(VBL_L)', 'I(VBL_R)': 'V(VBL_R)',
         'I(VV2P)': 'V(V2P)', 'I(VV2N)': 'V(V2N)'}

def energy_from_sources(seg):
    t = seg['TIME'].values
    total = 0.0
    for isrc, vsrc in src_v.items():
        if vsrc not in seg.columns or isrc not in seg.columns:
            continue
        v = seg[vsrc].values
        i = seg[isrc].values
        total += np.trapz(-v * i, t)
    return total

# ---------- Option A ----------
dfA = pd.read_csv('phase123_dual_die4.cir.csv')
dfA = dfA.loc[:, ~dfA.columns.duplicated()]
reset_idx = dfA.index[dfA['TIME'] == 0].tolist() + [len(dfA)]
t_inj_A = 10000.1e-6
v2_listA = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
diffA_raw, EA_raw = [], []
for v2, (i0, i1) in zip(v2_listA, zip(reset_idx, reset_idx[1:])):
    seg = dfA.iloc[i0:i1]
    seg = seg[(seg['TIME'] > t_inj_A) & (seg['TIME'] <= t_inj_A + SENSE_US*1e-6)]
    e = energy_from_sources(seg)
    diffA_raw.append(abs(1.0 - v2))
    EA_raw.append(e)
diffA_raw = np.array(diffA_raw); EA_raw = np.array(EA_raw)
order = np.argsort(diffA_raw)
diffA, EA = diffA_raw[order], EA_raw[order]
print("Option A anchor points |diff| -> E(fJ):", list(zip(diffA, EA*1e15)))

# ---------- Option C ----------
dfC = pd.read_csv('phase13_optionC_die4.cir.csv')
dfC = dfC.loc[:, ~dfC.columns.duplicated()]
reset_idx = dfC.index[dfC['TIME'] == 0].tolist() + [len(dfC)]
t_inj_C = 0.2001e-6
v2_listC = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
diffC_raw, EC_raw = [], []
for v2, (i0, i1) in zip(v2_listC, zip(reset_idx, reset_idx[1:])):
    seg = dfC.iloc[i0:i1]
    seg = seg[(seg['TIME'] > t_inj_C) & (seg['TIME'] <= t_inj_C + SENSE_US*1e-6)]
    e = energy_from_sources(seg)
    diffC_raw.append(abs(v2))
    EC_raw.append(e)
diffC_raw = np.array(diffC_raw); EC_raw = np.array(EC_raw)
order = np.argsort(diffC_raw)
diffC_sorted, EC_sorted = diffC_raw[order], EC_raw[order]
diffC, uniq_idx, inv = np.unique(diffC_sorted, return_index=True, return_inverse=True)
EC = np.array([EC_sorted[inv == k].mean() for k in range(len(diffC))])
print("Option C anchor points |diff| -> E(fJ):", list(zip(diffC, EC*1e15)))

# ---------- 실제 q2 diff 분포로 가중평균 ----------
diff_real = np.load('/mnt/c/Users/nmdl-khb/ColBERT/centroidset_vector 크기 조절/06_newq_margin/q2_diff_flat.npy')
absdiff_real = np.abs(diff_real)

EA_interp = np.interp(np.clip(absdiff_real, diffA.min(), diffA.max()), diffA, EA)
EC_interp = np.interp(np.clip(absdiff_real, diffC.min(), diffC.max()), diffC, EC)

print(f"\n실제 |diff|(Q-C) 샘플수: {absdiff_real.size}")
print(f"Option A E_phase3 기댓값 = {EA_interp.mean()*1e15:.4f} fJ")
print(f"Option C E_phase3 기댓값 = {EC_interp.mean()*1e15:.4f} fJ")
