"""Phase3(search) 에너지 -- Option A(phase123) vs Option C(phase13) -- 1us sensing window."""
import pandas as pd
import numpy as np

SENSE_US = 1.0  # user-agreed sensing time

src_v = {'I(VVCOMP)': 'V(VCOMPN)', 'I(VVREAD)': 'V(VREADN)',
         'I(VBL_L)': 'V(VBL_L)', 'I(VBL_R)': 'V(VBL_R)',
         'I(VV2P)': 'V(V2P)', 'I(VV2N)': 'V(V2N)'}

def energy_from_sources(seg):
    t = seg['TIME'].values
    total = 0.0
    detail = {}
    for isrc, vsrc in src_v.items():
        if vsrc not in seg.columns or isrc not in seg.columns:
            continue
        v = seg[vsrc].values
        i = seg[isrc].values
        p = -v * i
        e = np.trapz(p, t)
        detail[isrc] = e
        total += e
    return total, detail

v2_list = [-1.0, -0.5, 0.0, 0.5, 1.0]

print("=========== Option A (phase123_dual_die4, V2 inject @ t2=10000.1us) ===========")
dfA = pd.read_csv('phase123_dual_die4.cir.csv')
dfA = dfA.loc[:, ~dfA.columns.duplicated()]
reset_idx = dfA.index[dfA['TIME'] == 0].tolist() + [len(dfA)]
t_inj_A = 10000.1e-6
tA_end = t_inj_A + SENSE_US*1e-6
E_A = []
for v2, (i0, i1) in zip(v2_list, zip(reset_idx, reset_idx[1:])):
    seg = dfA.iloc[i0:i1]
    seg = seg[(seg['TIME'] > t_inj_A) & (seg['TIME'] <= tA_end)]
    e, detail = energy_from_sources(seg)
    E_A.append(e)
    print(f"V2={v2:+.2f}V  E_phase3 = {e*1e15:.4f} fJ   " + "  ".join(f"{k}:{v*1e15:+.3f}" for k,v in detail.items()))

print("\n=========== Option C (phase13_optionC_die4, V2 inject @ t1=0.2001us) ===========")
dfC = pd.read_csv('phase13_optionC_die4.cir.csv')
dfC = dfC.loc[:, ~dfC.columns.duplicated()]
reset_idx = dfC.index[dfC['TIME'] == 0].tolist() + [len(dfC)]
t_inj_C = 0.2001e-6
tC_end = t_inj_C + SENSE_US*1e-6
E_C = []
for v2, (i0, i1) in zip(v2_list, zip(reset_idx, reset_idx[1:])):
    seg = dfC.iloc[i0:i1]
    seg = seg[(seg['TIME'] > t_inj_C) & (seg['TIME'] <= tC_end)]
    e, detail = energy_from_sources(seg)
    E_C.append(e)
    print(f"V2={v2:+.2f}V  E_phase3 = {e*1e15:.4f} fJ   " + "  ".join(f"{k}:{v*1e15:+.3f}" for k,v in detail.items()))

print("\n=========== 요약 ===========")
print("Option A E_phase3(fJ) per V2:", [f"{e*1e15:.4f}" for e in E_A])
print("Option C E_phase3(fJ) per V2:", [f"{e*1e15:.4f}" for e in E_C])
print(f"Option A E_phase3 평균(|V2| 전체) = {np.mean(np.abs(E_A))*1e15:.4f} fJ")
print(f"Option C E_phase3 평균(|V2| 전체) = {np.mean(np.abs(E_C))*1e15:.4f} fJ")
