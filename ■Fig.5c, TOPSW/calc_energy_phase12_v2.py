"""Phase1/Phase2 에너지 -- Vth별 실제 settling 시간까지만 적분 + worst-case 고정시간 버전 둘 다 계산."""
import pandas as pd
import numpy as np

df = pd.read_csv('phase12_dual_die4.cir.csv')
df = df.loc[:, ~df.columns.duplicated()]

vth_list = [0.03, 0.53, 1.03, 2.03, 3.03]
reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]

src_v = {'I(VVML)': 'V(VML)' if 'V(VML)' in df.columns else None,
         'I(VVCOMP)': 'V(VCOMPN)',
         'I(VVREAD)': 'V(VREADN)',
         'I(VBL_L)': 'V(VBL_L)',
         'I(VBL_R)': 'V(VBL_R)'}

def energy_from_sources(seg):
    t = seg['TIME'].values
    total = 0.0
    for isrc, vsrc in src_v.items():
        if vsrc is None or vsrc not in seg.columns:
            continue
        v = seg[vsrc].values
        i = seg[isrc].values
        p = -v * i
        total += np.trapz(p, t)
    return total

# per-Vth own settling time (from check_phase2_settle.py, +20% margin)
settle_us = {0.03: 8.776, 0.53: 904.598, 1.03: 2666.729, 2.03: 2550.337, 3.03: 2147.684}
WORST_CASE_US = max(settle_us.values()) * 1.2   # fixed write time for all Vth (real chip: unknown Vth in advance)

print("=== Phase1 (0~0.2us, Vth 무관) ===")
seg0 = df.iloc[reset_idx[0]:reset_idx[1]]
seg0 = seg0[(seg0['TIME'] > 0) & (seg0['TIME'] <= 0.2e-6)]
e1 = energy_from_sources(seg0)
print(f"E_phase1 = {e1*1e15:.4f} fJ")

print(f"\n=== Phase2, 방법1: Vth별 실제 settling까지(x1.2 margin) ===")
for vth, (i0, i1) in zip(vth_list, zip(reset_idx, reset_idx[1:])):
    seg = df.iloc[i0:i1]
    tmax = 0.2e-6 + settle_us[vth]*1.2*1e-6
    seg = seg[(seg['TIME'] > 0.2e-6) & (seg['TIME'] <= tmax)]
    e2 = energy_from_sources(seg)
    print(f"  Vth={vth:.2f}V  t_cut={settle_us[vth]*1.2:.1f}us  E_phase2={e2*1e15:.4f} fJ  total={( e1+e2)*1e15:.4f} fJ")

print(f"\n=== Phase2, 방법2: 전체 Vth 공통 고정 write-time = worst-case*1.2 = {WORST_CASE_US:.1f}us ===")
for vth, (i0, i1) in zip(vth_list, zip(reset_idx, reset_idx[1:])):
    seg = df.iloc[i0:i1]
    tmax = 0.2e-6 + WORST_CASE_US*1e-6
    seg = seg[(seg['TIME'] > 0.2e-6) & (seg['TIME'] <= tmax)]
    e2 = energy_from_sources(seg)
    print(f"  Vth={vth:.2f}V  E_phase2={e2*1e15:.4f} fJ  total={(e1+e2)*1e15:.4f} fJ")
