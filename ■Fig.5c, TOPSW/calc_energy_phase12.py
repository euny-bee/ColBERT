"""Phase1(precharge)/Phase2(store) 에너지 계산 -- V1=1.0V, Vth 5개 스윕."""
import pandas as pd
import numpy as np

df = pd.read_csv('phase12_dual_die4.cir.csv')
df = df.loc[:, ~df.columns.duplicated()]

vth_list = [0.03, 0.53, 1.03, 2.03, 3.03]
reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]

src_cols = ['I(VVML)', 'I(VVCOMP)', 'I(VVREAD)', 'I(VBL_L)', 'I(VBL_R)']
src_v = {'I(VVML)': 'V(VML)' if 'V(VML)' in df.columns else None,  # VML=0 -> 에너지 기여 0, 생략 가능
         'I(VVCOMP)': 'V(VCOMPN)',
         'I(VVREAD)': 'V(VREADN)',
         'I(VBL_L)': 'V(VBL_L)',
         'I(VBL_R)': 'V(VBL_R)'}

def energy_from_sources(seg):
    """각 독립 전압원이 회로에 공급한 에너지 합 (부호: 양수=공급, |E| 사용)."""
    t = seg['TIME'].values
    total = 0.0
    detail = {}
    for isrc, vsrc in src_v.items():
        if vsrc is None or vsrc not in seg.columns:
            continue
        v = seg[vsrc].values
        i = seg[isrc].values
        # Xyce 컨벤션: I(V) = + 단자로 유입되는 전류 -> 소스가 공급하는 전력 = -V*I
        p = -v * i
        e = np.trapz(p, t)
        detail[isrc] = e
        total += e
    return total, detail

print("=== Phase1 (0 ~ 0.2us) 에너지 -- Vth 무관하므로 첫 세그먼트만 계산 ===")
seg0 = df.iloc[reset_idx[0]:reset_idx[1]]
seg0 = seg0[(seg0['TIME'] > 0) & (seg0['TIME'] <= 0.2e-6)]
e1_total, e1_detail = energy_from_sources(seg0)
print(f"E_phase1_total = {e1_total*1e15:.4f} fJ")
for k, v in e1_detail.items():
    print(f"   {k}: {v*1e15:+.4f} fJ")

print("\n=== Phase2 (0.2us ~ 500ms) 에너지 -- Vth별 ===")
results = []
for vth, (i0, i1) in zip(vth_list, zip(reset_idx, reset_idx[1:])):
    seg = df.iloc[i0:i1]
    seg = seg[seg['TIME'] > 0.2e-6]
    e2_total, e2_detail = energy_from_sources(seg)
    results.append((vth, e2_total))
    print(f"Vth={vth:.2f}V  E_phase2_total = {e2_total*1e15:.4f} fJ")
    for k, v in e2_detail.items():
        print(f"   {k}: {v*1e15:+.4f} fJ")

print("\n=== 요약 (fJ) ===")
print(f"E_phase1 (공통, Vth 무관) = {e1_total*1e15:.4f} fJ")
for vth, e2 in results:
    print(f"Vth={vth:.2f}V  E_phase2 = {e2*1e15:.4f} fJ   E_phase1+phase2(Option A 쓰기 총합) = {(e1_total+e2)*1e15:.4f} fJ")
print(f"\nOption C 쓰기(=Phase1만) = {e1_total*1e15:.4f} fJ  (Vth 무관, 5개 Vth 모두 동일)")
