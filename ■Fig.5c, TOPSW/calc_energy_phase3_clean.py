"""Phase3 에너지 -- 스냅샷 전류(0.5us, steady-state) x 고정 전압 x T_sense, 타임스텝 이슈 없이 산술적으로 계산."""
import pandas as pd
import numpy as np

T_SENSE = 1.0e-6   # 1us

def get_snapshot(csv_path, t_inj, v2_list, diff_fn, v_bl):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]
    rows = []
    for v2, (i0, i1) in zip(v2_list, zip(reset_idx, reset_idx[1:])):
        seg = df.iloc[i0:i1]
        seg = seg[seg['TIME'] > t_inj]
        row = seg.iloc[(seg['TIME'] - (t_inj+0.5e-6)).abs().argmin()]
        d = diff_fn(v2)
        # 고정 바이어스 전압 (netlist 상수, Phase3 구간)
        V_VCOMP, V_VREAD = -3.0, 3.0
        V_BL_L, V_BL_R = v_bl, v_bl
        V_V2P, V_V2N = v2, -v2
        P = -(V_VCOMP*row['I(VVCOMP)'] + V_VREAD*row['I(VVREAD)']
              + V_BL_L*row['I(VBL_L)'] + V_BL_R*row['I(VBL_R)']
              + V_V2P*row['I(VV2P)'] + V_V2N*row['I(VV2N)'])
        E = P * T_SENSE
        rows.append((d, P, E))
    rows.sort(key=lambda r: r[0])
    return np.array(rows)

v2_listA = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
rA = get_snapshot('phase123_dual_die4.cir.csv', 10000.1e-6, v2_listA, lambda v2: 1.0-v2, v_bl=1.7)

v2_listC = [-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
rC_raw = get_snapshot('phase13_optionC_die4.cir.csv', 0.2001e-6, v2_listC, lambda v2: abs(v2), v_bl=1.7)
# Option C diff는 절대값이라 중복(±V2) 발생 -> 평균으로 중복 제거
diffC_u, inv = np.unique(rC_raw[:,0], return_inverse=True)
EC_u = np.array([rC_raw[inv==k,2].mean() for k in range(len(diffC_u))])
PC_u = np.array([rC_raw[inv==k,1].mean() for k in range(len(diffC_u))])

print("Option A: diff, P(W), E(fJ)")
for d,p,e in rA:
    print(f"  diff={d:+.2f}  P={p*1e6:.4f}uW  E={e*1e15:.4f}fJ")
print("\nOption C: diff, P(W), E(fJ)")
for d,p,e in zip(diffC_u, PC_u, EC_u):
    print(f"  diff={d:.2f}  P={p*1e6:.4f}uW  E={e*1e15:.4f}fJ")

# 실제 q2 diff 분포로 가중평균
diff_real = np.load('/mnt/c/Users/nmdl-khb/ColBERT/centroidset_vector 크기 조절/06_newq_margin/q2_diff_flat.npy')
absdiff_real = np.abs(diff_real)

diffA, EA = rA[:,0], rA[:,2]
EA_interp = np.interp(np.clip(absdiff_real, diffA.min(), diffA.max()), diffA, EA)
EC_interp = np.interp(np.clip(absdiff_real, diffC_u.min(), diffC_u.max()), diffC_u, EC_u)

print(f"\n실제 |diff| 샘플수: {absdiff_real.size}")
print(f"Option A E_phase3 기댓값 = {EA_interp.mean()*1e15:.4f} fJ")
print(f"Option C E_phase3 기댓값 = {EC_interp.mean()*1e15:.4f} fJ")
