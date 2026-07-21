import pandas as pd
import numpy as np

READ_US = 0.5

def get_currents(csv_path, t_inj, v2_list, diff_fn):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]
    rows = []
    for v2, (i0, i1) in zip(v2_list, zip(reset_idx, reset_idx[1:])):
        seg = df.iloc[i0:i1]
        seg = seg[seg['TIME'] > t_inj]
        row = seg.iloc[(seg['TIME'] - (t_inj+READ_US*1e-6)).abs().argmin()]
        rows.append((diff_fn(v2), row['I(VVCOMP)'], row['I(VVREAD)'], row['I(VV2P)'], row['I(VV2N)'],
                     row['I(VBL_L)'], row['I(VBL_R)']))
    rows.sort(key=lambda r: r[0])
    return rows

print("=== Option A ===")
print(f"{'diff':>6} {'I_VCOMP(A)':>14} {'I_VREAD(A)':>14} {'I_V2P(A)':>14} {'I_V2N(A)':>14} {'I_VBL_L(A)':>14} {'I_VBL_R(A)':>14}")
v2_listA = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
for d, ic, ir, ip, iN, il, iR in get_currents('phase123_dual_die4.cir.csv', 10000.1e-6, v2_listA, lambda v2: 1.0-v2):
    print(f"{d:6.2f} {ic:14.3e} {ir:14.3e} {ip:14.3e} {iN:14.3e} {il:14.3e} {iR:14.3e}")

print("\n=== Option C ===")
print(f"{'diff':>6} {'I_VCOMP(C)':>14} {'I_VREAD(C)':>14} {'I_V2P(C)':>14} {'I_V2N(C)':>14} {'I_VBL_L(C)':>14} {'I_VBL_R(C)':>14}")
v2_listC = [-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0]
for d, ic, ir, ip, iN, il, iR in get_currents('phase13_optionC_die4.cir.csv', 0.2001e-6, v2_listC, lambda v2: -v2):
    print(f"{d:6.2f} {ic:14.3e} {ir:14.3e} {ip:14.3e} {iN:14.3e} {il:14.3e} {iR:14.3e}")

print("\n참고: RGATE=1e12 ohm 기준, VCOMP/VREAD 예상 누설전류 크기 ~ (수V)/1e12 = 수 pA(1e-12) 수준")
print("V2P/V2N은 CCPL(캐패시터)로만 연결, 저항경로 없음 -> steady-state 전류는 정확히 0이어야 함 (0이 아니면 100% 수치노이즈)")
