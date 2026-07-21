"""I(M0), I(M3) vs |diff| -- 실제 SPICE 시뮬레이션 결과 (steady-state 값, 0.5us 지점)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

READ_US = 0.5   # steady-state 구간 내 임의 시점 (평평하므로 어디든 동일)

def get_I_vs_diff(csv_path, t_inj, v2_list, diff_fn):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]
    diffs, I_L, I_R = [], [], []
    for v2, (i0, i1) in zip(v2_list, zip(reset_idx, reset_idx[1:])):
        seg = df.iloc[i0:i1]
        seg = seg[seg['TIME'] > t_inj]
        row = seg.iloc[(seg['TIME'] - (t_inj+READ_US*1e-6)).abs().argmin()]
        diffs.append(diff_fn(v2))
        I_L.append(abs(row['I(VBL_L)']))
        I_R.append(abs(row['I(VBL_R)']))
    order = np.argsort(diffs)
    return np.array(diffs)[order], np.array(I_L)[order], np.array(I_R)[order]

v2_listA = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
dA, ILa, IRa = get_I_vs_diff('phase123_dual_die4.cir.csv', 10000.1e-6, v2_listA, lambda v2: 1.0-v2)  # signed diff = V2-V1

v2_listC = [-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
dC, ILc, IRc = get_I_vs_diff('phase13_optionC_die4.cir.csv', 0.2001e-6, v2_listC, lambda v2: -v2)  # signed diff

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].plot(dA, ILa*1e6, 'o-', color='tab:blue', label='I_M0 (=I(VBL_L))')
axes[0].plot(dA, IRa*1e6, 's-', color='tab:red', label='I_M3 (=I(VBL_R))')
axes[0].set_yscale('log')
axes[0].set_xlabel('signed diff = V2 - V1  [V]')
axes[0].set_ylabel('|I|  [uA]  (log)')
axes[0].set_title('Option A (Vth-comp, 실제 SPICE)')
axes[0].legend(); axes[0].grid(True, ls='--', alpha=0.4)

axes[1].plot(dC, ILc*1e6, 'o-', color='tab:blue', label='I_M0 (=I(VBL_L))')
axes[1].plot(dC, IRc*1e6, 's-', color='tab:red', label='I_M3 (=I(VBL_R))')
axes[1].set_yscale('log')
axes[1].set_xlabel('signed diff = -V2  [V]')
axes[1].set_ylabel('|I|  [uA]  (log)')
axes[1].set_title('Option C (No comp, 실제 SPICE)')
axes[1].legend(); axes[1].grid(True, ls='--', alpha=0.4)

fig.suptitle('Steady-state I vs diff -- 실제 Xyce SPICE 결과 (0.5us 지점, 9~17개 anchor point)')
fig.tight_layout()
fig.savefig('I_vs_diff_actual_spice.png', dpi=150)
print("Saved: I_vs_diff_actual_spice.png")

for d,l,r in zip(dA, ILa, IRa):
    print(f"A  diff={d:+.2f}  I_M0={l*1e6:.4f}uA  I_M3={r*1e6:.4f}uA")
print()
for d,l,r in zip(dC, ILc, IRc):
    print(f"C  diff={d:+.2f}  I_M0={l*1e6:.4f}uA  I_M3={r*1e6:.4f}uA")
