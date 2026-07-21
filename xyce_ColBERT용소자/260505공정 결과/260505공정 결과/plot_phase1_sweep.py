"""Phase1 Vth 스윕 결과 플롯 (t1 재산정용)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('phase1_dual_die4_sweep.cir.csv')
df = df.loc[:, ~df.columns.duplicated()]

vth_list = [0.03, 0.53, 1.03, 2.03, 3.03]
reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]
segments = []
for i in range(len(vth_list)):
    seg = df.iloc[reset_idx[i]:reset_idx[i+1]].copy()
    seg = seg[seg['TIME'] > 0]
    segments.append(seg)

colors = plt.cm.viridis(np.linspace(0, 1, len(vth_list)))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for vth, seg, col in zip(vth_list, segments, colors):
    t_us = seg['TIME'] * 1e6
    axes[0].plot(t_us, seg['V(SN)'], color=col, lw=1.8, label=f'Vth={vth:.2f}V')
    axes[1].plot(t_us, seg['V(SN)'], color=col, lw=1.8, label=f'Vth={vth:.2f}V')

axes[0].axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
axes[0].set_xlabel('Time [us]'); axes[0].set_ylabel('V(SN) [V]')
axes[0].set_title('Full window (0-20us)')
axes[0].legend(fontsize=8); axes[0].grid(True, ls='--', alpha=0.4)

axes[1].axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
axes[1].set_xlim(0, 2)
axes[1].set_xlabel('Time [us]'); axes[1].set_ylabel('V(SN) [V]')
axes[1].set_title('Zoom (0-2us)')
axes[1].legend(fontsize=8); axes[1].grid(True, ls='--', alpha=0.4)

fig.suptitle('Phase1 precharge, Vth sweep -- worst-case settling time 확인')
fig.tight_layout()
fig.savefig('phase1_sweep_settling.png', dpi=150)
print("Saved: phase1_sweep_settling.png")

print("\n[Vth별 정착시간 (|SN|<1mV 기준)]")
tol = 1e-3
for vth, seg in zip(vth_list, segments):
    below = seg['V(SN)'].abs() < tol
    if below.any():
        t_settle = seg.loc[below.idxmax(), 'TIME'] * 1e6
        print(f"  Vth={vth:.2f}V  t_settle={t_settle:.4f}us   SN_final={seg['V(SN)'].iloc[-1]*1e3:.4f}mV")
    else:
        print(f"  Vth={vth:.2f}V  20us 내에 정착 못함(|SN|>=1mV)   SN_final={seg['V(SN)'].iloc[-1]*1e3:.4f}mV")
