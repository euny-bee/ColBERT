"""Phase1->Phase2 dual Xyce 결과 플롯 (Vth 5개 스윕): V(SN), V(SNB) vs time. (RLEAK=10G, V1=0.5V)"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('phase12_dual_die4.cir.csv')
df = df.loc[:, ~df.columns.duplicated()]

vth_list = [0.03, 0.53, 1.03, 2.03, 3.03]
reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]

segments = []
for i in range(len(vth_list)):
    seg = df.iloc[reset_idx[i]:reset_idx[i+1]].copy()
    seg = seg[seg['TIME'] > 0]  # t=0 DC 아티팩트 제외
    segments.append(seg)

colors = plt.cm.viridis(np.linspace(0, 1, len(vth_list)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for vth, seg, col in zip(vth_list, segments, colors):
    t_us = seg['TIME'] * 1e6
    axes[0].plot(t_us, seg['V(SN)'], color=col, lw=1.8, ls='-', label=f'SN  Vth={vth:.2f}V')
    axes[0].plot(t_us, seg['V(SNB)'], color=col, lw=1.8, ls='--')
    axes[1].plot(t_us, seg['V(SN)'], color=col, lw=1.8, ls='-', label=f'SN  Vth={vth:.2f}V')
    axes[1].plot(t_us, seg['V(SNB)'], color=col, lw=1.8, ls='--')

axes[0].axvline(0.2, color='gray', ls=':', lw=1, alpha=0.7, label='t1=0.2us (Phase1->2)')
axes[0].set_xscale('log')
axes[0].set_xlabel('Time [us] (log)'); axes[0].set_ylabel('V(SN) solid / V(SNB) dashed [V]')
axes[0].set_title('Full window (0-500000us=500ms, log time)')
axes[0].legend(fontsize=7); axes[0].grid(True, ls='--', alpha=0.4)

axes[1].axvline(0.2, color='gray', ls=':', lw=1, alpha=0.7, label='t1=0.2us (Phase1->2)')
axes[1].set_xlim(0, 1.0)
axes[1].set_xlabel('Time [us]'); axes[1].set_ylabel('V(SN) solid / V(SNB) dashed [V]')
axes[1].set_title('Zoom (0-1us): Phase1->Phase2 transition')
axes[1].legend(fontsize=7); axes[1].grid(True, ls='--', alpha=0.4)

fig.suptitle('Phase1->Phase2 dual die4 (RLEAK=10G), Vth sweep, V1=0.5V bitline')
fig.tight_layout()
fig.savefig('phase12_dual_settling.png', dpi=150)
print("Saved: phase12_dual_settling.png")

print("\n[Phase2 최종 정착값 확인]")
for vth, seg in zip(vth_list, segments):
    sn_final = seg['V(SN)'].iloc[-1]
    snb_final = seg['V(SNB)'].iloc[-1]
    print(f"  Vth={vth:.2f}V  ->  SN_final={sn_final:+.4f}V   SNB_final={snb_final:+.4f}V")
