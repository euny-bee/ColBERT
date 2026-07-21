"""Phase1->Phase2->Phase3 dual Xyce 결과 플롯 (V2 5개 스윕): V(SN), V(SNB) vs time."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('phase123_dual_die4.cir.csv')
df = df.loc[:, ~df.columns.duplicated()]

v2_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]

segments = []
for i in range(len(v2_list)):
    seg = df.iloc[reset_idx[i]:reset_idx[i+1]].copy()
    seg = seg[seg['TIME'] > 0]  # t=0 DC 아티팩트 제외
    segments.append(seg)

colors = plt.cm.plasma(np.linspace(0, 1, len(v2_list)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for v2, seg, col in zip(v2_list, segments, colors):
    t_us = seg['TIME'] * 1e6
    axes[0].plot(t_us, seg['V(SN)'], color=col, lw=1.8, ls='-', label=f'SN  V2={v2:+.2f}V')
    axes[0].plot(t_us, seg['V(SNB)'], color=col, lw=1.8, ls='--')
    axes[1].plot(t_us, seg['V(SN)'], color=col, lw=1.8, ls='-', label=f'SN  V2={v2:+.2f}V')
    axes[1].plot(t_us, seg['V(SNB)'], color=col, lw=1.8, ls='--')

axes[0].axvline(0.2, color='gray', ls=':', lw=1, alpha=0.7, label='t1 (Phase1->2)')
axes[0].axvline(10000, color='black', ls=':', lw=1, alpha=0.7, label='t2 (Phase2->3)')
axes[0].set_xscale('log')
axes[0].set_xlabel('Time [us] (log)'); axes[0].set_ylabel('V(SN) solid / V(SNB) dashed [V]')
axes[0].set_title('Full window (0-20000us, log time)')
axes[0].legend(fontsize=7); axes[0].grid(True, ls='--', alpha=0.4)

axes[1].axvline(10000, color='black', ls=':', lw=1, alpha=0.7, label='t2 (Phase2->3)')
axes[1].set_xlim(9900, 10100)
axes[1].set_xlabel('Time [us]'); axes[1].set_ylabel('V(SN) solid / V(SNB) dashed [V]')
axes[1].set_title('Zoom around t2: Phase2->Phase3 transition')
axes[1].legend(fontsize=7); axes[1].grid(True, ls='--', alpha=0.4)

fig.suptitle('Phase1->Phase2->Phase3 dual die4, V1=1.0V, V2 sweep (Vth=0.151V nominal)')
fig.tight_layout()
fig.savefig('phase123_dual_settling.png', dpi=150)
print("Saved: phase123_dual_settling.png")

# --- I(M0), I(M3) vs time ---
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

for v2, seg, col in zip(v2_list, segments, colors):
    t_us = seg['TIME'] * 1e6
    axes2[0].plot(t_us, seg['I(XM0:BIGZO)'].abs(), color=col, lw=1.8, ls='-', label=f'I_M0  V2={v2:+.2f}V')
    axes2[0].plot(t_us, seg['I(XM3:BIGZO)'].abs(), color=col, lw=1.8, ls='--')
    axes2[1].plot(t_us, seg['I(XM0:BIGZO)'].abs(), color=col, lw=1.8, ls='-', label=f'I_M0  V2={v2:+.2f}V')
    axes2[1].plot(t_us, seg['I(XM3:BIGZO)'].abs(), color=col, lw=1.8, ls='--')

axes2[0].axvline(0.2, color='gray', ls=':', lw=1, alpha=0.7, label='t1 (Phase1->2)')
axes2[0].axvline(10000, color='black', ls=':', lw=1, alpha=0.7, label='t2 (Phase2->3)')
axes2[0].set_xscale('log'); axes2[0].set_yscale('log')
axes2[0].set_xlabel('Time [us] (log)'); axes2[0].set_ylabel('|I| [A] (log), M0 solid / M3 dashed')
axes2[0].set_title('Full window (0-20000us, log-log)')
axes2[0].legend(fontsize=7); axes2[0].grid(True, ls='--', alpha=0.4)

axes2[1].axvline(10000, color='black', ls=':', lw=1, alpha=0.7, label='t2 (Phase2->3)')
axes2[1].set_xlim(9900, 10100)
axes2[1].set_yscale('log')
axes2[1].set_xlabel('Time [us]'); axes2[1].set_ylabel('|I| [A] (log), M0 solid / M3 dashed')
axes2[1].set_title('Zoom around t2: Phase2->Phase3 transition')
axes2[1].legend(fontsize=7); axes2[1].grid(True, ls='--', alpha=0.4)

fig2.suptitle('I(M0), I(M3) dual die4, V1=1.0V, V2 sweep (Vth=0.151V nominal)')
fig2.tight_layout()
fig2.savefig('phase123_dual_currents.png', dpi=150)
print("Saved: phase123_dual_currents.png")
