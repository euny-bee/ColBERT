"""Phase1->Phase2 write 전류(I) vs 시간 -- V(t) 그림과 짝을 이루는 전류 버전."""
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
    seg = seg[seg['TIME'] > 0]
    segments.append(seg)

colors = plt.cm.viridis(np.linspace(0, 1, len(vth_list)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for vth, seg, col in zip(vth_list, segments, colors):
    t_us = seg['TIME'] * 1e6
    # I(VBL_L), I(VBL_R) = 실제 write 전류 (P=V*I 적분에 쓰인 값)
    axes[0].plot(t_us, seg['I(VBL_L)'].abs(), color=col, lw=1.8, ls='-', label=f'I_BL_L  Vth={vth:.2f}V')
    axes[0].plot(t_us, seg['I(VBL_R)'].abs(), color=col, lw=1.8, ls='--')
    axes[1].plot(t_us, seg['I(VBL_L)'].abs(), color=col, lw=1.8, ls='-', label=f'I_BL_L  Vth={vth:.2f}V')
    axes[1].plot(t_us, seg['I(VBL_R)'].abs(), color=col, lw=1.8, ls='--')

axes[0].axvline(0.2, color='gray', ls=':', lw=1, alpha=0.7, label='t1=0.2us (Phase1->2)')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_xlabel('Time [us] (log)'); axes[0].set_ylabel('|I(VBL_L,VBL_R)| [A] (log)')
axes[0].set_title('Full window (0-500000us=500ms, log-log)\n이게 E=integral V*I dt 에 실제 쓰인 전류')
axes[0].legend(fontsize=7); axes[0].grid(True, ls='--', alpha=0.4)

axes[1].axvline(0.2, color='gray', ls=':', lw=1, alpha=0.7, label='t1=0.2us (Phase1->2)')
axes[1].set_xlim(0, 1.0)
axes[1].set_yscale('log')
axes[1].set_xlabel('Time [us]'); axes[1].set_ylabel('|I(VBL_L,VBL_R)| [A] (log)')
axes[1].set_title('Zoom (0-1us): Phase1->Phase2 transition')
axes[1].legend(fontsize=7); axes[1].grid(True, ls='--', alpha=0.4)

fig.suptitle('Phase1->Phase2 write 전류 I(t)  (V(SN) 그림과 같은 시뮬레이션의 전류 데이터)')
fig.tight_layout()
fig.savefig('phase12_dual_currents.png', dpi=150)
print("Saved: phase12_dual_currents.png")
