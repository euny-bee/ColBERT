"""Phase3 search 과도현상 디버그: I(t), 누적에너지(t) vs diff, 1us sensing window 주변."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SENSE_US = 1.0
src_v = {'I(VVCOMP)': 'V(VCOMPN)', 'I(VVREAD)': 'V(VREADN)',
         'I(VBL_L)': 'V(VBL_L)', 'I(VBL_R)': 'V(VBL_R)',
         'I(VV2P)': 'V(V2P)', 'I(VV2N)': 'V(V2N)'}

def cum_energy(seg, t0):
    t = seg['TIME'].values - t0
    p = np.zeros(len(seg))
    for isrc, vsrc in src_v.items():
        if vsrc not in seg.columns or isrc not in seg.columns:
            continue
        v = seg[vsrc].values
        i = seg[isrc].values
        p += -v * i
    e_cum = np.concatenate([[0], np.cumsum((p[1:]+p[:-1])/2 * np.diff(t))])
    return t, e_cum

def make_plot(csv_path, t_inj, v2_list, diff_fn, title, outpng):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    colors = plt.cm.turbo(np.linspace(0, 1, len(v2_list)))

    for v2, col, (i0, i1) in zip(v2_list, colors, zip(reset_idx, reset_idx[1:])):
        d = diff_fn(v2)
        seg = df.iloc[i0:i1]
        seg = seg[(seg['TIME'] > t_inj) & (seg['TIME'] <= t_inj + 5e-6)]
        if len(seg) < 2:
            continue
        t_us = (seg['TIME'].values - t_inj) * 1e6

        # (a) I(M0), I(M3) magnitude
        axes[0].plot(t_us, seg['I(XM0:BIGZO)'].abs(), color=col, lw=1.3, ls='-')
        axes[0].plot(t_us, seg['I(XM3:BIGZO)'].abs(), color=col, lw=1.3, ls='--')

        # (b) V(SN), V(SNB)
        axes[1].plot(t_us, seg['V(SN)'], color=col, lw=1.3, ls='-')
        axes[1].plot(t_us, seg['V(SNB)'], color=col, lw=1.3, ls='--')

        # (c) cumulative energy
        t_e, e_cum = cum_energy(seg, t_inj)
        axes[2].plot(t_e*1e6, e_cum*1e15, color=col, lw=1.8, label=f'|diff|={d:.2f}')

    for ax in axes[:2]:
        ax.set_xscale('log')
        ax.axvline(SENSE_US, color='k', ls=':', lw=1.2, alpha=0.7)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Time since V2 injection [us] (log)')
    axes[0].set_ylabel('|I| [A] (log), M0 solid / M3 dashed')
    axes[0].set_title('Current transient')
    axes[0].grid(True, ls='--', alpha=0.4)

    axes[1].set_xlabel('Time since V2 injection [us] (log)')
    axes[1].set_ylabel('V(SN) solid / V(SNB) dashed [V]')
    axes[1].set_title('Voltage transient')
    axes[1].grid(True, ls='--', alpha=0.4)

    axes[2].axvline(SENSE_US, color='k', ls=':', lw=1.2, alpha=0.7, label='1us sensing point')
    axes[2].set_xlabel('Time since V2 injection [us] (linear)')
    axes[2].set_ylabel('Cumulative energy [fJ]')
    axes[2].set_title('Cumulative energy (crossing = non-monotonic @ 1us)')
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, ls='--', alpha=0.4)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpng, dpi=150)
    print("Saved:", outpng)

# Option A: V1=1.0 fixed, diff = |1 - V2|
v2_listA = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
make_plot('phase123_dual_die4.cir.csv', 10000.1e-6, v2_listA, lambda v2: abs(1.0-v2),
          'Option A (Vth-comp) Phase3 transient, V1=1.0V fixed, V2 sweep',
          'phase3_transient_debug_optionA.png')

# Option C: diff = |V2| directly
v2_listC = [-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
make_plot('phase13_optionC_die4.cir.csv', 0.2001e-6, v2_listC, lambda v2: abs(v2),
          'Option C (No comp) Phase3 transient, V2=diff direct',
          'phase3_transient_debug_optionC.png')
