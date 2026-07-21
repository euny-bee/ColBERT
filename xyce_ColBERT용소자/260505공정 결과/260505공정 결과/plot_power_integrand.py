"""E = integral P(t) dt 에서 P(t) 자체와 누적적분(E) 그래프를 같이 그림."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SENSE_US = 1.0
src_v = {'I(VVCOMP)': 'V(VCOMPN)', 'I(VVREAD)': 'V(VREADN)',
         'I(VBL_L)': 'V(VBL_L)', 'I(VBL_R)': 'V(VBL_R)',
         'I(VV2P)': 'V(V2P)', 'I(VV2N)': 'V(V2N)'}

def power_and_cumE(seg, t0):
    t = seg['TIME'].values - t0
    p = np.zeros(len(seg))
    for isrc, vsrc in src_v.items():
        if vsrc not in seg.columns or isrc not in seg.columns:
            continue
        v = seg[vsrc].values
        i = seg[isrc].values
        p += -v * i
    e_cum = np.concatenate([[0], np.cumsum((p[1:]+p[:-1])/2 * np.diff(t))])
    return t, p, e_cum

def make_plot(csv_path, t_inj, v2_list, diff_fn, title, outpng, pick_diffs):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    reset_idx = df.index[df['TIME'] == 0].tolist() + [len(df)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(pick_diffs)))

    for v2, (i0, i1) in zip(v2_list, zip(reset_idx, reset_idx[1:])):
        d = diff_fn(v2)
        match = [k for k, pd_ in enumerate(pick_diffs) if abs(pd_ - d) < 0.01]
        if not match:
            continue
        col = colors[match[0]]
        seg = df.iloc[i0:i1]
        seg = seg[(seg['TIME'] > t_inj) & (seg['TIME'] <= t_inj + SENSE_US*1e-6)]
        if len(seg) < 2:
            continue
        t, p, e_cum = power_and_cumE(seg, t_inj)

        axes[0].plot(t*1e6, p*1e6, color=col, lw=1.8, label=f'|diff|={d:.2f}')  # uW
        axes[1].plot(t*1e6, e_cum*1e15, color=col, lw=1.8, label=f'|diff|={d:.2f}  E(1us)={e_cum[-1]*1e15:.1f}fJ')

    axes[0].set_xlabel('Time since V2 injection [us]')
    axes[0].set_ylabel('P(t) = -sum(V_i x I_i)  [uW]')
    axes[0].set_title('적분되는 순간전력 P(t)  (이 아래 면적 = E)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls='--', alpha=0.4)

    axes[1].set_xlabel('Time since V2 injection [us]')
    axes[1].set_ylabel('Cumulative E(t) = integral P dt  [fJ]')
    axes[1].set_title('P(t)를 적분해서 쌓은 누적 에너지 (t=1us 값 = E)')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, ls='--', alpha=0.4)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpng, dpi=150)
    print("Saved:", outpng)

v2_listA = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
make_plot('phase123_dual_die4.cir.csv', 10000.1e-6, v2_listA, lambda v2: abs(1.0-v2),
          'Option A: P(t) and cumulative E(t), 1us sensing window',
          'power_integrand_optionA.png', pick_diffs=[0.0, 0.5, 1.0, 1.5, 2.0])

v2_listC = [-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
make_plot('phase13_optionC_die4.cir.csv', 0.2001e-6, v2_listC, lambda v2: abs(v2),
          'Option C: P(t) and cumulative E(t), 1us sensing window',
          'power_integrand_optionC.png', pick_diffs=[0.0, 0.5, 1.0, 1.5, 2.0])
