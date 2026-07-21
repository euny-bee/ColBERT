"""
plot_seed_sweep.py에서 q1 제거한 버전 -- 기존 [vth_v3_fixedcell]seed_sweep_optC_vs_A.png는 그대로 두고 새 파일로 저장.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'

def _setup_matplotlib():
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": 13,
        "font.weight": "bold", "axes.labelsize": 15, "axes.labelweight": "bold",
        "axes.titlesize": 15, "axes.titleweight": "bold",
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.unicode_minus": False,
    })
_setup_matplotlib()

df = pd.read_csv(f'{BASE}/seed_sweep_optC_results.csv')

Q_INFO = {
    0: {'label': 'q0 (margin=1.62, tight race)',   'color': '#E53935'},
    2: {'label': 'q2 (margin=20.06, dominant win)', 'color': '#43A047'},
}

fig, ax = plt.subplots(figsize=(11, 6))

for q_id, info in Q_INFO.items():
    sub = df[df['q_id'] == q_id].sort_values('seed')
    ax.plot(sub['seed'], sub['rank_C'], marker='o', ms=5, lw=1.3,
            color=info['color'], alpha=0.9, label=f"{info['label']} — No Comp", zorder=3)
    rank_A = sub['rank_A'].iloc[0]
    ax.axhline(rank_A, color=info['color'], ls='--', lw=1.6, alpha=0.6, zorder=2)

ax.set_xlabel('Vth random seed')
ax.set_ylabel('No Comp (Option C) rank of true passage')
ax.set_title('No Comp rank volatility vs margin (dashed line = Vth Comp, seed-invariant)')
ax.set_xticks(range(0, 30, 2))
ax.grid(True, ls='--', alpha=0.3)
ax.set_ylim(0, df['rank_C'].max() + 1)

handles = []
for q_id, info in Q_INFO.items():
    handles.append(mlines.Line2D([], [], color=info['color'], marker='o', ms=5, lw=1.3,
                                  label=f"{info['label']} — No Comp"))
handles.append(mlines.Line2D([], [], color='black', ls='--', lw=1.6, alpha=0.6,
                              label='Vth Comp (Option A) — all seeds, rank 1 for every query'))
ax.legend(handles=handles, loc='upper left', fontsize=10.5, framealpha=0.9)

fig.tight_layout()
out = f'{BASE}/[vth_v3_fixedcell]seed_sweep_optC_vs_A_no_q1.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'저장: {out}')
