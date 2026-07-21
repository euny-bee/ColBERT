"""
plot_seed19_zoom100.py의 60x60 확대 버전 -- 기존 파일들은 그대로 두고 새 파일로 저장.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm

FONT_BASE, FONT_AXIS, FONT_PANEL, GRID_ALPHA = 13, 14, 15, 0.3
ZOOM_LIM = 60

def _setup_matplotlib():
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": FONT_BASE,
        "font.weight": "bold", "axes.labelsize": FONT_AXIS, "axes.labelweight": "bold",
        "axes.titlesize": FONT_PANEL, "axes.titleweight": "bold",
        "xtick.labelsize": FONT_BASE, "ytick.labelsize": FONT_BASE, "axes.unicode_minus": False,
    })

def _style_ax(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8, ls="--")
    ax.minorticks_off()

_setup_matplotlib()

IN_BASE  = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
OUT_BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
QUERY_RELEVANT = {0: 471602, 1: 822108, 2: 347885}
Q_LABELS = [
    'q0: "...house reshingled" (margin=1.62, seed=19)',
    'q1: "wave amplitude def." (margin=11.35, seed=19)',
    'q2: "where is hartwell ga" (margin=20.06, seed=19)',
]

C3 = {'optA': '#2196F3', 'optC': '#FF7811', 'true': '#E91E63'}

print("데이터 로드 중...")
step6 = pd.read_excel(f'{IN_BASE}/[seed19]step6_all_results.xlsx', sheet_name=None)

dig_rank, optA_rank, optC_rank = {}, {}, {}
for q_id in [0, 1, 2]:
    dig_rank[q_id]  = step6[f'q{q_id}_digital'].set_index('pid')
    optA_rank[q_id] = step6[f'q{q_id}_optA'].set_index('pid')
    optC_rank[q_id] = step6[f'q{q_id}_optC'].set_index('pid')

print("Row 0 (scatter, zoom 0~60) 그리는 중...")
fig0, axes0 = plt.subplots(1, 3, figsize=(18, 5))

for panel_idx, q_id in enumerate([0, 1, 2]):
    ax       = axes0[panel_idx]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig, df_A, df_C = dig_rank[q_id], optA_rank[q_id], optC_rank[q_id]

    common_A = [p for p in df_dig.index if p in df_A.index]
    xA = [int(df_dig.loc[p, 'rank_dig_f32']) for p in common_A]
    yA = [int(df_A.loc[p, 'rank_optA'])      for p in common_A]
    sA = [200 if p == true_pid else 20 for p in common_A]
    ax.scatter(xA, yA, c=C3['optA'], s=sA, alpha=0.85, zorder=4, clip_on=False)

    common_C = [p for p in df_dig.index if p in df_C.index]
    xC = [int(df_dig.loc[p, 'rank_dig_f32']) for p in common_C]
    yC = [int(df_C.loc[p, 'rank_optC'])      for p in common_C]
    sC = [180 if p == true_pid else 20 for p in common_C]
    ax.scatter(xC, yC, c=C3['optC'], s=sC, alpha=0.85, zorder=4, marker='D', clip_on=False)

    for df_m, rank_col in [(df_A, 'rank_optA'), (df_C, 'rank_optC')]:
        if true_pid in df_m.index and true_pid in df_dig.index:
            tx = int(df_dig.loc[true_pid, 'rank_dig_f32'])
            ty = int(df_m.loc[true_pid, rank_col])
            ax.scatter(tx, ty, c=C3['true'], s=260, marker='*', zorder=6, clip_on=False)

    ax.plot([0, ZOOM_LIM], [0, ZOOM_LIM], 'k--', lw=0.8, alpha=0.3)
    ax.set_xlabel('Digital rank', fontsize=18)
    ax.set_ylabel('Analog rank', fontsize=18)
    ax.tick_params(labelsize=17)
    ax.set_title(f'{Q_LABELS[q_id]}\ncommon w/ A: {len(common_A)}  |  w/ C: {len(common_C)}', fontsize=12)
    ax.set_xlim(0, ZOOM_LIM)
    ax.set_ylim(0, ZOOM_LIM)
    _style_ax(ax)
    if panel_idx == 0:
        ax.legend(handles=[
            mlines.Line2D([], [], color=C3['optA'], marker='o', ms=7, ls='None', label='Vth Comp'),
            mlines.Line2D([], [], color=C3['optC'], marker='D', ms=6, ls='None', label='No Comp'),
            mlines.Line2D([], [], color='black', ls='--', lw=0.8, alpha=0.5, label='y = x'),
            mlines.Line2D([], [], color=C3['true'], marker='*', ms=11, ls='None', label='True'),
        ], fontsize=14, loc='upper right', title="Ranking\nComparison", title_fontsize=16)

fig0.tight_layout()
out0 = f'{OUT_BASE}/[newq_margin_seed19]row0_scatter_newline_zoom60.png'
fig0.savefig(out0, dpi=300, bbox_inches='tight')
plt.close(fig0)
print(f"저장: {out0}")
