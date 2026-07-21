"""
[vth_v3]compare_ranking_all3.png 두번째 줄(bump chart)만 별도 파일로 저장.
패널 순서: q1, q0, q2
출력: [vth_v3]row_bump.png
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\03_vth_v3'
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}
Q_LABELS = [
    'q0: "where does real insulin come from"',
    'q1: "where does name nora come from"',
    'q2: "where does most of the iron ore come from"',
]
TOP_K = 25

C3 = {
    'optA':   '#2196F3',
    'optC':   '#FF9800',
    'true':   '#E91E63',
    'dig':    '#4CAF50',
}

X_POS = {'dig': 0, 'optA': 1, 'optC': 2}

FONT_BASE   = 13
FONT_AXIS   = 14
FONT_PANEL  = 15
GRID_ALPHA  = 0.3

def _setup_matplotlib():
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    sans,
        "font.size":          FONT_BASE,
        "font.weight":        "bold",
        "axes.labelsize":     FONT_AXIS,
        "axes.labelweight":   "bold",
        "axes.titlesize":     FONT_PANEL,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    FONT_BASE,
        "ytick.labelsize":    FONT_BASE,
        "axes.unicode_minus": False,
    })

def _style_ax(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.grid(True, axis='y', which="major", alpha=GRID_ALPHA, linewidth=0.8, ls="--")
    ax.minorticks_off()

_setup_matplotlib()

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("데이터 로드 중...")
step6 = pd.read_excel(f'{BASE}/[vth_v3]step6_all_results.xlsx', sheet_name=None)

dig_rank, optA_rank, optC_rank = {}, {}, {}
for q_id in [0, 1, 2]:
    dig_rank[q_id]  = step6[f'q{q_id}_digital'].set_index('pid')
    optA_rank[q_id] = step6[f'q{q_id}_optA'].set_index('pid')
    optC_rank[q_id] = step6[f'q{q_id}_optC'].set_index('pid')

# ── Bump chart (1×3, 순서: q1, q0, q2) ────────────────────────────────────────
print("Bump chart 그리는 중...")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 5))

for panel_idx, q_id in enumerate([1, 0, 2]):
    ax       = axes[panel_idx]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_rank[q_id].reset_index()
    df_A     = optA_rank[q_id].reset_index()
    df_C     = optC_rank[q_id].reset_index()

    dig_topk  = df_dig.nsmallest(TOP_K, 'rank_dig_f32')['pid'].tolist()
    optA_topk = df_A.nsmallest(TOP_K,   'rank_optA')['pid'].tolist()
    optC_topk = df_C.nsmallest(TOP_K,   'rank_optC')['pid'].tolist()
    all_pids  = list(dict.fromkeys(dig_topk + optA_topk + optC_topk))

    dig_rd  = dict(zip(df_dig['pid'], df_dig['rank_dig_f32']))
    optA_rd = dict(zip(df_A['pid'],   df_A['rank_optA']))
    optC_rd = dict(zip(df_C['pid'],   df_C['rank_optC']))

    for pid in all_pids:
        dr = dig_rd.get(pid); ar = optA_rd.get(pid); cr = optC_rd.get(pid)
        in_dig  = dr is not None and dr <= TOP_K
        in_A    = ar is not None and ar <= TOP_K
        in_C    = cr is not None and cr <= TOP_K
        is_true = pid == true_pid
        n_m     = sum([in_dig, in_A, in_C])

        col  = C3['true'] if is_true else (
               C3['dig']  if n_m >= 2 else
               C3['optA'] if in_A and not in_dig and not in_C else
               C3['optC'] if in_C and not in_dig and not in_A else C3['dig'])
        lw   = 2.5 if is_true else 0.7
        alph = 1.0 if is_true else 0.4

        pts = []
        if in_dig: pts.append((X_POS['dig'],  dr))
        if in_A:   pts.append((X_POS['optA'], ar))
        if in_C:   pts.append((X_POS['optC'], cr))

        for k in range(len(pts) - 1):
            ax.plot([pts[k][0], pts[k+1][0]], [pts[k][1], pts[k+1][1]],
                    color=col, lw=lw, alpha=alph, zorder=2)
        for xp, yp in pts:
            ax.scatter(xp, yp, color=col, s=80 if is_true else 35,
                       marker='*' if is_true else 'o', zorder=4, alpha=max(alph, 0.7))


    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(TOP_K + 1, 0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Digital', 'Vth Comp', 'No Comp'], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel(f'Rank (top-{TOP_K})')
    ax.set_title(f'{Q_LABELS[q_id]}\nTop-{TOP_K} rank comparison', fontsize=9)
    _style_ax(ax)

fig.tight_layout()
out = f'{BASE}/[vth_v3]row_bump.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  저장: {out}")

# ── Legend-only figure ────────────────────────────────────────────────────────
legend_handles = [
    mlines.Line2D([], [], color=C3['dig'],  lw=1.5, label='Digital & overlap'),
    mlines.Line2D([], [], color=C3['optA'], lw=1.5, label='Vth comp only'),
    mlines.Line2D([], [], color=C3['optC'], lw=1.5, label='No comp only'),
    mlines.Line2D([], [], color=C3['true'], lw=2.5, label='True (qrel)'),
]
fig_leg, ax_leg = plt.subplots(figsize=(2.6, 1.5))
ax_leg.axis('off')
ax_leg.legend(handles=legend_handles, fontsize=FONT_BASE - 2, loc='center', frameon=True)
fig_leg.tight_layout()
out_leg = f'{BASE}/[vth_v3]row_bump_legend.png'
fig_leg.savefig(out_leg, dpi=300, bbox_inches='tight')
plt.close(fig_leg)
print(f"  저장: {out_leg}")
print("완료!")
