"""
newq_margin 폴더 전용 scatter/bar plot 생성
  q0: qid 329114 "how much to have your house reshingled"  margin=1.62  (경쟁 치열)
  q1: qid 498398 "simple definition wave amplitude"        margin=11.35 (중간)
  q2: qid 984178 "where is hartwell ga"                    margin=20.06 (압도적 1등)
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

FONT_BASE     = 13
FONT_AXIS     = 14
FONT_PANEL    = 15
GRID_ALPHA    = 0.3

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
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8, ls="--")
    ax.minorticks_off()

_setup_matplotlib()

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\07_newq_margin_scale\newq_margin_500_top100'
QUERY_RELEVANT = {0: 471602, 1: 822108, 2: 347885}
Q_LABELS = [
    'q0: "...house reshingled" (margin=1.62, tight race)',
    'q1: "wave amplitude def." (margin=11.35, medium)',
    'q2: "where is hartwell ga" (margin=20.06, dominant win)',
]
TOP_K = 30

COLORS = {
    'all3':      '#4CAF50',
    'dig_A':     '#90CAF9',
    'dig_C':     '#FFCC80',
    'dig_only':  '#78909C',
    'A_only':    '#2196F3',
    'C_only':    '#FF9800',
    'AC_only':   '#9C27B0',
    'true':      '#E91E63',
}

C3 = {
    'optA':   '#2196F3',
    'optC':   '#FF7811',
    'true':   '#E91E63',
    'dig':    '#4CAF50',
}

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("데이터 로드 중...")
dig_sheets  = pd.read_excel(f'{BASE}/[vth_v3_fixedcell]step3_digital_candidate_pids.xlsx', sheet_name=None)
optA_sheets = pd.read_excel(f'{BASE}/[vth_v3_fixedcell]step3_optA_candidate_pids.xlsx',   sheet_name=None)
optC_sheets = pd.read_excel(f'{BASE}/[vth_v3_fixedcell]step3_optC_candidate_pids.xlsx',   sheet_name=None)
step6       = pd.read_excel(f'{BASE}/[vth_v3_fixedcell]step6_all_results.xlsx',            sheet_name=None)

dig_rank, optA_rank, optC_rank = {}, {}, {}
for q_id in [0, 1, 2]:
    dig_rank[q_id]  = step6[f'q{q_id}_digital'].set_index('pid')
    optA_rank[q_id] = step6[f'q{q_id}_optA'].set_index('pid')
    optC_rank[q_id] = step6[f'q{q_id}_optC'].set_index('pid')

dig_pids_dict, optA_pids_dict, optC_pids_dict = {}, {}, {}
for q_id in [0, 1, 2]:
    dig_pids_dict[q_id]  = set(dig_sheets[f'q{q_id}']['pid'])
    optA_pids_dict[q_id] = set(optA_sheets[f'q{q_id}']['pid'])
    optC_pids_dict[q_id] = set(optC_sheets[f'q{q_id}']['pid'])

W = 0.28

def draw_stacked(ax, x, segments, total):
    b = 0
    for val, col in segments:
        if val > 0:
            ax.bar(x, val, bottom=b, color=col, width=W)
            txt_col = 'black' if col in (COLORS['dig_A'], COLORS['dig_C']) else 'white'
            ax.text(x, b + val/2, str(val), ha='center', va='center',
                    fontsize=8, color=txt_col, fontweight='bold')
            b += val
    ax.text(x, total + 1.5, str(total), ha='center', fontsize=10, fontweight='bold')

# ── Row 0: ranking scatter (3 queries) ──────────────────────────────────────────
print("Row 0 (scatter) 그리는 중...")
fig0, axes0 = plt.subplots(1, 3, figsize=(18, 5))

for panel_idx, q_id in enumerate([0, 1, 2]):
    ax       = axes0[panel_idx]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_rank[q_id]
    df_A     = optA_rank[q_id]
    df_C     = optC_rank[q_id]

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

    true_xs = []
    for df_m, rank_col, lbl, off in [(df_A, 'rank_optA', 'A', (+8, +4)),
                                      (df_C, 'rank_optC', 'C', (+8, -12))]:
        if true_pid in df_m.index and true_pid in df_dig.index:
            tx = int(df_dig.loc[true_pid, 'rank_dig_f32'])
            ty = int(df_m.loc[true_pid, rank_col])
            ax.scatter(tx, ty, c=C3['true'], s=260, marker='*', zorder=6, clip_on=False)
            true_xs.append(tx)

    max_rank = max(xA + yA + xC + yC + true_xs + [10])
    lim = max_rank * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.3)
    ax.set_xlabel('Digital rank', fontsize=18)
    ax.set_ylabel('Analog rank', fontsize=18)
    ax.tick_params(labelsize=17)
    ax.set_title(f'{Q_LABELS[q_id]}\ncommon w/ A: {len(common_A)}  |  w/ C: {len(common_C)}', fontsize=11)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    _style_ax(ax)
    if panel_idx == 0:
        ax.legend(handles=[
            mlines.Line2D([], [], color=C3['optA'], marker='o',  ms=7,  ls='None', label='Vth Comp'),
            mlines.Line2D([], [], color=C3['optC'], marker='D',  ms=6,  ls='None', label='No Comp'),
            mlines.Line2D([], [], color='black',    ls='--', lw=0.8, alpha=0.5, label='y = x'),
            mlines.Line2D([], [], color=C3['true'], marker='*',  ms=11, ls='None', label='True'),
        ], fontsize=12, loc='upper right',
           title="Ranking\nComparison", title_fontsize=13)

fig0.tight_layout()
out0 = f'{BASE}/[vth_v3_fixedcell]row0_scatter_newline.png'
fig0.savefig(out0, dpi=300, bbox_inches='tight')
plt.close(fig0)
print(f"  저장: {out0}")

# ── Row 1: candidate stacked bar ───────────────────────────────────────────────
print("Row 1 (bar) 그리는 중...")
fig1, axes1 = plt.subplots(1, 3, figsize=(10.2, 5))

for panel_idx, q_id in enumerate([0, 1, 2]):
    ax       = axes1[panel_idx]
    dig_pids = dig_pids_dict[q_id]
    A_pids   = optA_pids_dict[q_id]
    C_pids   = optC_pids_dict[q_id]

    s_all3     = dig_pids & A_pids & C_pids
    s_dig_A    = (dig_pids & A_pids) - C_pids
    s_dig_C    = (dig_pids & C_pids) - A_pids
    s_dig_only = dig_pids - A_pids - C_pids
    s_A_only   = A_pids - dig_pids - C_pids
    s_C_only   = C_pids - dig_pids - A_pids
    s_AC_only  = (A_pids & C_pids) - dig_pids

    jac_A = len(dig_pids & A_pids) / len(dig_pids | A_pids)
    jac_C = len(dig_pids & C_pids) / len(dig_pids | C_pids)

    draw_stacked(ax, 0, [(len(s_dig_only), COLORS['dig_only']), (len(s_dig_A), COLORS['dig_A']),
                          (len(s_dig_C),   COLORS['dig_C']),    (len(s_all3),  COLORS['all3'])], len(dig_pids))
    draw_stacked(ax, 1, [(len(s_A_only),   COLORS['A_only']),  (len(s_AC_only), COLORS['AC_only']),
                          (len(s_dig_A),   COLORS['dig_A']),    (len(s_all3),  COLORS['all3'])], len(A_pids))
    draw_stacked(ax, 2, [(len(s_C_only),   COLORS['C_only']),  (len(s_AC_only), COLORS['AC_only']),
                          (len(s_dig_C),   COLORS['dig_C']),    (len(s_all3),  COLORS['all3'])], len(C_pids))

    ax.text(0.5, -0.07,
            f'Jaccard(Dig↔A)={jac_A:.3f}   Jaccard(Dig↔C)={jac_C:.3f}',
            ha='center', transform=ax.transAxes, fontsize=7.5, color='dimgray', style='italic')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Digital', 'OptionA', 'OptionC'], fontsize=9, fontweight='bold')
    ax.set_ylabel('Candidate passages', fontsize=8)
    ax.set_title(Q_LABELS[q_id], fontsize=7, fontweight='bold')
    ax.set_ylim(0, max(len(dig_pids), len(A_pids), len(C_pids)) * 1.22)
    ax.grid(axis='y', ls='--', alpha=0.3)
    if panel_idx == 0:
        ax.legend(handles=[
            mpatches.Patch(color=COLORS['all3'],     label='dig ∩ A ∩ C'),
            mpatches.Patch(color=COLORS['dig_A'],    label='dig ∩ A \\ C'),
            mpatches.Patch(color=COLORS['dig_C'],    label='dig ∩ C \\ A'),
            mpatches.Patch(color=COLORS['dig_only'], label='dig only'),
            mpatches.Patch(color=COLORS['A_only'],   label='A only'),
            mpatches.Patch(color=COLORS['C_only'],   label='C only'),
            mpatches.Patch(color=COLORS['AC_only'],  label='A ∩ C \\ dig'),
        ], fontsize=7, loc='upper right')

fig1.tight_layout()
out1 = f'{BASE}/[vth_v3_fixedcell]row1_bar.png'
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  저장: {out1}")

print("완료!")
