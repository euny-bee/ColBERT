"""
2×3 combined figure:
  Row 0: compare_ranking_all3 첫째 줄 — Digital L2 vs Option A/C rank scatter
  Row 1: compare_candidates_all 첫째 줄 — candidate set stacked bar
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\03_vth_v3'
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}
Q_LABELS = [
    'q0: "where does real insulin come from"',
    'q1: "where does name nora come from"',
    'q2: "where does most of the iron ore come from"',
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
    'common':    '#4CAF50',
    'optA_only': '#2196F3',
    'optC_only': '#FF9800',
}

C3 = {
    'optA':   '#2196F3',
    'optC':   '#FF9800',
    'true':   '#E91E63',
    'dig':    '#4CAF50',
}

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("데이터 로드 중...")
dig_sheets  = pd.read_excel(f'{BASE}/[vth_v3]step3_digital_candidate_pids.xlsx', sheet_name=None)
optA_sheets = pd.read_excel(f'{BASE}/[vth_v3]step3_optA_candidate_pids.xlsx',   sheet_name=None)
optC_sheets = pd.read_excel(f'{BASE}/[vth_v3]step3_optC_candidate_pids.xlsx',   sheet_name=None)
step6       = pd.read_excel(f'{BASE}/[vth_v3]step6_all_results.xlsx',            sheet_name=None)

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

# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
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

# ── 2×3 figure ────────────────────────────────────────────────────────────────
print("그리는 중...")
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# ── Row 0: ranking scatter (compare_ranking_all3 첫째 줄) ─────────────────────
for q_id in [0, 1, 2]:
    ax       = axes[0, q_id]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_rank[q_id]
    df_A     = optA_rank[q_id]
    df_C     = optC_rank[q_id]

    common_A = [p for p in df_dig.index if p in df_A.index]
    xA = [int(df_dig.loc[p, 'rank_dig_f32']) for p in common_A]
    yA = [int(df_A.loc[p, 'rank_optA'])      for p in common_A]
    cA = [C3['true'] if p == true_pid else C3['optA'] for p in common_A]
    sA = [200 if p == true_pid else 20 for p in common_A]
    ax.scatter(xA, yA, c=cA, s=sA, alpha=0.65, zorder=3, label='Option A')

    common_C = [p for p in df_dig.index if p in df_C.index]
    xC = [int(df_dig.loc[p, 'rank_dig_f32']) for p in common_C]
    yC = [int(df_C.loc[p, 'rank_optC'])      for p in common_C]
    cC = [C3['true'] if p == true_pid else C3['optC'] for p in common_C]
    sC = [200 if p == true_pid else 20 for p in common_C]
    ax.scatter(xC, yC, c=cC, s=sC, alpha=0.65, zorder=3, marker='D', label='Option C')

    for df_m, rank_col, lbl, off in [(df_A, 'rank_optA', 'A', (+8, +4)),
                                      (df_C, 'rank_optC', 'C', (+8, -12))]:
        if true_pid in df_m.index and true_pid in df_dig.index:
            tx = int(df_dig.loc[true_pid, 'rank_dig_f32'])
            ty = int(df_m.loc[true_pid, rank_col])
            ax.scatter(tx, ty, c=C3['true'], s=260, marker='*', zorder=6)
            ax.annotate(f'True ({lbl}): dig={tx}, rank={ty}', (tx, ty),
                        textcoords='offset points', xytext=off,
                        fontsize=7, color=C3['true'], fontweight='bold')

    mx = max(max(xA + xC, default=50), max(yA + yC, default=50)) + 5
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3)
    ax.set_xlabel('Digital L2 rank', fontsize=9)
    ax.set_ylabel('Method rank', fontsize=9)
    ax.set_title(f'{Q_LABELS[q_id]}\ncommon w/ A: {len(common_A)}  |  w/ C: {len(common_C)}',
                 fontsize=8, fontweight='bold')
    ax.grid(ls='--', alpha=0.25)
    ax.legend(handles=[
        mlines.Line2D([], [], color=C3['optA'], marker='o',  ms=6, ls='None', label='Option A rank'),
        mlines.Line2D([], [], color=C3['optC'], marker='D',  ms=5, ls='None', label='Option C rank'),
        mlines.Line2D([], [], color='black', ls='--', lw=0.8, alpha=0.5, label='y = x'),
        mlines.Line2D([], [], color=C3['true'], marker='*',  ms=10, ls='None', label=f'True ({true_pid})'),
    ], fontsize=7.5, loc='upper left')

# ── Row 1: candidate stacked bar (compare_candidates_all 첫째 줄) ─────────────
for q_id in [0, 1, 2]:
    ax       = axes[1, q_id]
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
                          (len(s_dig_C), COLORS['dig_C']),      (len(s_all3),  COLORS['all3'])], len(dig_pids))
    draw_stacked(ax, 1, [(len(s_A_only),  COLORS['A_only']),   (len(s_AC_only), COLORS['AC_only']),
                          (len(s_dig_A),   COLORS['dig_A']),    (len(s_all3),  COLORS['all3'])], len(A_pids))
    draw_stacked(ax, 2, [(len(s_C_only),  COLORS['C_only']),   (len(s_AC_only), COLORS['AC_only']),
                          (len(s_dig_C),   COLORS['dig_C']),    (len(s_all3),  COLORS['all3'])], len(C_pids))

    ax.text(0.5, -0.07,
            f'Jaccard(Dig↔A)={jac_A:.3f}   Jaccard(Dig↔C)={jac_C:.3f}',
            ha='center', transform=ax.transAxes, fontsize=7.5, color='dimgray', style='italic')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Digital', 'OptionA', 'OptionC'], fontsize=9, fontweight='bold')
    ax.set_ylabel('Candidate passages', fontsize=8)
    ax.set_title(Q_LABELS[q_id], fontsize=8.5, fontweight='bold')
    ax.set_ylim(0, max(len(dig_pids), len(A_pids), len(C_pids)) * 1.22)
    ax.grid(axis='y', ls='--', alpha=0.3)
    if q_id == 0:
        ax.legend(handles=[
            mpatches.Patch(color=COLORS['all3'],     label='dig ∩ A ∩ C'),
            mpatches.Patch(color=COLORS['dig_A'],    label='dig ∩ A \\ C'),
            mpatches.Patch(color=COLORS['dig_C'],    label='dig ∩ C \\ A'),
            mpatches.Patch(color=COLORS['dig_only'], label='dig only'),
            mpatches.Patch(color=COLORS['A_only'],   label='A only'),
            mpatches.Patch(color=COLORS['C_only'],   label='C only'),
            mpatches.Patch(color=COLORS['AC_only'],  label='A ∩ C \\ dig'),
        ], fontsize=7, loc='upper right')

plt.tight_layout()
out = f'{BASE}/[vth_v3]combined_2x3.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"  저장: {out}")
plt.close()
print("완료!")
