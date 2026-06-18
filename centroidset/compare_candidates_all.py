"""
Digital vs OptionA & OptionC — Candidate Set Comparison (combined)
Row 1: Combined bar  (Digital / OptionA / OptionC — 3 bars per query)
Row 2: n_tokens scatter  Digital vs OptionA
Row 3: n_tokens scatter  Digital vs OptionC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset'
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}
Q_TITLES = [
    'q0: "where does real insulin come from"',
    'q1: "where does name nora come from"',
    'q2: "where does most of the iron ore come from"',
]

COLORS = {
    'all3':      '#4CAF50',   # 진초록  — dig ∩ A ∩ C  (셋 다 공통)
    'dig_A':     '#90CAF9',   # 연파랑  — dig ∩ A \ C
    'dig_C':     '#FFCC80',   # 연주황  — dig ∩ C \ A
    'dig_only':  '#78909C',   # 회색    — dig \ A \ C
    'A_only':    '#2196F3',   # 진파랑  — A \ dig \ C
    'C_only':    '#FF9800',   # 진주황  — C \ dig \ A
    'AC_only':   '#9C27B0',   # 보라    — A ∩ C \ dig
    'true':      '#E91E63',   # 분홍    — 정답
}

# =============================================================================
# 데이터 로드 및 집합 계산
# =============================================================================
dig_sheets  = pd.read_excel(f'{BASE}/step3_candidate_pids.xlsx',         sheet_name=None)
optA_sheets = pd.read_excel(f'{BASE}/step3_analog_candidate_pids.xlsx',  sheet_name=None)
optC_sheets = pd.read_excel(f'{BASE}/step3_optionC_candidate_pids.xlsx', sheet_name=None)

dataA, dataC = {}, {}
for q_id in [0, 1, 2]:
    q        = f'q{q_id}'
    dig_pids = set(dig_sheets[q]['pid'])
    true_pid = QUERY_RELEVANT[q_id]

    A_pids = set(optA_sheets[q]['pid'])
    interA = dig_pids & A_pids
    dataA[q_id] = {'dig': dig_pids, 'other': A_pids, 'inter': interA,
                   'dig_only': dig_pids - A_pids, 'other_only': A_pids - dig_pids,
                   'jaccard': len(interA) / len(dig_pids | A_pids), 'true_pid': true_pid}

    C_pids = set(optC_sheets[q]['pid'])
    interC = dig_pids & C_pids
    dataC[q_id] = {'dig': dig_pids, 'other': C_pids, 'inter': interC,
                   'dig_only': dig_pids - C_pids, 'other_only': C_pids - dig_pids,
                   'jaccard': len(interC) / len(dig_pids | C_pids), 'true_pid': true_pid}

optA_dfs = {q: optA_sheets[f'q{q}'].set_index('pid') for q in [0,1,2]}
optC_dfs = {q: optC_sheets[f'q{q}'].set_index('pid') for q in [0,1,2]}
dig_dfs  = {q: dig_sheets[f'q{q}'].set_index('pid')  for q in [0,1,2]}

# =============================================================================
# 헬퍼: scatter
# =============================================================================
def draw_scatter(ax, data, dig_df, other_df, label_other, q_id, other_color):
    true_pid    = data['true_pid']
    common_pids = list(data['inter'])
    dig_only    = list(data['dig_only'])
    other_only  = list(data['other_only'])

    if common_pids:
        ax.scatter(dig_df.loc[common_pids, 'n_tokens'].values,
                   other_df.loc[common_pids, 'n_tokens'].values,
                   alpha=0.55, s=30, color=COLORS['all3'],
                   label=f'Common ({len(common_pids)})')
    if dig_only:
        ax.scatter(dig_df.loc[dig_only, 'n_tokens'].values,
                   np.zeros(len(dig_only)), alpha=0.5, s=25,
                   color=COLORS['dig_only'], marker='^',
                   label=f'Digital only ({len(dig_only)})')
    if other_only:
        ax.scatter(np.zeros(len(other_only)),
                   other_df.loc[other_only, 'n_tokens'].values,
                   alpha=0.5, s=25, color=other_color, marker='s',
                   label=f'{label_other} only ({len(other_only)})')

    if true_pid in data['inter']:
        tx = dig_df.loc[true_pid, 'n_tokens']
        ty = other_df.loc[true_pid, 'n_tokens']
        ax.scatter(tx, ty, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid}')
        ax.annotate(f'True\n({tx},{ty})', (tx, ty),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7.5, color=COLORS['true'], fontweight='bold')
    elif true_pid in data['dig']:
        tx = dig_df.loc[true_pid, 'n_tokens']
        ax.scatter(tx, 0, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid} (dig only)')
    elif true_pid in data['other']:
        ty = other_df.loc[true_pid, 'n_tokens']
        ax.scatter(0, ty, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid} ({label_other} only)')

    mx = max(ax.get_xlim()[1], ax.get_ylim()[1], 33)
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3)
    ax.set_xlabel('Digital n_tokens', fontsize=8)
    ax.set_ylabel(f'{label_other} n_tokens', fontsize=8)
    ax.set_title(f'q{q_id}: n_tokens of common candidates (Digital vs {label_other})',
                 fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(ls='--', alpha=0.25)

# =============================================================================
# 그래프 (3×3)
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle(
    "Digital vs OptionA & OptionC  —  Candidate Set Comparison  (nprobe=2)",
    fontsize=13, fontweight='bold', y=0.998
)

W = 0.28   # bar width
GAP = 0.08

for q_id in [0, 1, 2]:
    ax   = axes[0, q_id]
    dA   = dataA[q_id]
    dC   = dataC[q_id]

    dig_pids = dA['dig']
    A_pids   = dA['other']
    C_pids   = dC['other']

    # 7-way Venn 분할
    s_all3    = dig_pids & A_pids & C_pids          # dig ∩ A ∩ C
    s_dig_A   = (dig_pids & A_pids) - C_pids        # dig ∩ A \ C
    s_dig_C   = (dig_pids & C_pids) - A_pids        # dig ∩ C \ A
    s_dig_only= dig_pids - A_pids - C_pids          # dig only
    s_A_only  = A_pids - dig_pids - C_pids          # A only
    s_C_only  = C_pids - dig_pids - A_pids          # C only
    s_AC_only = (A_pids & C_pids) - dig_pids        # A ∩ C \ dig

    n_all3    = len(s_all3);  n_dig_A  = len(s_dig_A)
    n_dig_C   = len(s_dig_C); n_dig_only= len(s_dig_only)
    n_A_only  = len(s_A_only);n_C_only  = len(s_C_only)
    n_AC_only = len(s_AC_only)
    n_dig = len(dig_pids)

    xs = [0, 1, 2]

    def draw_stacked(ax, x, segments, total):
        """segments: list of (count, color). 아래부터 쌓음. 숫자 표시."""
        b = 0
        for val, col in segments:
            if val > 0:
                ax.bar(x, val, bottom=b, color=col, width=W)
                txt_col = 'black' if col in (COLORS['dig_A'], COLORS['dig_C']) else 'white'
                ax.text(x, b + val / 2, str(val), ha='center', va='center',
                        fontsize=8, color=txt_col, fontweight='bold')
                b += val
        ax.text(x, total + 1.5, str(total), ha='center', fontsize=10, fontweight='bold')

    # Digital bar: dig_only | dig_A | dig_C | all3  (bottom → top)
    draw_stacked(ax, xs[0], [
        (n_dig_only, COLORS['dig_only']),
        (n_dig_A,    COLORS['dig_A']),
        (n_dig_C,    COLORS['dig_C']),
        (n_all3,     COLORS['all3']),
    ], n_dig)

    # OptionA bar: A_only | AC_only | dig_A | all3
    draw_stacked(ax, xs[1], [
        (n_A_only,  COLORS['A_only']),
        (n_AC_only, COLORS['AC_only']),
        (n_dig_A,   COLORS['dig_A']),
        (n_all3,    COLORS['all3']),
    ], len(A_pids))

    # OptionC bar: C_only | AC_only | dig_C | all3
    draw_stacked(ax, xs[2], [
        (n_C_only,  COLORS['C_only']),
        (n_AC_only, COLORS['AC_only']),
        (n_dig_C,   COLORS['dig_C']),
        (n_all3,    COLORS['all3']),
    ], len(C_pids))

    ax.text(0.5, -0.07,
            f'Jaccard(Dig↔A)={dA["jaccard"]:.3f}   Jaccard(Dig↔C)={dC["jaccard"]:.3f}',
            ha='center', transform=ax.transAxes, fontsize=7.5, color='dimgray', style='italic')

    ax.set_xticks(xs)
    ax.set_xticklabels(['Digital', 'OptionA', 'OptionC'], fontsize=9, fontweight='bold')
    ax.set_ylabel('Candidate passages', fontsize=8)
    ax.set_title(Q_TITLES[q_id], fontsize=8.5, fontweight='bold')
    ax.set_ylim(0, max(n_dig, len(A_pids), len(C_pids)) * 1.22)
    ax.grid(axis='y', ls='--', alpha=0.3)

    if q_id == 0:
        legend_patches = [
            mpatches.Patch(color=COLORS['all3'],     label='dig ∩ A ∩ C  (all three)'),
            mpatches.Patch(color=COLORS['dig_A'],    label='dig ∩ A \\ C'),
            mpatches.Patch(color=COLORS['dig_C'],    label='dig ∩ C \\ A'),
            mpatches.Patch(color=COLORS['dig_only'], label='dig only'),
            mpatches.Patch(color=COLORS['A_only'],   label='A only'),
            mpatches.Patch(color=COLORS['C_only'],   label='C only'),
            mpatches.Patch(color=COLORS['AC_only'],  label='A ∩ C \\ dig'),
        ]
        ax.legend(handles=legend_patches, fontsize=7, loc='upper right')

    # Row 1: scatter Digital vs OptionA
    draw_scatter(axes[1, q_id], dA, dig_dfs[q_id], optA_dfs[q_id],
                 'OptionA', q_id, COLORS['A_only'])

    # Row 2: scatter Digital vs OptionC
    draw_scatter(axes[2, q_id], dC, dig_dfs[q_id], optC_dfs[q_id],
                 'OptionC', q_id, COLORS['C_only'])

plt.tight_layout()
out = f'{BASE}/compare_candidates_all.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out}")
