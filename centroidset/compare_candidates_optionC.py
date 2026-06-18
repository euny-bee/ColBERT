"""
Digital vs OptionC 후보 집합 비교 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset'
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}

# -- 후보 집합 로드 ------------------------------------------------------------
dig_sheets  = pd.read_excel(f'{BASE}/step3_candidate_pids.xlsx',        sheet_name=None)
optC_sheets = pd.read_excel(f'{BASE}/step3_optionC_candidate_pids.xlsx', sheet_name=None)

data = {}
for q_id in [0, 1, 2]:
    q        = f'q{q_id}'
    dig_pids  = set(dig_sheets[q]['pid'].tolist())
    optC_pids = set(optC_sheets[q]['pid'].tolist())
    true_pid  = QUERY_RELEVANT[q_id]

    inter     = dig_pids & optC_pids
    dig_only  = dig_pids - optC_pids
    optC_only = optC_pids - dig_pids
    union     = dig_pids | optC_pids
    jaccard   = len(inter) / len(union)

    data[q_id] = {
        'dig_pids':       dig_pids,
        'optC_pids':      optC_pids,
        'inter':          inter,
        'dig_only':       dig_only,
        'optC_only':      optC_only,
        'jaccard':        jaccard,
        'true_pid':       true_pid,
        'true_in_inter':  true_pid in inter,
    }
    print(f"[q{q_id}]  digital={len(dig_pids)}  optionC={len(optC_pids)}  "
          f"교집합={len(inter)}  Jaccard={jaccard:.3f}  "
          f"정답 in 교집합: {true_pid in inter}")

# =============================================================================
# 시각화
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Digital vs OptionC Candidate Set Comparison  (nprobe=2)",
             fontsize=13, fontweight='bold', y=0.98)

COLORS = {
    'dig_only':  '#2196F3',   # 파랑 - digital만
    'inter':     '#4CAF50',   # 초록 - 공통
    'optC_only': '#FF9800',   # 주황 - optionC만
    'true':      '#E91E63',   # 분홍 - 정답
}
Q_TITLES = [
    'q0: "where does real insulin come from"',
    'q1: "where does name nora come from"',
    'q2: "where does most of the iron ore come from"',
]

# -- 상단: stacked bar --------------------------------------------------------
for q_id in [0, 1, 2]:
    ax = axes[0, q_id]
    d  = data[q_id]

    n_dig_only  = len(d['dig_only'])
    n_inter     = len(d['inter'])
    n_optC_only = len(d['optC_only'])

    # Digital 막대
    ax.bar(0, n_dig_only, color=COLORS['dig_only'], width=0.5)
    ax.bar(0, n_inter,    bottom=n_dig_only, color=COLORS['inter'], width=0.5)

    # OptionC 막대
    ax.bar(1, n_optC_only, color=COLORS['optC_only'], width=0.5)
    ax.bar(1, n_inter,     bottom=n_optC_only, color=COLORS['inter'], width=0.5)

    # 수치 표시
    for x, total, excl in [
        (0, len(d['dig_pids']),  n_dig_only),
        (1, len(d['optC_pids']), n_optC_only),
    ]:
        ax.text(x, total + 1.5, f'{total}', ha='center', fontsize=11, fontweight='bold')
        if excl > 0:
            ax.text(x, excl / 2, f'{excl}', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        ax.text(x, excl + n_inter / 2, f'{n_inter}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Digital', 'OptionC'], fontsize=10)
    ax.set_ylabel('Candidate passages', fontsize=9)
    ax.set_title(f'{Q_TITLES[q_id]}\nJaccard = {d["jaccard"]:.3f}', fontsize=8.5, fontweight='bold')
    ax.set_ylim(0, max(len(d['dig_pids']), len(d['optC_pids'])) * 1.18)
    ax.grid(axis='y', ls='--', alpha=0.3)

    if q_id == 0:
        legend_patches = [
            mpatches.Patch(color=COLORS['dig_only'],  label='Digital only'),
            mpatches.Patch(color=COLORS['inter'],      label='Common (intersection)'),
            mpatches.Patch(color=COLORS['optC_only'], label='OptionC only'),
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc='lower right')

# -- 하단: n_tokens scatter ---------------------------------------------------
for q_id in [0, 1, 2]:
    ax  = axes[1, q_id]
    d   = data[q_id]

    dig_df  = dig_sheets[f'q{q_id}'].set_index('pid')
    optC_df = optC_sheets[f'q{q_id}'].set_index('pid')

    # 공통 pid
    common_pids = list(d['inter'])
    if common_pids:
        dig_ntok  = dig_df.loc[common_pids,  'n_tokens'].values
        optC_ntok = optC_df.loc[common_pids, 'n_tokens'].values
        ax.scatter(dig_ntok, optC_ntok, alpha=0.55, s=30,
                   color=COLORS['inter'], label=f'Common ({len(common_pids)})')

    # digital only
    dig_only_pids = list(d['dig_only'])
    if dig_only_pids:
        ntok = dig_df.loc[dig_only_pids, 'n_tokens'].values
        ax.scatter(ntok, np.zeros(len(ntok)), alpha=0.5, s=25,
                   color=COLORS['dig_only'], marker='^',
                   label=f'Digital only ({len(dig_only_pids)})')

    # optionC only
    optC_only_pids = list(d['optC_only'])
    if optC_only_pids:
        ntok = optC_df.loc[optC_only_pids, 'n_tokens'].values
        ax.scatter(np.zeros(len(ntok)), ntok, alpha=0.5, s=25,
                   color=COLORS['optC_only'], marker='s',
                   label=f'OptionC only ({len(optC_only_pids)})')

    # 정답 pid
    true_pid = d['true_pid']
    if true_pid in d['inter']:
        tx = dig_df.loc[true_pid, 'n_tokens']
        ty = optC_df.loc[true_pid, 'n_tokens']
        ax.scatter(tx, ty, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid}')
        ax.annotate(f'True\n({tx},{ty})', (tx, ty), textcoords='offset points',
                    xytext=(6, 4), fontsize=7.5, color=COLORS['true'], fontweight='bold')
    elif true_pid in d['dig_pids']:
        tx = dig_df.loc[true_pid, 'n_tokens']
        ax.scatter(tx, 0, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid} (dig only)')
    elif true_pid in d['optC_pids']:
        ty = optC_df.loc[true_pid, 'n_tokens']
        ax.scatter(0, ty, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid} (optC only)')

    # 대각선
    mx = max(ax.get_xlim()[1], ax.get_ylim()[1], 33)
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3)

    ax.set_xlabel('Digital n_tokens', fontsize=9)
    ax.set_ylabel('OptionC n_tokens', fontsize=9)
    ax.set_title(f'q{q_id}: n_tokens of common candidates', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(ls='--', alpha=0.25)

plt.tight_layout()
out = f'{BASE}/compare_candidates_optionC.png'
plt.savefig(out, dpi=150)
plt.show()
print(f"\nSaved: {out}")
