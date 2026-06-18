"""
Digital vs Analog 후보 집합 비교 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

# -- font ------------------------------------------------------------------
_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}

# -- 후보 집합 로드 ---------------------------------------------------------
dig_sheets = pd.read_excel(f'{BASE}/step3_candidate_pids.xlsx',        sheet_name=None)
ana_sheets = pd.read_excel(f'{BASE}/step3_analog_candidate_pids.xlsx', sheet_name=None)

data = {}
for q_id in [0, 1, 2]:
    q = f'q{q_id}'
    dig_pids = set(dig_sheets[q]['pid'].tolist())
    ana_pids = set(ana_sheets[q]['pid'].tolist())
    true_pid = QUERY_RELEVANT[q_id]

    inter  = dig_pids & ana_pids
    dig_only = dig_pids - ana_pids
    ana_only = ana_pids - dig_pids

    data[q_id] = {
        'dig_pids':   dig_pids,
        'ana_pids':   ana_pids,
        'inter':      inter,
        'dig_only':   dig_only,
        'ana_only':   ana_only,
        'union':      dig_pids | ana_pids,
        'jaccard':    len(inter) / len(dig_pids | ana_pids),
        'true_pid':   true_pid,
        'true_in_dig': true_pid in dig_pids,
        'true_in_ana': true_pid in ana_pids,
        'true_in_inter': true_pid in inter,
    }
    print(f"[q{q_id}]  digital={len(dig_pids)}  analog={len(ana_pids)}  "
          f"교집합={len(inter)}  합집합={len(dig_pids|ana_pids)}  "
          f"Jaccard={len(inter)/len(dig_pids|ana_pids):.3f}  "
          f"정답 in 교집합: {true_pid in inter}")

# ==========================================================================
# 시각화
# ==========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Digital vs Analog 후보 집합 비교  (nprobe=2)",
             fontsize=13, fontweight='bold', y=0.98)

COLORS = {
    'dig_only': '#2196F3',   # 파랑 - digital만
    'inter':    '#4CAF50',   # 초록 - 공통
    'ana_only': '#FF9800',   # 주황 - analog만
    'true':     '#E91E63',   # 분홍 - 정답
}
Q_TITLES = [
    'q0: "where does real insulin come from"',
    'q1: "where does name nora come from"',
    'q2: "where does most of the iron ore come from"',
]

# -- 상단: 막대그래프 (구성 비교) -------------------------------------------
for q_id in [0, 1, 2]:
    ax = axes[0, q_id]
    d = data[q_id]

    n_dig_only = len(d['dig_only'])
    n_inter    = len(d['inter'])
    n_ana_only = len(d['ana_only'])

    # stacked bar: digital
    ax.bar(0, n_dig_only, color=COLORS['dig_only'], label='Digital만', width=0.5)
    ax.bar(0, n_inter,    bottom=n_dig_only, color=COLORS['inter'], label='공통', width=0.5)

    # stacked bar: analog
    ax.bar(1, n_ana_only, color=COLORS['ana_only'], label='Analog만', width=0.5)
    ax.bar(1, n_inter,    bottom=n_ana_only, color=COLORS['inter'], width=0.5)

    # 수치 표시
    for x, total, excl, label in [
        (0, len(d['dig_pids']), n_dig_only, 'Digital'),
        (1, len(d['ana_pids']), n_ana_only, 'Analog'),
    ]:
        ax.text(x, total + 1.5, f'{total}개', ha='center', fontsize=11, fontweight='bold')
        ax.text(x, excl/2, f'{excl}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax.text(x, excl + n_inter/2, f'{n_inter}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Digital', 'Analog'], fontsize=10)
    ax.set_ylabel('후보 passage 수', fontsize=9)
    ax.set_title(f'{Q_TITLES[q_id]}\nJaccard = {d["jaccard"]:.3f}', fontsize=8.5, fontweight='bold')
    ax.set_ylim(0, max(len(d['dig_pids']), len(d['ana_pids'])) * 1.18)
    ax.grid(axis='y', ls='--', alpha=0.3)

    if q_id == 0:
        legend_patches = [
            mpatches.Patch(color=COLORS['dig_only'], label=f'Digital만'),
            mpatches.Patch(color=COLORS['inter'],    label=f'공통 (교집합)'),
            mpatches.Patch(color=COLORS['ana_only'], label=f'Analog만'),
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc='lower right')

# -- 하단: n_tokens 분포 비교 -----------------------------------------------
for q_id in [0, 1, 2]:
    ax = axes[1, q_id]
    d  = data[q_id]

    dig_df = dig_sheets[f'q{q_id}'].set_index('pid')
    ana_df = ana_sheets[f'q{q_id}'].set_index('pid')

    # 공통 pid의 n_tokens 비교
    common_pids = list(d['inter'])
    if common_pids:
        dig_ntok = dig_df.loc[common_pids, 'n_tokens'].values
        ana_ntok = ana_df.loc[common_pids, 'n_tokens'].values
        ax.scatter(dig_ntok, ana_ntok, alpha=0.55, s=30,
                   color=COLORS['inter'], label=f'공통 ({len(common_pids)}개)')

    # digital only
    dig_only_pids = list(d['dig_only'])
    if dig_only_pids:
        ntok = dig_df.loc[dig_only_pids, 'n_tokens'].values
        ax.scatter(ntok, np.zeros(len(ntok)), alpha=0.5, s=25,
                   color=COLORS['dig_only'], marker='^', label=f'Digital만 ({len(dig_only_pids)}개)')

    # analog only
    ana_only_pids = list(d['ana_only'])
    if ana_only_pids:
        ntok = ana_df.loc[ana_only_pids, 'n_tokens'].values
        ax.scatter(np.zeros(len(ntok)), ntok, alpha=0.5, s=25,
                   color=COLORS['ana_only'], marker='s', label=f'Analog만 ({len(ana_only_pids)}개)')

    # 정답 pid 표시
    true_pid = d['true_pid']
    if true_pid in d['inter']:
        tx = dig_df.loc[true_pid, 'n_tokens']
        ty = ana_df.loc[true_pid, 'n_tokens']
        ax.scatter(tx, ty, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'정답 pid {true_pid}')
        ax.annotate(f'정답\n({tx},{ty})', (tx, ty), textcoords='offset points',
                    xytext=(6, 4), fontsize=7.5, color=COLORS['true'])

    # 대각선
    mx = max(ax.get_xlim()[1], ax.get_ylim()[1], 33)
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3)

    ax.set_xlabel('Digital n_tokens', fontsize=9)
    ax.set_ylabel('Analog n_tokens', fontsize=9)
    ax.set_title(f'q{q_id}: 공통 후보의 n_tokens 일치도', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(ls='--', alpha=0.25)

plt.tight_layout()
out = f'{BASE}/compare_candidates.png'
plt.savefig(out, dpi=150)
plt.show()
print(f"\nSaved: {out}")
