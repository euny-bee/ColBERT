"""
Digital L2 vs OptionC Ranking Comparison
방향 1 (상단): Rank scatter  — 공통 후보들의 두 방법 rank 비교
방향 2 (하단): Top-k bump chart — 상위 k개 문서 ranking 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset'
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}
Q_LABELS       = [
    'q0: "where does real insulin come from"',
    'q1: "where does name nora come from"',
    'q2: "where does most of the iron ore come from"',
]
TOP_K = 30   # bump chart에서 보여줄 상위 k개

COLORS = {
    'common':   '#4CAF50',
    'dig_only': '#2196F3',
    'optC_only':'#FF9800',
    'true':     '#E91E63',
    'line':     '#AAAAAA',
}

# =============================================================================
# 데이터 로드
# =============================================================================
print("데이터 로드 중...")
Q_raw = pd.read_excel(f'{BASE}/query_embs_96x128.xlsx',     header=0, index_col=0)
doc   = pd.read_excel(f'{BASE}/doc_embs_12919x128.xlsx',    header=0, index_col=0)
dig_sheets  = pd.read_excel(f'{BASE}/step3_candidate_pids.xlsx',         sheet_name=None)
optC_sheets = pd.read_excel(f'{BASE}/step3_optionC_candidate_pids.xlsx', sheet_name=None)
optC_step6  = pd.read_excel(f'{BASE}/step6_optionC_results.xlsx',        sheet_name=None)

Q             = Q_raw.values.astype(float)
dim_cols      = [c for c in doc.columns if c.startswith('dim_')]
D_f32         = doc[dim_cols].values.astype(float)
doc_token_idx = {tok: i for i, tok in enumerate(doc.index)}


# =============================================================================
# Digital L2 score 계산
# =============================================================================
def digital_minL2(Q_q, D_pid):
    """Σᵢ minⱼ ||Qᵢ - Dⱼ||₂  (낮을수록 유사)"""
    diffs = np.sqrt(((Q_q[:, None, :] - D_pid[None, :, :])**2).sum(axis=2))
    return float(diffs.min(axis=1).sum())


print("Digital L2 score 계산 중...")
dig_L2 = {}
for q_id in [0, 1, 2]:
    Q_q      = Q[q_id * 32:(q_id + 1) * 32]
    true_pid = QUERY_RELEVANT[q_id]
    rows = []
    for pid in dig_sheets[f'q{q_id}']['pid'].tolist():
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks:
            continue
        idxs = [doc_token_idx[k] for k in toks]
        rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                     'score_L2': digital_minL2(Q_q, D_f32[idxs])})
    df = pd.DataFrame(rows)
    df['rank_L2'] = df['score_L2'].rank(ascending=True).astype(int)
    dig_L2[q_id] = df.set_index('pid')
    rel = df[df['is_relevant']]
    if len(rel):
        r = rel.iloc[0]
        print(f"  [q{q_id}] Digital L2 정답 rank: {int(r['rank_L2'])}/{len(df)}")


# OptionC rank 로드 (step6 결과)
optC_rank = {}
for q_id in [0, 1, 2]:
    df = optC_step6[f'q{q_id}_optionC'].set_index('pid')
    optC_rank[q_id] = df
    rel = df[df['is_relevant']]
    if len(rel):
        r = rel.iloc[0]
        print(f"  [q{q_id}] OptionC   정답 rank: {int(r['rank_optC_f32'])}/{len(df)}")


# =============================================================================
# 그래프
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 12))
fig.suptitle(
    "Digital L2  vs  OptionC Analog Current  —  Ranking Comparison\n"
    "(doc vector: float32,  OptionC: Vth ~ TruncGaussian[0,0.5]V, seed=42)",
    fontsize=12, fontweight='bold', y=0.99
)

# ─────────────────────────────────────────────────────────────────────────────
# 방향 1 (상단): Rank scatter
# ─────────────────────────────────────────────────────────────────────────────
for q_id in [0, 1, 2]:
    ax       = axes[0, q_id]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_L2[q_id]
    df_optC  = optC_rank[q_id]

    # 공통 pids (Digital ⊂ OptionC 이므로 digital pids 전체)
    common_pids = [p for p in df_dig.index if p in df_optC.index]

    x_ranks, y_ranks, colors, sizes = [], [], [], []
    for pid in common_pids:
        xr = int(df_dig.loc[pid, 'rank_L2'])
        yr = int(df_optC.loc[pid, 'rank_optC_f32'])
        x_ranks.append(xr)
        y_ranks.append(yr)
        if pid == true_pid:
            colors.append(COLORS['true'])
            sizes.append(220)
        else:
            colors.append(COLORS['common'])
            sizes.append(25)

    ax.scatter(x_ranks, y_ranks, c=colors, s=sizes, alpha=0.6, zorder=3)

    # 정답 별표 + 주석
    if true_pid in df_dig.index and true_pid in df_optC.index:
        tx = int(df_dig.loc[true_pid, 'rank_L2'])
        ty = int(df_optC.loc[true_pid, 'rank_optC_f32'])
        ax.scatter(tx, ty, c=COLORS['true'], s=250, marker='*', zorder=5)
        ax.annotate(f'True\n(L2:{tx}, optC:{ty})', (tx, ty),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=7.5, color=COLORS['true'], fontweight='bold')

    # 대각선 (y=x, 완벽 일치선)
    mx = max(max(x_ranks), max(y_ranks)) + 5
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3, label='y = x (perfect match)')

    ax.set_xlabel('Digital L2 rank  (lower = more similar)', fontsize=9)
    ax.set_ylabel('OptionC Current rank  (lower = more similar)', fontsize=9)
    ax.set_title(f'{Q_LABELS[q_id]}\n'
                 f'common candidates: {len(common_pids)}',
                 fontsize=8.5, fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(ls='--', alpha=0.25)

    # 범례
    h_common = mlines.Line2D([], [], color=COLORS['common'], marker='o',
                              linestyle='None', ms=6, label='Common candidate')
    h_true   = mlines.Line2D([], [], color=COLORS['true'], marker='*',
                              linestyle='None', ms=10, label=f'True pid ({true_pid})')
    ax.legend(handles=[h_common, h_true], fontsize=8, loc='upper left')

# ─────────────────────────────────────────────────────────────────────────────
# 방향 2 (하단): Top-k bump chart
# ─────────────────────────────────────────────────────────────────────────────
for q_id in [0, 1, 2]:
    ax       = axes[1, q_id]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_L2[q_id].reset_index()
    df_optC  = optC_rank[q_id].reset_index()

    # Top-k pid 목록
    dig_topk  = df_dig.nsmallest(TOP_K, 'rank_L2')['pid'].tolist()
    optC_topk = df_optC.nsmallest(TOP_K, 'rank_optC_f32')['pid'].tolist()
    all_pids  = list(dict.fromkeys(dig_topk + optC_topk))  # 순서 유지, 중복 제거

    dig_rank_dict  = dict(zip(df_dig['pid'],  df_dig['rank_L2']))
    optC_rank_dict = dict(zip(df_optC['pid'], df_optC['rank_optC_f32']))

    for pid in all_pids:
        dr = dig_rank_dict.get(pid)
        or_ = optC_rank_dict.get(pid)

        is_true   = (pid == true_pid)
        in_both   = (dr is not None and dr <= TOP_K) and (or_ is not None and or_ <= TOP_K)
        dig_only  = (dr is not None and dr <= TOP_K) and (or_ is None or or_ > TOP_K)
        optC_only = (or_ is not None and or_ <= TOP_K) and (dr is None or dr > TOP_K)

        col  = COLORS['true'] if is_true else (
               COLORS['common']   if in_both   else
               COLORS['dig_only'] if dig_only  else COLORS['optC_only'])
        lw   = 2.5 if is_true else 0.8
        alph = 1.0 if is_true else 0.45

        # 연결선
        pts = []
        if dr is not None and dr <= TOP_K:
            pts.append((0, dr))
        if or_ is not None and or_ <= TOP_K:
            pts.append((1, or_))

        if len(pts) == 2:
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                    color=col, lw=lw, alpha=alph, zorder=2)

        # 점
        if dr is not None and dr <= TOP_K:
            ax.scatter(0, dr, color=col, s=60 if is_true else 20,
                       marker='*' if is_true else 'o', zorder=4, alpha=alph)
        if or_ is not None and or_ <= TOP_K:
            ax.scatter(1, or_, color=col, s=60 if is_true else 20,
                       marker='*' if is_true else 'o', zorder=4, alpha=alph)

        # 정답 주석
        if is_true:
            if dr is not None and dr <= TOP_K:
                ax.annotate(f'True (rank {dr})', (0, dr),
                            textcoords='offset points', xytext=(-65, 0),
                            fontsize=7.5, color=COLORS['true'], fontweight='bold',
                            ha='left')
            if or_ is not None and or_ <= TOP_K:
                ax.annotate(f'True (rank {or_})', (1, or_),
                            textcoords='offset points', xytext=(6, 0),
                            fontsize=7.5, color=COLORS['true'], fontweight='bold')

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(TOP_K + 1, 0)   # rank 1이 위
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Digital L2', 'OptionC Current'], fontsize=10, fontweight='bold')
    ax.set_ylabel(f'Rank (top-{TOP_K} shown)', fontsize=9)
    ax.set_title(f'{Q_LABELS[q_id]}\nTop-{TOP_K} rank comparison', fontsize=8.5, fontweight='bold')
    ax.grid(axis='y', ls='--', alpha=0.2)

    # 범례
    h_both  = mlines.Line2D([], [], color=COLORS['common'],   lw=2, label='In both top-k')
    h_dig   = mlines.Line2D([], [], color=COLORS['dig_only'], lw=2, label='Digital only')
    h_optC  = mlines.Line2D([], [], color=COLORS['optC_only'],lw=2, label='OptionC only')
    h_true  = mlines.Line2D([], [], color=COLORS['true'],     lw=2.5, label='True document')
    ax.legend(handles=[h_both, h_dig, h_optC, h_true], fontsize=7.5, loc='lower right')

plt.tight_layout()
out = f'{BASE}/compare_ranking_L2_optC.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out}")
