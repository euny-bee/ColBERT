"""
Digital L2 vs Option A vs Option C — 3-way Ranking Comparison
방향 1 (상단): Rank scatter  — Digital L2 rank(x) vs OptionA/OptionC rank(y) 동시 표시
방향 2 (하단): Bump chart   — Digital L2 / Option A / Option C 3열 비교
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
TOP_K = 30

COLORS = {
    'optA':     '#2196F3',   # 파랑
    'optC':     '#FF9800',   # 주황
    'true_A':   '#1565C0',   # 진파랑 (정답 in optA)
    'true_C':   '#E65100',   # 진주황 (정답 in optC)
    'true':     '#E91E63',   # 분홍
    'dig':      '#4CAF50',   # 초록
    'line_A':   '#90CAF9',
    'line_C':   '#FFCC80',
    'line_both':'#4CAF50',
}

# =============================================================================
# 데이터 로드
# =============================================================================
print("데이터 로드 중...")
Q_raw      = pd.read_excel(f'{BASE}/query_embs_96x128.xlsx',     header=0, index_col=0)
doc        = pd.read_excel(f'{BASE}/doc_embs_12919x128.xlsx',    header=0, index_col=0)
dig_sheets = pd.read_excel(f'{BASE}/step3_candidate_pids.xlsx',          sheet_name=None)
optA_step6 = pd.read_excel(f'{BASE}/step6_all_results.xlsx',             sheet_name=None)
optC_step6 = pd.read_excel(f'{BASE}/step6_optionC_results.xlsx',         sheet_name=None)

Q             = Q_raw.values.astype(float)
dim_cols      = [c for c in doc.columns if c.startswith('dim_')]
D_f32         = doc[dim_cols].values.astype(float)
doc_token_idx = {tok: i for i, tok in enumerate(doc.index)}

# =============================================================================
# Digital L2 score 계산
# =============================================================================
def digital_minL2(Q_q, D_pid):
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

# Option A / C rank 로드
optA_rank, optC_rank = {}, {}
for q_id in [0, 1, 2]:
    optA_rank[q_id] = optA_step6[f'q{q_id}_analog'].set_index('pid')
    optC_rank[q_id] = optC_step6[f'q{q_id}_optionC'].set_index('pid')

# 결과 출력
for q_id in [0, 1, 2]:
    true_pid = QUERY_RELEVANT[q_id]
    r_dig  = int(dig_L2[q_id].loc[true_pid, 'rank_L2'])           if true_pid in dig_L2[q_id].index  else 'N/A'
    r_optA = int(optA_rank[q_id].loc[true_pid, 'rank_ana_f32'])   if true_pid in optA_rank[q_id].index else 'N/A'
    r_optC = int(optC_rank[q_id].loc[true_pid, 'rank_optC_f32'])  if true_pid in optC_rank[q_id].index else 'N/A'
    n_dig  = len(dig_L2[q_id]);  n_A = len(optA_rank[q_id]);  n_C = len(optC_rank[q_id])
    print(f"  [q{q_id}]  Digital L2: {r_dig}/{n_dig}  |  OptionA: {r_optA}/{n_A}  |  OptionC: {r_optC}/{n_C}")

# =============================================================================
# 그래프
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(17, 12))
fig.suptitle(
    "Digital L2  vs  Option A  vs  Option C  —  3-way Ranking Comparison\n"
    "(doc vector: float32  |  OptionA: fixed Vth=0.151V  |  OptionC: Vth~TruncGauss[0,0.5]V, seed=42)",
    fontsize=11, fontweight='bold', y=0.99
)

# ─────────────────────────────────────────────────────────────────────────────
# 방향 1 (상단): Rank scatter (Digital L2 x, OptionA/C y)
# ─────────────────────────────────────────────────────────────────────────────
for q_id in [0, 1, 2]:
    ax       = axes[0, q_id]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_L2[q_id]
    df_A     = optA_rank[q_id]
    df_C     = optC_rank[q_id]

    # Option A 점
    common_A = [p for p in df_dig.index if p in df_A.index]
    xA = [int(df_dig.loc[p, 'rank_L2'])         for p in common_A]
    yA = [int(df_A.loc[p,   'rank_ana_f32'])     for p in common_A]
    cA = [COLORS['true'] if p == true_pid else COLORS['optA'] for p in common_A]
    sA = [200 if p == true_pid else 20 for p in common_A]
    ax.scatter(xA, yA, c=cA, s=sA, alpha=0.65, zorder=3, label='Option A')

    # Option C 점
    common_C = [p for p in df_dig.index if p in df_C.index]
    xC = [int(df_dig.loc[p, 'rank_L2'])         for p in common_C]
    yC = [int(df_C.loc[p,   'rank_optC_f32'])   for p in common_C]
    cC = [COLORS['true'] if p == true_pid else COLORS['optC'] for p in common_C]
    sC = [200 if p == true_pid else 20 for p in common_C]
    ax.scatter(xC, yC, c=cC, s=sC, alpha=0.65, zorder=3,
               marker='D', label='Option C')

    # 정답 별표 + 주석
    for df_m, rank_col, label, offset in [
        (df_A, 'rank_ana_f32',  'A', (+8, +4)),
        (df_C, 'rank_optC_f32', 'C', (+8, -12)),
    ]:
        if true_pid in df_m.index:
            tx = int(df_dig.loc[true_pid, 'rank_L2']) if true_pid in df_dig.index else None
            ty = int(df_m.loc[true_pid, rank_col])
            if tx:
                ax.scatter(tx, ty, c=COLORS['true'], s=260, marker='*', zorder=6)
                ax.annotate(f'True ({label}): L2={tx}, rank={ty}', (tx, ty),
                            textcoords='offset points', xytext=offset,
                            fontsize=7, color=COLORS['true'], fontweight='bold')

    # 대각선
    all_y = yA + yC
    mx = max(max(xA + xC), max(all_y)) + 5 if all_y else 50
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3)

    ax.set_xlabel('Digital L2 rank', fontsize=9)
    ax.set_ylabel('Method rank  (lower = more similar)', fontsize=9)
    ax.set_title(f'{Q_LABELS[q_id]}\n'
                 f'common w/ OptionA: {len(common_A)}  |  w/ OptionC: {len(common_C)}',
                 fontsize=8, fontweight='bold')
    ax.grid(ls='--', alpha=0.25)

    h_A    = mlines.Line2D([], [], color=COLORS['optA'], marker='o',  ms=6, ls='None', label='Option A rank')
    h_C    = mlines.Line2D([], [], color=COLORS['optC'], marker='D',  ms=5, ls='None', label='Option C rank')
    h_diag = mlines.Line2D([], [], color='black', ls='--', lw=0.8, alpha=0.5, label='y = x (perfect match)')
    h_true = mlines.Line2D([], [], color=COLORS['true'], marker='*',  ms=10, ls='None', label=f'True pid ({true_pid})')
    ax.legend(handles=[h_A, h_C, h_diag, h_true], fontsize=7.5, loc='upper left')

# ─────────────────────────────────────────────────────────────────────────────
# 방향 2 (하단): Bump chart (3열: Digital L2 / Option A / Option C)
# ─────────────────────────────────────────────────────────────────────────────
X_POS = {'dig': 0, 'optA': 1, 'optC': 2}

for q_id in [0, 1, 2]:
    ax       = axes[1, q_id]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_L2[q_id].reset_index()
    df_A     = optA_rank[q_id].reset_index()
    df_C     = optC_rank[q_id].reset_index()

    dig_topk  = df_dig.nsmallest(TOP_K, 'rank_L2')['pid'].tolist()
    optA_topk = df_A.nsmallest(TOP_K,   'rank_ana_f32')['pid'].tolist()
    optC_topk = df_C.nsmallest(TOP_K,   'rank_optC_f32')['pid'].tolist()
    all_pids  = list(dict.fromkeys(dig_topk + optA_topk + optC_topk))

    dig_rd  = dict(zip(df_dig['pid'], df_dig['rank_L2']))
    optA_rd = dict(zip(df_A['pid'],   df_A['rank_ana_f32']))
    optC_rd = dict(zip(df_C['pid'],   df_C['rank_optC_f32']))

    for pid in all_pids:
        dr = dig_rd.get(pid);   ar = optA_rd.get(pid);   cr = optC_rd.get(pid)
        in_dig  = dr is not None and dr <= TOP_K
        in_A    = ar is not None and ar <= TOP_K
        in_C    = cr is not None and cr <= TOP_K
        is_true = pid == true_pid

        n_methods = sum([in_dig, in_A, in_C])
        col  = COLORS['true'] if is_true else (
               COLORS['dig']  if n_methods >= 2 else
               COLORS['optA'] if in_A and not in_dig and not in_C else
               COLORS['optC'] if in_C and not in_dig and not in_A else
               COLORS['dig'])
        lw   = 2.5 if is_true else 0.7
        alph = 1.0 if is_true else 0.4

        # 연결선 및 점
        pts = []
        if in_dig: pts.append((X_POS['dig'],  dr))
        if in_A:   pts.append((X_POS['optA'], ar))
        if in_C:   pts.append((X_POS['optC'], cr))

        for k in range(len(pts) - 1):
            ax.plot([pts[k][0], pts[k+1][0]], [pts[k][1], pts[k+1][1]],
                    color=col, lw=lw, alpha=alph, zorder=2)

        for (xp, yp) in pts:
            ax.scatter(xp, yp, color=col,
                       s=80 if is_true else 18,
                       marker='*' if is_true else 'o',
                       zorder=4, alpha=max(alph, 0.6))

        # 정답 주석
        if is_true:
            for xp, yp, lbl in [(X_POS['dig'], dr, f'L2:{dr}') if in_dig else (None, None, None),
                                 (X_POS['optA'], ar, f'A:{ar}') if in_A  else (None, None, None),
                                 (X_POS['optC'], cr, f'C:{cr}') if in_C  else (None, None, None)]:
                if xp is None:
                    continue
                offset = (-45, 4) if xp == 0 else (6, 4)
                ax.annotate(f'True\n({lbl})', (xp, yp),
                            textcoords='offset points', xytext=offset,
                            fontsize=7, color=COLORS['true'], fontweight='bold',
                            ha='left' if xp == 0 else 'left')

    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(TOP_K + 1, 0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Digital L2', 'Option A', 'Option C'],
                       fontsize=9, fontweight='bold')
    ax.set_ylabel(f'Rank (top-{TOP_K})', fontsize=9)
    ax.set_title(f'{Q_LABELS[q_id]}\nTop-{TOP_K} rank comparison',
                 fontsize=8.5, fontweight='bold')
    ax.grid(axis='y', ls='--', alpha=0.2)

    h_all  = mlines.Line2D([], [], color=COLORS['dig'],  lw=2,   label='In 2+ methods')
    h_A    = mlines.Line2D([], [], color=COLORS['optA'], lw=1.5, label='Option A only')
    h_C    = mlines.Line2D([], [], color=COLORS['optC'], lw=1.5, label='Option C only')
    h_true = mlines.Line2D([], [], color=COLORS['true'], lw=2.5, label='True document')
    ax.legend(handles=[h_all, h_A, h_C, h_true], fontsize=7.5, loc='lower right')

plt.tight_layout()
out = f'{BASE}/compare_ranking_all3.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out}")
