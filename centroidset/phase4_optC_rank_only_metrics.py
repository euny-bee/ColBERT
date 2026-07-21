"""
Phase 4 (rank-only). 후보 검색 문제 vs 랭킹 문제 기여도 분리
  입력 : phase3_vth_results.csv            (Option C, 자체 후보)
         phase3_optC_rank_only_results.csv (Option C, Option A 후보 재사용)
  출력 : phase4_optC_rank_only_metrics.csv
         조건별로 두 결과를 나란히 비교
"""

import numpy as np
import pandas as pd

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'

df_own  = pd.read_csv(f'{BASE}/phase3_vth_results.csv')
df_rank = pd.read_csv(f'{BASE}/phase3_optC_rank_only_results.csv')

print(f"자체 후보 결과: {len(df_own)} rows")
print(f"rank-only 결과: {len(df_rank)} rows")

# ── 지표 계산 함수 (phase4_vth_metrics.py와 동일) ────────────────
def mrr_at_k(ranks, k=10):
    return float(np.mean([1/r if (r is not None and not np.isnan(float(r)) and r <= k) else 0.0
                          for r in ranks]))

def r_at_k(ranks, k):
    return float(np.mean([1 if (r is not None and not np.isnan(float(r)) and r <= k) else 0.0
                          for r in ranks]))

def ndcg_at_k(ranks, k=10):
    return float(np.mean([1/np.log2(r+1) if (r is not None and not np.isnan(float(r)) and r <= k) else 0.0
                          for r in ranks]))

def metrics(ranks):
    return {
        'MRR@10':  round(mrr_at_k(ranks, 10), 4),
        'nDCG@10': round(ndcg_at_k(ranks, 10), 4),
        'R@50':    round(r_at_k(ranks, 50),   4),
        'R@1k':    round(r_at_k(ranks, 1000), 4),
    }

ALL_CONDS = [
    ('v1', 'shift [0, 0.5]V'),
    ('v2', 'shift [0, 1.0]V'),
    ('v3', 'shift [0, 2.0]V'),
    ('v4', 'shift [0, 3.0]V'),
]
available = set(df_rank['vth_cond'].unique())
CONDS = [c for c in ALL_CONDS if c[0] in available]
print(f"rank-only 결과가 있는 조건: {[c[0] for c in CONDS]}")

rows = []
for cname, label in CONDS:
    sub_own  = df_own[df_own['vth_cond'] == cname]
    sub_rank = df_rank[df_rank['vth_cond'] == cname]

    m_own  = metrics(sub_own['rank_optC'].tolist())
    rows.append({'Variant': 'Option C (own candidates)', 'Vth condition': label, **m_own})

    m_rank = metrics(sub_rank['rank_optC_rankonly'].tolist())
    rows.append({'Variant': 'Option C (Option A candidates, rank-only)', 'Vth condition': label, **m_rank})

result_df = pd.DataFrame(rows)
out = f'{BASE}/phase4_optC_rank_only_metrics.csv'
result_df.to_csv(out, index=False)

# ── 콘솔 출력 ──────────────────────────────────────────────────
print()
print("=" * 98)
print(f"{'Variant':<45} {'Vth condition':<18} {'MRR@10':>7} {'nDCG@10':>8} {'R@50':>6} {'R@1k':>6}")
print("-" * 98)
for _, row in result_df.iterrows():
    print(f"{row['Variant']:<45} {row['Vth condition']:<18} "
          f"{row['MRR@10']:>7.4f} {row['nDCG@10']:>8.4f} "
          f"{row['R@50']:>6.4f} {row['R@1k']:>6.4f}")
print("=" * 98)

# ── 후보 검색률 비교 ──────────────────────────────────────────────
print()
print("후보 검색률 (true_pid in candidates):")
for cname, label in CONDS:
    sub_own  = df_own[df_own['vth_cond'] == cname]
    sub_rank = df_rank[df_rank['vth_cond'] == cname]
    in_own  = (sub_own['rank_optC'].notna()).mean()
    in_rank = (sub_rank['rank_optC_rankonly'].notna()).mean()
    print(f"  [{cname}] own_candidates={in_own:.3f}  optA_candidates={in_rank:.3f}")

print()
print(f"저장: {out}")
print("Phase 4 (rank-only) 완료!")
