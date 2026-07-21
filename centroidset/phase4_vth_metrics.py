"""
Phase 4 Vth 변이 지표 계산
  입력 : phase3_vth_results.csv  (255 x 4 = 1020 rows)
  출력 : phase4_vth_metrics.csv  (논문 Table 4 형식)
         MRR@10 / nDCG@10 / R@50 / R@1k
"""

import numpy as np
import pandas as pd

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'

df = pd.read_csv(f'{BASE}/phase3_vth_results.csv')
NQ = df['query_idx'].nunique()
print(f"로드: {len(df)} rows  queries={NQ}  conditions={df['vth_cond'].nunique()}")

# ── 지표 계산 함수 ──────────────────────────────────────────────
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

CONDS = [
    ('v1', 'shift [0, 0.5]V  (actual [0.151, 0.651]V)'),
    ('v2', 'shift [0, 1.0]V  (actual [0.151, 1.151]V)'),
    ('v3', 'shift [0, 2.0]V  (actual [0.151, 2.151]V)'),
    ('v4', 'shift [0, 3.0]V  (actual [0.151, 3.151]V)'),
]

rows = []

# Digital (Vth 조건 무관 — v1 대표값)
sub0 = df[df['vth_cond'] == 'v1']
m = metrics(sub0['rank_dig'].tolist())
rows.append({'Method': 'Digital', 'Vth condition': '--', **m})

# Option A + Option C per condition
for cname, label in CONDS:
    sub = df[df['vth_cond'] == cname]

    ma = metrics(sub['rank_optA'].tolist())
    rows.append({'Method': 'Option A (VthComp)', 'Vth condition': label, **ma})

    mc = metrics(sub['rank_optC'].tolist())
    rows.append({'Method': 'Option C (NoComp)', 'Vth condition': label, **mc})

result_df = pd.DataFrame(rows)
out = f'{BASE}/phase4_vth_metrics.csv'
result_df.to_csv(out, index=False)

# ── 콘솔 출력 ──────────────────────────────────────────────────
print()
print("=" * 78)
print(f"{'Method':<22} {'Vth condition':<42} {'MRR@10':>7} {'nDCG@10':>8} {'R@50':>6} {'R@1k':>6}")
print("-" * 78)
for _, row in result_df.iterrows():
    print(f"{row['Method']:<22} {row['Vth condition']:<42} "
          f"{row['MRR@10']:>7.4f} {row['nDCG@10']:>8.4f} "
          f"{row['R@50']:>6.4f} {row['R@1k']:>6.4f}")
print("=" * 78)

# ── 검색률 요약 (정답이 후보에 포함되는 비율) ────────────────────
print()
print("후보 검색률 (true_pid in candidates):")
for cname, label in CONDS:
    sub = df[df['vth_cond'] == cname]
    in_d = (sub['rank_dig'].notna()).mean()
    in_a = (sub['rank_optA'].notna()).mean()
    in_c = (sub['rank_optC'].notna()).mean()
    print(f"  [{cname}] Digital={in_d:.3f}  OptA={in_a:.3f}  OptC={in_c:.3f}")

print()
print(f"저장: {out}")
print("Phase 4 Vth 완료!")
