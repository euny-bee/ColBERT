# -*- coding: utf-8 -*-
"""Phase 4. 지표 계산 및 분석"""

import pandas as pd
import numpy as np

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'

# ── 데이터 합치기 ──────────────────────────────────────────────
dig  = pd.read_csv(f'{BASE}/phase3_digital_results.csv')
ana0 = pd.read_csv(f'{BASE}/phase3_analog_results_gpu0.csv')
ana1 = pd.read_csv(f'{BASE}/phase3_analog_results_gpu1.csv')
ana  = pd.concat([ana1, ana0]).sort_values('query_idx').reset_index(drop=True)

df = dig.merge(ana[['query_idx','n_cands_ana','in_ana',
                     'rank_ana_f32','rank_ana_2bt',
                     'curr_ana_f32','curr_ana_2bt']], on='query_idx')
df.to_csv(f'{BASE}/phase3_combined.csv', index=False)
print(f"combined: {df.shape}")
print(f"in_dig: {df['in_dig'].sum()}  in_ana: {df['in_ana'].sum()}")

N = len(df)

# ── 지표 계산 함수 ──────────────────────────────────────────────
def mrr_at_k(ranks, k=10):
    rr = []
    for r in ranks:
        if r is None or np.isnan(r) or r > k:
            rr.append(0.0)
        else:
            rr.append(1.0 / r)
    return np.mean(rr)

def success_at_k(ranks, k):
    hits = sum(1 for r in ranks if r is not None and not np.isnan(r) and r <= k)
    return hits / len(ranks)

def ndcg_at_k(ranks, k=10):
    scores = []
    for r in ranks:
        if r is None or np.isnan(r) or r > k:
            scores.append(0.0)
        else:
            scores.append(1.0 / np.log2(r + 1))
    return np.mean(scores)

# ── [4-1] 전체 지표 ─────────────────────────────────────────────
methods = {
    'dig_f32': df['rank_dig_f32'].tolist(),
    'dig_2bt': df['rank_dig_2bt'].tolist(),
    'ana_f32': df['rank_ana_f32'].tolist(),
    'ana_2bt': df['rank_ana_2bt'].tolist(),
}

print("\n" + "="*60)
print(f"[4-1] 전체 지표 (N={N} queries, nprobe=2)")
print("="*60)

metric_rows = []
for name, ranks in methods.items():
    row = {
        'method':      name,
        'MRR@10':      round(mrr_at_k(ranks, 10), 4),
        'Success@1':   round(success_at_k(ranks, 1)*100, 2),
        'Success@5':   round(success_at_k(ranks, 5)*100, 2),
        'Success@10':  round(success_at_k(ranks, 10)*100, 2),
        'Success@50':  round(success_at_k(ranks, 50)*100, 2),
        'R@50':        round(success_at_k(ranks, 50)*100, 2),
        'R@1000':      round(success_at_k(ranks, 1000)*100, 2),
        'nDCG@10':     round(ndcg_at_k(ranks, 10), 4),
        'in_cands':    f"{sum(1 for r in ranks if r is not None and not np.isnan(r))}/{N}",
    }
    metric_rows.append(row)
    print(f"\n  [{name}]")
    for k, v in row.items():
        if k != 'method':
            print(f"    {k:12s}: {v}")

df_metrics = pd.DataFrame(metric_rows)

# ── [4-2] margin 구간별 분석 ────────────────────────────────────
print("\n" + "="*60)
print("[4-2] Margin 구간별 Success@1 (dig_f32 기준 margin)")
print("="*60)

bins   = [(-np.inf,1), (1,2), (2,5), (5,np.inf)]
labels = ['<1', '1~2', '2~5', '5+']

margin_rows = []
for (lo, hi), lbl in zip(bins, labels):
    mask = df['margin'].apply(lambda m: m is not None and not (isinstance(m,float) and np.isnan(m)) and lo <= m < hi)
    sub  = df[mask]
    n_sub = len(sub)
    if n_sub == 0:
        continue
    row = {'margin': lbl, 'n_queries': n_sub}
    for name, col in [('dig_f32','rank_dig_f32'),('dig_2bt','rank_dig_2bt'),
                      ('ana_f32','rank_ana_f32'),('ana_2bt','rank_ana_2bt')]:
        ranks = sub[col].tolist()
        row[f'S@1_{name}']  = round(success_at_k(ranks,1)*100,1)
        row[f'S@5_{name}']  = round(success_at_k(ranks,5)*100,1)
    margin_rows.append(row)
    print(f"\n  margin {lbl} (n={n_sub}):")
    print(f"    Success@1: dig_f32={row['S@1_dig_f32']}%  dig_2bt={row['S@1_dig_2bt']}%  ana_f32={row['S@1_ana_f32']}%  ana_2bt={row['S@1_ana_2bt']}%")

# margin=None (relevant not in cands) 쿼리
no_margin = df['margin'].isna().sum()
print(f"\n  margin=None (정답 미검색): {no_margin}/{N} = {no_margin/N*100:.1f}%")

# ── [4-3] digital != analog 케이스 ──────────────────────────────
print("\n" + "="*60)
print("[4-3] Digital vs Analog 결과 불일치 케이스")
print("="*60)

# 정답이 어느 한쪽에서는 찾아지는 경우
dig_found = df['in_dig']
ana_found = df['in_ana']

only_dig = df[dig_found & ~ana_found]
only_ana = df[~dig_found & ana_found]
both     = df[dig_found & ana_found]
neither  = df[~dig_found & ~ana_found]

print(f"  둘 다 찾음:          {len(both)}/{N}")
print(f"  Digital만 찾음:      {len(only_dig)}/{N}")
print(f"  Analog만 찾음:       {len(only_ana)}/{N}")
print(f"  둘 다 못 찾음:       {len(neither)}/{N}")

query_meta = pd.read_csv(f'{BASE}/scale_query_meta.csv')
if len(only_dig) > 0:
    print(f"\n  [Digital만 찾은 쿼리]")
    for _, row in only_dig.iterrows():
        qid = row['qid']
        text = query_meta[query_meta['qid']==qid]['text'].values
        text = text[0][:60] if len(text) > 0 else '?'
        print(f"    q{int(row['query_idx'])}: \"{text}\"  rank_dig={row['rank_dig_f32']}")

if len(only_ana) > 0:
    print(f"\n  [Analog만 찾은 쿼리]")
    for _, row in only_ana.iterrows():
        qid = row['qid']
        text = query_meta[query_meta['qid']==qid]['text'].values
        text = text[0][:60] if len(text) > 0 else '?'
        print(f"    q{int(row['query_idx'])}: \"{text}\"  rank_ana={row['rank_ana_f32']}")

# ── 저장 ────────────────────────────────────────────────────────
df_metrics.to_csv(f'{BASE}/phase4_metrics.csv', index=False)

df_margin = pd.DataFrame(margin_rows)
df_margin.to_csv(f'{BASE}/phase4_margin_breakdown.csv', index=False)

print(f"\n저장 완료:")
print(f"  phase4_metrics.csv")
print(f"  phase4_margin_breakdown.csv")
print(f"  phase3_combined.csv")
print("\nPhase 4 완료!")
