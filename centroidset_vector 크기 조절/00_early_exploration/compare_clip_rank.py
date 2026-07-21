"""
[clip] vs [original] query_centroid_ranking top-5 rank_order 비교
"""
import pandas as pd
import numpy as np

BASE      = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline'
ORIG_BASE = f'{BASE}\clip99.9외'

# [original] rank_order 로드
orig_order = pd.read_excel(f'{ORIG_BASE}/[original]query_centroid_ranking.xlsx',
                           sheet_name='rank_order', index_col=0)

# [clip] scores 로드 후 rank_order 직접 계산
clip_scores = pd.read_excel(f'{ORIG_BASE}/[clip]query_centroid_ranking.xlsx', index_col=0)
centroid_cols = [c for c in clip_scores.columns if c.startswith('centroid_')]
scores_vals   = clip_scores[centroid_cols].values
clip_order_arr = np.argsort(-scores_vals, axis=1)   # (96, 100)

query_ids = orig_order['query_id'].values
rank_cols  = [f'rank_{i+1}' for i in range(100)]
orig_arr   = orig_order[rank_cols].values           # (96, 100)

TOP = 5

print(f"{'='*60}")
print(f"[clip] vs [original]  top-{TOP} rank_order 비교")
print(f"{'='*60}")

for q_id in sorted(set(query_ids)):
    mask   = query_ids == q_id
    tokens = orig_order.index[mask].tolist()
    o_top  = orig_arr[mask, :TOP]         # (n_tokens, 5)
    c_top  = clip_order_arr[mask, :TOP]   # (n_tokens, 5)

    print(f"\n[Query {q_id}]  토큰 {len(tokens)}개")
    print(f"{'token':<10} {'original top5':^35} {'clip top5':^35} {'일치':^8}")
    print("-" * 95)

    all_matches = []
    for i, tok in enumerate(tokens):
        o5 = o_top[i].tolist()
        c5 = c_top[i].tolist()
        match = len(set(o5) & set(c5))
        all_matches.append(match)
        o_str = str(o5)
        c_str = str(c5)
        marker = "" if match == TOP else f"<- {match}/5"
        print(f"{tok:<10} {o_str:<35} {c_str:<35} {match}/5  {marker}")

    avg = np.mean(all_matches)
    full = sum(1 for m in all_matches if m == TOP)
    print(f"\n  -> 완전 일치(5/5): {full}/{len(tokens)} 토큰  |  평균 일치: {avg:.2f}/5")

# 전체 요약
print(f"\n{'='*60}")
all_o = orig_arr[:, :TOP]
all_c = clip_order_arr[:, :TOP]
total_match = np.mean([len(set(all_o[i]) & set(all_c[i])) for i in range(len(all_o))])
full_total  = sum(1 for i in range(len(all_o)) if set(all_o[i]) == set(all_c[i]))
print(f"전체 96토큰  완전 일치(5/5): {full_total}/96  |  평균 일치: {total_match:.2f}/5")
