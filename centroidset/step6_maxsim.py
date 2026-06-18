"""
Step 6: MaxSim 계산 → 최종 랭킹
두 가지 doc 벡터 비교:
  - float32: C_t + r_float32 (원본과 동일)
  - 2bit:    C_t + r_2bit    (ColBERTv2 실제 압축)
"""

import pandas as pd
import numpy as np
import os

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}

# -- 데이터 로드 --------------------------------------------------------------
Q_raw = pd.read_excel(f'{BASE}/query_embs_96x128.xlsx',      header=0, index_col=0)
C     = pd.read_excel(f'{BASE}/centroids_100x128.xlsx',       header=0, index_col=0).values.astype(float)
ivf   = pd.read_excel(f'{BASE}/ivf_centroid2token2pid.xlsx',  header=0, index_col=0)
r32   = pd.read_excel(f'{BASE}/residuals_float32.xlsx',       header=0, index_col=0)
r2b   = pd.read_csv(  f'{BASE}/residuals_2bit.csv',           header=0, index_col=0)
doc   = pd.read_excel(f'{BASE}/doc_embs_12919x128.xlsx',      header=0, index_col=0)
cands = pd.read_excel(f'{BASE}/step3_candidate_pids.xlsx',    sheet_name=None)  # 시트별 로드

dim_cols = [c for c in doc.columns if c.startswith('dim_')]

# -- doc 벡터 두 가지 준비 ---------------------------------------------------
# float32: doc_embs 그대로 (= C_t + r_float32)
D_f32 = doc[dim_cols].values.astype(float)

# 2bit: C_t + r_2bit (token_id 정렬 맞춤)
token2centroid = dict(zip(ivf['token_id'], ivf.index))
centroid_ids   = [token2centroid[t] for t in doc.index]
Ct             = C[centroid_ids]
D_2bt          = Ct + r2b.values.astype(float)

# doc index → 행 번호 매핑
doc_token_index = {tok: i for i, tok in enumerate(doc.index)}

# -- MaxSim 함수 -------------------------------------------------------------
def maxsim_score(Q, D_pid):
    """
    Q:     (32, 128) query token embeddings
    D_pid: (n_tokens, 128) passage token embeddings
    반환:  scalar MaxSim score
    """
    sim = Q @ D_pid.T          # (32, n_tokens)
    return sim.max(axis=1).sum()

# -- query별 처리 ------------------------------------------------------------
results = {}

for q_id in [0, 1, 2]:
    Q = Q_raw.iloc[q_id*32:(q_id+1)*32].values.astype(float)  # (32, 128)

    # 이 query의 후보 pid 목록
    cand_pids = cands[f'q{q_id}']['pid'].tolist()
    true_pid  = QUERY_RELEVANT[q_id]

    rows = []
    for pid in cand_pids:
        # 해당 pid의 토큰 행 인덱스
        token_keys = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not token_keys:
            continue
        idxs = [doc_token_index[k] for k in token_keys]

        score_f32 = maxsim_score(Q, D_f32[idxs])
        score_2bt = maxsim_score(Q, D_2bt[idxs])

        rows.append({
            'pid':          pid,
            'is_relevant':  pid == true_pid,
            'score_float32': round(score_f32, 4),
            'score_2bit':    round(score_2bt, 4),
        })

    df = pd.DataFrame(rows)
    df['rank_float32'] = df['score_float32'].rank(ascending=False).astype(int)
    df['rank_2bit']    = df['score_2bit'].rank(ascending=False).astype(int)
    df = df.sort_values('rank_float32').reset_index(drop=True)

    rel_row = df[df['is_relevant']]
    print(f"[q{q_id}]  정답 pid: {true_pid}")
    if len(rel_row):
        r = rel_row.iloc[0]
        print(f"  float32  → score: {r['score_float32']:.4f},  rank: {int(r['rank_float32'])}/{len(df)}")
        print(f"  2bit     → score: {r['score_2bit']:.4f},  rank: {int(r['rank_2bit'])}/{len(df)}")
    else:
        print("  정답 pid 후보에 없음")
    print()

    results[f'q{q_id}'] = df

# -- 저장 --------------------------------------------------------------------
out_path = f'{BASE}/step6_maxsim_results.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    for q_name, df_r in results.items():
        df_r.to_excel(writer, sheet_name=q_name, index=False)

    # summary
    summary = []
    for q_id in [0, 1, 2]:
        df_r = results[f'q{q_id}']
        rel  = df_r[df_r['is_relevant']]
        if len(rel):
            r = rel.iloc[0]
            summary.append({
                'query':         f'q{q_id}',
                'true_pid':      QUERY_RELEVANT[q_id],
                'n_candidates':  len(df_r),
                'rank_float32':  int(r['rank_float32']),
                'rank_2bit':     int(r['rank_2bit']),
                'score_float32': r['score_float32'],
                'score_2bit':    r['score_2bit'],
            })
    pd.DataFrame(summary).to_excel(writer, sheet_name='summary', index=False)

print(f"Saved: {out_path}")
