"""
Step 3: Inverted list lookup
각 query token의 top-nprobe centroid를 찾고,
inverted list에서 해당 centroid에 연결된 pid를 수집
"""

import pandas as pd
import numpy as np
import os

NPROBE = 2
BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'

# query별 정답 pid (MaxSim으로 유추)
QUERY_RELEVANT = {
    0: 7264308,
    1: 7264266,
    2: 7264253,
}

# -- load data ----------------------------------------------------------------
df_rank = pd.read_excel(os.path.join(BASE, 'query_centroid_ranking.xlsx'),
                        header=0, index_col=0)   # (96, 101)
df_ivf  = pd.read_excel(os.path.join(BASE, 'ivf_centroid2pid.xlsx'),
                        header=0, index_col=0)   # (1773, 2)

centroid_cols = [c for c in df_rank.columns if c.startswith('centroid_')]

# -- per-query processing -----------------------------------------------------
results = {}   # query_id -> DataFrame

for q_id in [0, 1, 2]:
    # 이 query에 속한 토큰 행만 선택 (query_id 컬럼으로 필터)
    mask    = df_rank['query_id'] == q_id
    df_q    = df_rank.loc[mask, centroid_cols]   # (32, 100)

    # 각 token별 top-nprobe centroid ID 추출
    token_top_centroids = {}   # token_id -> [c_id, ...]
    for token_id, row in df_q.iterrows():
        top_c = row.nlargest(NPROBE).index.str.replace('centroid_', '').astype(int).tolist()
        token_top_centroids[token_id] = top_c

    # inverted list 조회 → pid 수집
    pid_info = {}   # pid -> {is_relevant, centroid_ids, token_ids}
    for token_id, c_ids in token_top_centroids.items():
        for c_id in c_ids:
            if c_id not in df_ivf.index:
                continue
            rows = df_ivf.loc[[c_id]]   # 해당 centroid의 모든 pid rows
            for _, row in rows.iterrows():
                pid = row['pid']
                is_rel = row['is_relevant']
                if pid not in pid_info:
                    pid_info[pid] = {'is_relevant': is_rel,
                                     'centroid_ids': set(),
                                     'token_ids': set()}
                pid_info[pid]['centroid_ids'].add(c_id)
                pid_info[pid]['token_ids'].add(token_id)

    # DataFrame으로 변환
    rows_list = []
    for pid, info in pid_info.items():
        rows_list.append({
            'pid':           pid,
            'is_relevant':   info['is_relevant'],
            'n_centroids':   len(info['centroid_ids']),   # 몇 개 centroid에서 등장
            'n_tokens':      len(info['token_ids']),       # 몇 개 토큰에서 등장
            'centroid_ids':  sorted(info['centroid_ids']),
            'token_ids':     sorted(info['token_ids']),
        })

    df_result = pd.DataFrame(rows_list).sort_values('n_tokens', ascending=False)
    df_result = df_result.reset_index(drop=True)

    # query별 정답 pid 기준으로 is_relevant 재정의
    true_pid = QUERY_RELEVANT[q_id]
    df_result['is_relevant'] = df_result['pid'] == true_pid

    results[f'q{q_id}'] = df_result

    found = true_pid in df_result['pid'].values
    print(f"[q{q_id}] 후보 passage 수: {len(df_result)}, "
          f"정답 pid {true_pid} 포함: {found}")

# -- summary ------------------------------------------------------------------
print()
all_pids   = set().union(*[set(df['pid']) for df in results.values()])
print(f"전체 unique 후보 pid 수 (3 queries 합계): {len(all_pids)}")

# -- save to xlsx -------------------------------------------------------------
out_path = os.path.join(BASE, 'step3_candidate_pids.xlsx')
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    for q_name, df_r in results.items():
        df_r.to_excel(writer, sheet_name=q_name, index=False)

    # summary sheet
    summary_rows = []
    for q_name, df_r in results.items():
        summary_rows.append({
            'query':            q_name,
            'n_candidates':     len(df_r),
            'n_relevant':       int(df_r['is_relevant'].sum()),
            'total_pids_in_corpus': df_ivf['pid'].nunique(),
        })
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='summary', index=False)

print(f"\nSaved: {out_path}")
