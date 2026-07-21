"""
[clip99.9] ivf_centroid2pid, query_centroid_ranking 을
원본과 동일한 탭 구조로 재생성
"""

import pandas as pd
import numpy as np
from collections import defaultdict

BASE      = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline'
ORIG_BASE = f'{BASE}\clip99.9외'

# ==========================================================================
# 로드
# ==========================================================================
cent_df  = pd.read_excel(f'{BASE}/[clip99.9]centroids_100x128.xlsx',    index_col=0)
doc_df   = pd.read_excel(f'{BASE}/[clip99.9]doc_embs_12919x128.xlsx',   index_col=0)
query_df = pd.read_excel(f'{BASE}/[clip99.9]query_embs_96x128.xlsx',    index_col=0)
qcr_orig = pd.read_excel(f'{ORIG_BASE}/[original]query_centroid_ranking.xlsx',
                          sheet_name='scores', index_col=0)

dim_cols      = [c for c in doc_df.columns if c.startswith('dim_')]
centroid_cols = [f'centroid_{i}' for i in range(100)]

C = cent_df.values.astype(float)
D = doc_df[dim_cols].values.astype(float)
Q = query_df.values.astype(float)

is_rel_arr = doc_df['is_relevant'].values.astype(bool)
token_ids  = doc_df.index.tolist()
pids       = np.array([int(t.rsplit('_t', 1)[0]) for t in token_ids], dtype=np.int64)
query_ids  = qcr_orig['query_id'].values

# ==========================================================================
# IVF 재계산
# ==========================================================================
D_n  = (D**2).sum(axis=1, keepdims=True)
C_n  = (C**2).sum(axis=1)
dist = D_n + C_n[np.newaxis, :] - 2*(D @ C.T)
asgn = np.argmin(dist, axis=1)

ivf = defaultdict(dict)
for c_id, pid, is_rel in zip(asgn, pids, is_rel_arr):
    if pid not in ivf[int(c_id)]:
        ivf[int(c_id)][pid] = bool(is_rel)

# long_format
long_rows = []
for c_id in sorted(ivf.keys()):
    for pid, is_rel in ivf[c_id].items():
        long_rows.append({'centroid_id': c_id, 'pid': int(pid), 'is_relevant': is_rel})
long_df = pd.DataFrame(long_rows).set_index('centroid_id')
long_df.index.name = 'centroid_id'

# wide_format
max_pids = max(len(v) for v in ivf.values())
wide_rows = {}
for c_id in range(100):
    pid_list = list(ivf[c_id].keys()) if c_id in ivf else []
    row = {'n_passages': len(pid_list)}
    for i, p in enumerate(pid_list):
        row[f'pid_{i}'] = int(p)
    wide_rows[c_id] = row

wide_df = pd.DataFrame(wide_rows).T
wide_df.index.name = 'centroid_id'
wide_df['n_passages'] = wide_df['n_passages'].astype(int)

# ==========================================================================
# QCR 재계산
# ==========================================================================
scores = Q @ C.T   # (96, 100)

# scores sheet
scores_df = pd.DataFrame(scores, index=query_df.index, columns=centroid_cols)
scores_df.insert(0, 'query_id', query_ids)
scores_df.index.name = qcr_orig.index.name

# ranks sheet: 각 centroid가 해당 토큰에서 몇 등인지 (1=1등)
ranks = scores.shape[1] - np.argsort(np.argsort(scores, axis=1), axis=1)
ranks_df = pd.DataFrame(ranks, index=query_df.index, columns=centroid_cols)
ranks_df.insert(0, 'query_id', query_ids)
ranks_df.index.name = qcr_orig.index.name

# rank_order sheet: 1등 centroid_id, 2등 centroid_id, ...
order = np.argsort(-scores, axis=1)   # (96, 100)
rank_cols = [f'rank_{i+1}' for i in range(100)]
order_df = pd.DataFrame(order, index=query_df.index, columns=rank_cols)
order_df.insert(0, 'query_id', query_ids)
order_df.index.name = qcr_orig.index.name

# ==========================================================================
# 저장
# ==========================================================================
ivf_path = f'{BASE}/[clip99.9]ivf_centroid2pid.xlsx'
with pd.ExcelWriter(ivf_path, engine='openpyxl') as writer:
    long_df.to_excel(writer, sheet_name='long_format')
    wide_df.to_excel(writer, sheet_name='wide_format')
print(f'저장: [clip99.9]ivf_centroid2pid.xlsx  (탭: long_format, wide_format)')

qcr_path = f'{BASE}/[clip99.9]query_centroid_ranking.xlsx'
with pd.ExcelWriter(qcr_path, engine='openpyxl') as writer:
    scores_df.to_excel(writer, sheet_name='scores')
    ranks_df.to_excel(writer,  sheet_name='ranks')
    order_df.to_excel(writer,  sheet_name='rank_order')
print(f'저장: [clip99.9]query_centroid_ranking.xlsx  (탭: scores, ranks, rank_order)')

# ==========================================================================
# 탭 구조 확인
# ==========================================================================
print()
for path in [ivf_path, qcr_path]:
    xl = pd.ExcelFile(path)
    fname = path.split('\\')[-1]
    print(f'{fname}:')
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, index_col=0)
        print(f'  [{sheet}] shape={df.shape}')
