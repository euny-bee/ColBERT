"""
99% / 99.9% percentile 기준 clip+scale 두 버전 생성 및 original 비교
"""

import pandas as pd
import numpy as np
from collections import defaultdict

ORIG_BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline\clip외'
OUT_BASE  = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline'

# ==========================================================================
# 1. 원본 로드
# ==========================================================================
print("원본 파일 로드 중...")
cent_df  = pd.read_excel(f'{ORIG_BASE}/centroids_100x128.xlsx',    index_col=0)
doc_df   = pd.read_excel(f'{ORIG_BASE}/doc_embs_12919x128.xlsx',   index_col=0)
query_df = pd.read_excel(f'{ORIG_BASE}/query_embs_96x128.xlsx',    index_col=0)
ivf_orig = pd.read_excel(f'{ORIG_BASE}/ivf_centroid2pid.xlsx',     index_col=0)
qcr_orig = pd.read_excel(f'{ORIG_BASE}/query_centroid_ranking.xlsx', index_col=0)

dim_cols    = [c for c in doc_df.columns if c.startswith('dim_')]
centroid_cols = [f'centroid_{i}' for i in range(100)]

C = cent_df.values.astype(float)
D = doc_df[dim_cols].values.astype(float)
Q = query_df.values.astype(float)

is_rel_arr = doc_df['is_relevant'].values.astype(bool)
token_ids  = doc_df.index.tolist()
pids       = np.array([int(t.rsplit('_t', 1)[0]) for t in token_ids], dtype=np.int64)
query_ids  = qcr_orig['query_id'].values

# ==========================================================================
# 2. Global percentile threshold 계산 (doc_dims + centroids + query 전체)
# ==========================================================================
all_vals = np.concatenate([D.flatten(), C.flatten(), Q.flatten()])
all_abs  = np.abs(all_vals)

thr_99   = np.percentile(all_abs, 99)
thr_99_9 = np.percentile(all_abs, 99.9)

print(f"\n[Global threshold]")
print(f"  99th  percentile: {thr_99:.6f}  → scale_factor = {1/thr_99:.6f}")
print(f"  99.9th percentile: {thr_99_9:.6f}  → scale_factor = {1/thr_99_9:.6f}")

# ==========================================================================
# 3. 공통 함수
# ==========================================================================
def build_ivf(D_emb, C_emb, pids_arr, is_rel_arr):
    D_n   = (D_emb**2).sum(axis=1, keepdims=True)
    C_n   = (C_emb**2).sum(axis=1)
    dist  = D_n + C_n[np.newaxis, :] - 2*(D_emb @ C_emb.T)
    asgn  = np.argmin(dist, axis=1)
    ivf   = defaultdict(dict)
    for c_id, pid, is_rel in zip(asgn, pids_arr, is_rel_arr):
        c_id = int(c_id)
        if pid not in ivf[c_id]:
            ivf[c_id][pid] = bool(is_rel)
    rows = []
    for c_id in sorted(ivf.keys()):
        for pid, is_rel in ivf[c_id].items():
            rows.append({'centroid_id': c_id, 'pid': int(pid), 'is_relevant': is_rel})
    df = pd.DataFrame(rows).set_index('centroid_id')
    df.index.name = 'centroid_id'
    return df, asgn

def build_qcr(Q_emb, C_emb, query_ids_arr, token_index, idx_name):
    dot = Q_emb @ C_emb.T
    df  = pd.DataFrame(dot, index=token_index, columns=centroid_cols)
    df.insert(0, 'query_id', query_ids_arr)
    df.index.name = idx_name
    return df

def ivf_to_set(df):
    return set(zip(df.index.tolist(), df['pid'].tolist()))

def compare_ivf(ivf_new, ivf_ref, label):
    new_set = ivf_to_set(ivf_new)
    ref_set = ivf_to_set(ivf_ref)
    only_ref = ref_set - new_set
    only_new = new_set - ref_set
    print(f"  ivf 비교 ({label} vs original):")
    print(f"    original rows : {len(ref_set)}")
    print(f"    new rows      : {len(new_set)}")
    print(f"    original에만  : {len(only_ref)}")
    print(f"    new에만       : {len(only_new)}")
    if len(only_ref) == 0 and len(only_new) == 0:
        print(f"    -> 완전 일치")
    else:
        print(f"    -> 달라진 pairs: {len(only_ref) + len(only_new)}")

def compare_qcr(qcr_new, qcr_ref, label):
    ref_scores = qcr_ref[centroid_cols].values
    new_scores = qcr_new[centroid_cols].values
    # top-1, top-2 일치율
    ref_top1 = np.argmax(ref_scores, axis=1)
    new_top1 = np.argmax(new_scores, axis=1)
    top1_match = (ref_top1 == new_top1).mean()

    ref_top2 = set(map(tuple, np.argsort(-ref_scores, axis=1)[:, :2].tolist()))
    new_top2 = set(map(tuple, np.argsort(-new_scores, axis=1)[:, :2].tolist()))
    top2_match = np.mean([
        len(set(np.argsort(-ref_scores[i])[:2]) & set(np.argsort(-new_scores[i])[:2])) / 2
        for i in range(len(ref_scores))
    ])

    print(f"  query_centroid_ranking 비교 ({label} vs original):")
    print(f"    top-1 centroid 일치율: {top1_match*100:.1f}%  ({int(top1_match*len(ref_top1))}/{len(ref_top1)} 토큰)")
    print(f"    top-2 centroid 일치율: {top2_match*100:.1f}%")

# ==========================================================================
# 4. 두 케이스 생성
# ==========================================================================
cases = [
    ('clip99',   thr_99,   '[clip99]'),
    ('clip99.9', thr_99_9, '[clip99.9]'),
]

for case_name, thr, prefix in cases:
    print(f"\n{'='*60}")
    print(f"{prefix} 생성 중... (threshold={thr:.6f}, scale={1/thr:.6f})")

    sf = 1.0 / thr

    C_new = np.clip(C, -thr, thr) * sf
    D_new = np.clip(D, -thr, thr) * sf
    Q_new = np.clip(Q, -thr, thr) * sf

    print(f"  범위 확인:")
    print(f"    centroids : {C_new.min():.4f} ~ {C_new.max():.4f}")
    print(f"    doc_embs  : {D_new.min():.4f} ~ {D_new.max():.4f}")
    print(f"    query_embs: {Q_new.min():.4f} ~ {Q_new.max():.4f}")

    # clip된 값 수
    n_clip_D = int((np.abs(D) > thr).sum())
    n_clip_C = int((np.abs(C) > thr).sum())
    n_clip_Q = int((np.abs(Q) > thr).sum())
    print(f"  clip된 값: doc={n_clip_D}({n_clip_D/D.size*100:.2f}%), "
          f"centroid={n_clip_C}({n_clip_C/C.size*100:.2f}%), "
          f"query={n_clip_Q}({n_clip_Q/Q.size*100:.2f}%)")

    # IVF & QCR 재계산
    ivf_new, _ = build_ivf(D_new, C_new, pids, is_rel_arr)
    qcr_new    = build_qcr(Q_new, C_new, query_ids, query_df.index, qcr_orig.index.name)

    # DataFrame 구성
    cent_new_df          = pd.DataFrame(C_new, index=cent_df.index,  columns=cent_df.columns)
    cent_new_df.index.name = cent_df.index.name
    doc_new_df           = doc_df.copy()
    doc_new_df[dim_cols] = D_new
    query_new_df         = pd.DataFrame(Q_new, index=query_df.index, columns=query_df.columns)
    query_new_df.index.name = query_df.index.name

    # 저장
    cent_new_df.to_excel( f'{OUT_BASE}/{prefix}centroids_100x128.xlsx')
    doc_new_df.to_excel(  f'{OUT_BASE}/{prefix}doc_embs_12919x128.xlsx')
    query_new_df.to_excel(f'{OUT_BASE}/{prefix}query_embs_96x128.xlsx')
    ivf_new.to_excel(     f'{OUT_BASE}/{prefix}ivf_centroid2pid.xlsx')
    qcr_new.to_excel(     f'{OUT_BASE}/{prefix}query_centroid_ranking.xlsx')
    print(f"  저장 완료: 5개")

    # 비교
    compare_ivf(ivf_new, ivf_orig, prefix)
    compare_qcr(qcr_new, qcr_orig, prefix)

print(f"\n{'='*60}")
print("완료!")
