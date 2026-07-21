"""
Global scale / Clip+scale 두 버전으로 임베딩 변환 후
ivf_centroid2pid, query_centroid_ranking 재계산
"""

import pandas as pd
import numpy as np
from collections import defaultdict

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\00_early_exploration'

# ==========================================================================
# 1. 원본 로드
# ==========================================================================
print("원본 파일 로드 중...")
cent_df  = pd.read_excel(f'{BASE}/centroids_100x128.xlsx',    index_col=0)
doc_df   = pd.read_excel(f'{BASE}/doc_embs_12919x128.xlsx',   index_col=0)
query_df = pd.read_excel(f'{BASE}/query_embs_96x128.xlsx',    index_col=0)
ivf_orig = pd.read_excel(f'{BASE}/ivf_centroid2pid.xlsx',     index_col=0)
qcr_orig = pd.read_excel(f'{BASE}/query_centroid_ranking.xlsx', index_col=0)

dim_cols = [c for c in doc_df.columns if c.startswith('dim_')]

C = cent_df.values.astype(float)              # (100, 128)
D = doc_df[dim_cols].values.astype(float)     # (12919, 128)
Q = query_df.values.astype(float)             # (96, 128)

is_rel_arr  = doc_df['is_relevant'].values.astype(bool)
token_ids   = doc_df.index.tolist()
pids        = np.array([int(t.rsplit('_t', 1)[0]) for t in token_ids], dtype=np.int64)
query_ids   = qcr_orig['query_id'].values
centroid_cols = [f'centroid_{i}' for i in range(100)]

# ==========================================================================
# 2. Scale factor 계산
# ==========================================================================
max_abs_doc     = np.abs(D).max()
max_abs_cent    = np.abs(C).max()
clip_threshold  = max_abs_cent          # centroids max_abs ≈ 0.3402

scale_global = 1.0 / max_abs_doc
scale_clip   = 1.0 / clip_threshold

print(f"\n[Scale factors]")
print(f"  max_abs(doc dims)  = {max_abs_doc:.6f}  → scale_global = {scale_global:.6f}")
print(f"  clip_threshold     = {clip_threshold:.6f}  → scale_clip   = {scale_clip:.6f}")

# ==========================================================================
# 3. 스케일된 임베딩 생성
# ==========================================================================
C_global = C * scale_global
D_global = D * scale_global
Q_global = Q * scale_global

D_clipped = np.clip(D, -clip_threshold, clip_threshold)
C_clip    = C * scale_clip
D_clip    = D_clipped * scale_clip
Q_clip    = Q * scale_clip

print(f"\n[Global 후 범위]")
print(f"  centroids : {C_global.min():.4f} ~ {C_global.max():.4f}  max_abs={np.abs(C_global).max():.4f}")
print(f"  doc_embs  : {D_global.min():.4f} ~ {D_global.max():.4f}  max_abs={np.abs(D_global).max():.4f}")
print(f"  query_embs: {Q_global.min():.4f} ~ {Q_global.max():.4f}  max_abs={np.abs(Q_global).max():.4f}")

print(f"\n[Clip 후 범위]")
print(f"  centroids : {C_clip.min():.4f} ~ {C_clip.max():.4f}  max_abs={np.abs(C_clip).max():.4f}")
print(f"  doc_embs  : {D_clip.min():.4f} ~ {D_clip.max():.4f}  max_abs={np.abs(D_clip).max():.4f}")
print(f"  query_embs: {Q_clip.min():.4f} ~ {Q_clip.max():.4f}  max_abs={np.abs(Q_clip).max():.4f}")

# ==========================================================================
# 4. ivf_centroid2pid 재계산 함수
# ==========================================================================
def build_ivf(D_emb, C_emb, pids_arr, is_rel_arr):
    """각 doc 토큰 → L2 nearest centroid → unique (pid, is_relevant) 수집"""
    # L2 distance: ||d-c||^2 = ||d||^2 + ||c||^2 - 2*d@c^T
    D_norm_sq = (D_emb ** 2).sum(axis=1, keepdims=True)   # (N, 1)
    C_norm_sq = (C_emb ** 2).sum(axis=1)                   # (100,)
    dist_sq   = D_norm_sq + C_norm_sq[np.newaxis, :] - 2 * (D_emb @ C_emb.T)
    assignments = np.argmin(dist_sq, axis=1)               # (N,)

    ivf = defaultdict(dict)   # centroid_id → {pid: is_relevant}
    for c_id, pid, is_rel in zip(assignments, pids_arr, is_rel_arr):
        c_id = int(c_id)
        if pid not in ivf[c_id]:
            ivf[c_id][pid] = bool(is_rel)

    rows = []
    for c_id in sorted(ivf.keys()):
        for pid, is_rel in ivf[c_id].items():
            rows.append({'centroid_id': c_id, 'pid': int(pid), 'is_relevant': is_rel})

    df = pd.DataFrame(rows).set_index('centroid_id')
    df.index.name = 'centroid_id'
    return df, assignments

# ==========================================================================
# 5. query_centroid_ranking 재계산 함수
# ==========================================================================
def build_qcr(Q_emb, C_emb, query_ids_arr, token_index, col_index_name):
    dot = Q_emb @ C_emb.T   # (96, 100)
    df = pd.DataFrame(dot, index=token_index, columns=centroid_cols)
    df.insert(0, 'query_id', query_ids_arr)
    df.index.name = col_index_name
    return df

# ==========================================================================
# 6. Global 버전 생성
# ==========================================================================
print("\n" + "="*60)
print("Global 버전 생성 중...")

ivf_global, assign_global = build_ivf(D_global, C_global, pids, is_rel_arr)
qcr_global = build_qcr(Q_global, C_global, query_ids, query_df.index, qcr_orig.index.name)

# 임베딩 DataFrame
cent_global_df          = pd.DataFrame(C_global, index=cent_df.index,   columns=cent_df.columns)
cent_global_df.index.name = cent_df.index.name
doc_global_df           = doc_df.copy()
doc_global_df[dim_cols] = D_global
query_global_df         = pd.DataFrame(Q_global, index=query_df.index,  columns=query_df.columns)
query_global_df.index.name = query_df.index.name

# 저장
cent_global_df.to_excel(   f'{BASE}/centroids_100x128_global.xlsx')
doc_global_df.to_excel(    f'{BASE}/doc_embs_12919x128_global.xlsx')
query_global_df.to_excel(  f'{BASE}/query_embs_96x128_global.xlsx')
ivf_global.to_excel(       f'{BASE}/ivf_centroid2pid_global.xlsx')
qcr_global.to_excel(       f'{BASE}/query_centroid_ranking_global.xlsx')

print("  저장 완료: 5개 (_global)")

# ==========================================================================
# 7. Clip 버전 생성
# ==========================================================================
print("\nClip 버전 생성 중...")

ivf_clip, assign_clip = build_ivf(D_clip, C_clip, pids, is_rel_arr)
qcr_clip = build_qcr(Q_clip, C_clip, query_ids, query_df.index, qcr_orig.index.name)

cent_clip_df          = pd.DataFrame(C_clip, index=cent_df.index,   columns=cent_df.columns)
cent_clip_df.index.name = cent_df.index.name
doc_clip_df           = doc_df.copy()
doc_clip_df[dim_cols] = D_clip
query_clip_df         = pd.DataFrame(Q_clip, index=query_df.index,  columns=query_df.columns)
query_clip_df.index.name = query_df.index.name

cent_clip_df.to_excel(   f'{BASE}/centroids_100x128_clip.xlsx')
doc_clip_df.to_excel(    f'{BASE}/doc_embs_12919x128_clip.xlsx')
query_clip_df.to_excel(  f'{BASE}/query_embs_96x128_clip.xlsx')
ivf_clip.to_excel(       f'{BASE}/ivf_centroid2pid_clip.xlsx')
qcr_clip.to_excel(       f'{BASE}/query_centroid_ranking_clip.xlsx')

print("  저장 완료: 5개 (_clip)")

# ==========================================================================
# 8. 검증: ivf_global vs 원본
# ==========================================================================
print("\n" + "="*60)
print("[검증] ivf_centroid2pid_global vs 원본")

# 정렬 후 비교
def ivf_to_set(df):
    return set(zip(df.index.tolist(), df['pid'].tolist()))

orig_set   = ivf_to_set(ivf_orig)
global_set = ivf_to_set(ivf_global)

in_orig_not_global = orig_set - global_set
in_global_not_orig = global_set - orig_set

print(f"  원본 rows     : {len(orig_set)}")
print(f"  global rows   : {len(global_set)}")
print(f"  원본에만 있는 pairs: {len(in_orig_not_global)}")
print(f"  global에만 있는 pairs: {len(in_global_not_orig)}")

if len(in_orig_not_global) == 0 and len(in_global_not_orig) == 0:
    print("  ✓ 완전 일치 — uniform scaling으로 L2 ranking 불변 확인")
else:
    print("  ✗ 불일치 존재")
    if in_orig_not_global:
        print("  원본에만:", list(in_orig_not_global)[:5])
    if in_global_not_orig:
        print("  global에만:", list(in_global_not_orig)[:5])

# ==========================================================================
# 9. 검증: ivf_global vs ivf_clip 차이
# ==========================================================================
print("\n[검증] ivf_global vs ivf_clip (clip으로 인한 재배정 변화)")
clip_set = ivf_to_set(ivf_clip)
changed  = orig_set.symmetric_difference(clip_set)
print(f"  clip에서 달라진 (centroid_id, pid) pairs: {len(changed)}")
if changed:
    print(f"  예시: {list(changed)[:5]}")

# ==========================================================================
# 10. 검증: clip된 토큰 수
# ==========================================================================
n_clipped = int((np.abs(D) > clip_threshold).sum())
print(f"\n[검증] clip된 doc dim 값 수: {n_clipped} / {D.size} ({n_clipped/D.size*100:.4f}%)")

# centroid 재배정이 달라진 토큰 수
reassigned = int((assign_global != assign_clip).sum())
print(f"[검증] centroid 재배정이 달라진 토큰 수: {reassigned} / {len(assign_global)}")

print("\n완료!")
