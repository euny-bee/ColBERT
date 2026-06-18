"""
Phase 2. 인덱스 구조 구성
  [2-1] 50K docs 토큰 벡터 복원 → k-means로 32K centroid 구축
  [2-2] IVF 구축 (centroid_id → [token_global_idx, passage_idx])
"""

import os, json, time
import numpy as np
import pandas as pd
import torch
import faiss

BASE     = r'C:\Users\nmdl-khb\ColBERT\centroidset'
IDX_BASE = r'C:\Users\nmdl-khb\ColBERT\experiments\msmarco_1m\indexes\1m.analog'

N_CENTROIDS  = 32768   # 2^15
KMEANS_ITERS = 20
DIM          = 128

# ==========================================================================
# 데이터 로드
# ==========================================================================
print("=" * 60)
print("기본 데이터 로드 중...")

# 131K centroids (msmarco_1m)
C_big = torch.load(f'{IDX_BASE}/centroids.pt', map_location='cpu').float().numpy()
print(f"  131K centroids: {C_big.shape}")

# 50K doc pool
doc_pool = pd.read_csv(f'{BASE}/scale_doc_pool_50k.csv')['passage_idx'].tolist()
doc_pool_set = set(doc_pool)
print(f"  doc pool 크기: {len(doc_pool):,}")

# 전체 doclens (passage별 토큰 수)
all_doclens = []
for i in range(44):
    with open(f'{IDX_BASE}/doclens.{i}.json') as f:
        all_doclens.extend(json.load(f))
print(f"  전체 passage 수: {len(all_doclens):,}")

# passage별 embedding offset 계산
doclen_arr    = np.array(all_doclens, dtype=np.int32)
emb_offsets   = np.concatenate([[0], np.cumsum(doclen_arr)])   # (n_passage+1,)

# shard별 metadata
shard_meta = []
for i in range(44):
    with open(f'{IDX_BASE}/{i}.metadata.json') as f:
        shard_meta.append(json.load(f))

# ==========================================================================
# [2-1] 50K passage 토큰 벡터 복원
# ==========================================================================
print()
print("=" * 60)
print("[2-1] 50K passage 토큰 벡터 복원 중...")
t0 = time.time()

# 50K passage의 전체 token embedding global indices 수집
target_passages = sorted(doc_pool)
token_global_idxs = []    # 각 토큰의 global embedding index
passage_of_token  = []    # 각 토큰이 속한 passage_idx

for pid in target_passages:
    start = int(emb_offsets[pid])
    end   = int(emb_offsets[pid + 1])
    token_global_idxs.extend(range(start, end))
    passage_of_token.extend([pid] * (end - start))

token_global_idxs = np.array(token_global_idxs, dtype=np.int64)
passage_of_token  = np.array(passage_of_token,  dtype=np.int32)
total_tokens = len(token_global_idxs)
print(f"  총 토큰 수: {total_tokens:,}")

# 토큰 벡터 복원: v = C_big[code] + residual
# shard별로 처리 (메모리 효율)
all_vectors = np.zeros((total_tokens, DIM), dtype=np.float32)

for shard_i in range(44):
    meta = shard_meta[shard_i]
    emb_offset_shard = meta['embedding_offset']
    n_emb_shard      = meta['num_embeddings']
    shard_global_range = range(emb_offset_shard, emb_offset_shard + n_emb_shard)

    # 이 shard에 해당하는 token indices
    mask = ((token_global_idxs >= emb_offset_shard) &
            (token_global_idxs < emb_offset_shard + n_emb_shard))
    if not mask.any():
        continue

    local_idxs    = (token_global_idxs[mask] - emb_offset_shard).astype(np.int64)
    global_pos    = np.where(mask)[0]

    codes = torch.load(f'{IDX_BASE}/{shard_i}.codes.pt',
                       map_location='cpu').numpy().astype(np.int32)
    resid = torch.load(f'{IDX_BASE}/{shard_i}.residuals.pt',
                       map_location='cpu').float().numpy()

    sel_codes = codes[local_idxs]
    sel_resid = resid[local_idxs]
    sel_vecs  = C_big[sel_codes] + sel_resid    # (n, 128)

    all_vectors[global_pos] = sel_vecs.astype(np.float32)

    print(f"  shard {shard_i:2d}: {mask.sum():6,} 토큰 처리")

print(f"  복원 완료 ({time.time()-t0:.1f}s)")
print(f"  벡터 shape: {all_vectors.shape}")

# ==========================================================================
# k-means로 32K centroid 구축 (FAISS GPU)
# ==========================================================================
print()
print("=" * 60)
print(f"[2-1] k-means {N_CENTROIDS:,}개 centroid 구축 중 (FAISS GPU)...")
t0 = time.time()

kmeans = faiss.Kmeans(DIM, N_CENTROIDS, niter=KMEANS_ITERS,
                      verbose=True, gpu=True, seed=42)
kmeans.train(all_vectors)
centroids_32k = kmeans.centroids   # (32768, 128)
print(f"  k-means 완료 ({time.time()-t0:.1f}s)")
print(f"  centroids shape: {centroids_32k.shape}")

# centroid 저장
np.save(f'{BASE}/scale_centroids_32k.npy', centroids_32k)
print(f"  저장: scale_centroids_32k.npy")

# ==========================================================================
# [2-2] IVF 구축
# ==========================================================================
print()
print("=" * 60)
print("[2-2] IVF 구축 중...")
t0 = time.time()

# 각 토큰을 가장 가까운 centroid에 할당 (FAISS)
index_flat = faiss.IndexFlatL2(DIM)
index_flat.add(centroids_32k.astype(np.float32))

# 배치로 처리
BATCH = 100_000
all_assignments = np.zeros(total_tokens, dtype=np.int32)

for start in range(0, total_tokens, BATCH):
    end = min(start + BATCH, total_tokens)
    _, I = index_flat.search(all_vectors[start:end], 1)
    all_assignments[start:end] = I[:, 0]

print(f"  centroid 할당 완료 ({time.time()-t0:.1f}s)")

# IVF: centroid_id → [token_local_idx, passage_idx]
ivf = {}   # centroid_id → list of (token_local_idx, passage_idx)
for tok_i in range(total_tokens):
    c_id = int(all_assignments[tok_i])
    pid  = int(passage_of_token[tok_i])
    if c_id not in ivf:
        ivf[c_id] = {'pids': [], 'tok_idxs': []}
    ivf[c_id]['pids'].append(pid)
    ivf[c_id]['tok_idxs'].append(tok_i)

# IVF 통계
ivf_sizes = [len(v['pids']) for v in ivf.values()]
print(f"  활성 centroid 수: {len(ivf):,} / {N_CENTROIDS:,}")
print(f"  centroid당 평균 토큰: {np.mean(ivf_sizes):.1f}")
print(f"  centroid당 최대 토큰: {np.max(ivf_sizes)}")
print(f"  centroid당 최소 토큰: {np.min(ivf_sizes)}")

# IVF 저장 (numpy 형태로 효율적 저장)
ivf_pids    = [np.array(ivf[c]['pids'],     dtype=np.int32) if c in ivf else np.array([], dtype=np.int32)
               for c in range(N_CENTROIDS)]
ivf_tokidxs = [np.array(ivf[c]['tok_idxs'], dtype=np.int32) if c in ivf else np.array([], dtype=np.int32)
               for c in range(N_CENTROIDS)]

np.save(f'{BASE}/scale_all_vectors.npy',      all_vectors)
np.save(f'{BASE}/scale_assignments.npy',      all_assignments)
np.save(f'{BASE}/scale_passage_of_token.npy', passage_of_token)
np.save(f'{BASE}/scale_ivf_pids.npy',         np.array(ivf_pids,    dtype=object))
np.save(f'{BASE}/scale_ivf_tokidxs.npy',      np.array(ivf_tokidxs, dtype=object))

print()
print("저장 완료:")
print(f"  scale_centroids_32k.npy     → {centroids_32k.shape}")
print(f"  scale_all_vectors.npy       → {all_vectors.shape}")
print(f"  scale_assignments.npy       → {all_assignments.shape}")
print(f"  scale_passage_of_token.npy  → {passage_of_token.shape}")
print(f"  scale_ivf_pids/tokidxs.npy  → {N_CENTROIDS} centroid lists")
print()
print("Phase 2 완료!")
