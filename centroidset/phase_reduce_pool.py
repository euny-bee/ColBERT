"""
Phase R. Doc pool 축소 + centroid 재학습 + IVF 재구축
  - 기존 scale_doc_pool_50k.csv에서 서브샘플 (true passage 255개는 항상 포함)
  - 기존 인코딩된 scale_all_vectors.npy / scale_passage_of_token.npy 재사용 (재인코딩 없음)
  - centroid 개수는 호출 시 인자로 지정 (sqrt(N) 비례로 미리 계산해서 넘김)
  사용법: python phase_reduce_pool.py <pool_size> <n_centroids>
    예)   python phase_reduce_pool.py 5000  640
          python phase_reduce_pool.py 20000 1280
  출력:
    scale_doc_pool_{suffix}.csv
    scale_all_vectors_{suffix}.npy
    scale_passage_of_token_{suffix}.npy
    scale_centroids_{suffix}.npy
    scale_ivf_pids_{suffix}.npy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import random
import numpy as np
import pandas as pd
import time
import faiss

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'

POOL_SIZE   = int(sys.argv[1])
N_CENTROIDS = int(sys.argv[2])
SUFFIX      = f'{POOL_SIZE // 1000}k'
KMEANS_ITERS = 20
DIM          = 128
SEED         = 42

print("=" * 60)
print(f"Phase R: pool_size={POOL_SIZE:,}  n_centroids={N_CENTROIDS}  suffix={SUFFIX}")

# ==========================================================================
# 1. 기존 50k pool에서 서브샘플 (true passage 항상 포함)
# ==========================================================================
print("\n[1] Doc pool 서브샘플링...")

pool_50k = pd.read_csv(f'{BASE}/scale_doc_pool_50k.csv')['passage_idx'].tolist()
meta     = pd.read_csv(f'{BASE}/scale_query_meta.csv')
required = sorted(set(meta['true_pid'].tolist()))

assert POOL_SIZE >= len(required), \
    f"pool_size({POOL_SIZE})가 필수 passage 수({len(required)})보다 작습니다"
assert POOL_SIZE <= len(pool_50k), \
    f"pool_size({POOL_SIZE})가 기존 50k pool 크기를 초과합니다"

required_set = set(required)
non_req = sorted([p for p in pool_50k if p not in required_set])

rng = random.Random(SEED)
extra = rng.sample(non_req, POOL_SIZE - len(required))
doc_pool_new = sorted(required + extra)

print(f"  필수(true passage): {len(required)}")
print(f"  추가 서브샘플:       {len(extra)}")
print(f"  최종 pool 크기:      {len(doc_pool_new):,}")

pd.Series(doc_pool_new, name='passage_idx').to_csv(
    f'{BASE}/scale_doc_pool_{SUFFIX}.csv', index=False)

# ==========================================================================
# 2. 기존 인코딩 벡터에서 필터링 (재인코딩 없음)
# ==========================================================================
print("\n[2] 벡터 필터링...")
t0 = time.time()

all_vectors      = np.load(f'{BASE}/scale_all_vectors.npy')       # (n_tokens, 128)
passage_of_token = np.load(f'{BASE}/scale_passage_of_token.npy')  # (n_tokens,)
print(f"  원본 토큰 수: {len(passage_of_token):,}  ({time.time()-t0:.1f}s)")

pool_set = set(doc_pool_new)
mask = np.isin(passage_of_token, list(pool_set))

vectors_new      = all_vectors[mask].astype(np.float32)
passage_token_new = passage_of_token[mask]

print(f"  필터링 후 토큰 수: {len(passage_token_new):,}")
assert set(np.unique(passage_token_new).tolist()) == pool_set, \
    "필터링 후 pool에 포함된 passage와 토큰의 passage 집합이 불일치합니다"

np.save(f'{BASE}/scale_all_vectors_{SUFFIX}.npy', vectors_new)
np.save(f'{BASE}/scale_passage_of_token_{SUFFIX}.npy', passage_token_new)
print(f"  저장: scale_all_vectors_{SUFFIX}.npy, scale_passage_of_token_{SUFFIX}.npy")

# ==========================================================================
# 3. k-means 재학습 (FAISS GPU)
# ==========================================================================
print()
print("=" * 60)
print(f"[3] k-means {N_CENTROIDS} centroid 재학습 중...")
t0 = time.time()

kmeans = faiss.Kmeans(DIM, N_CENTROIDS, niter=KMEANS_ITERS,
                      verbose=True, gpu=True, seed=SEED)
kmeans.train(vectors_new)
centroids_new = kmeans.centroids  # (N_CENTROIDS, 128)

print(f"  k-means 완료 ({time.time()-t0:.1f}s)")
print(f"  centroids shape: {centroids_new.shape}")

np.save(f'{BASE}/scale_centroids_{SUFFIX}.npy', centroids_new)
print(f"  저장: scale_centroids_{SUFFIX}.npy")

# ==========================================================================
# 4. IVF 재구축: token -> centroid 할당, centroid -> unique passage list
# ==========================================================================
print()
print("=" * 60)
print("[4] IVF 구축 중...")
t0 = time.time()

total_tokens = len(vectors_new)
index_flat = faiss.IndexFlatL2(DIM)
index_flat.add(centroids_new.astype(np.float32))

BATCH = 100_000
all_assignments = np.zeros(total_tokens, dtype=np.int32)

for start in range(0, total_tokens, BATCH):
    end = min(start + BATCH, total_tokens)
    _, I = index_flat.search(vectors_new[start:end], 1)
    all_assignments[start:end] = I[:, 0]

print(f"  할당 완료 ({time.time()-t0:.1f}s)")

ivf_pids = [[] for _ in range(N_CENTROIDS)]
for tok_i in range(total_tokens):
    c_id = int(all_assignments[tok_i])
    ivf_pids[c_id].append(int(passage_token_new[tok_i]))

ivf_pids_np = np.empty(N_CENTROIDS, dtype=object)
for c in range(N_CENTROIDS):
    ivf_pids_np[c] = np.array(list(dict.fromkeys(ivf_pids[c])), dtype=np.int32)

sizes = [len(ivf_pids_np[c]) for c in range(N_CENTROIDS)]
print(f"  centroid당 평균 unique passages: {np.mean(sizes):.1f}")
print(f"  centroid당 최대: {np.max(sizes)}  최소: {np.min(sizes)}")

np.save(f'{BASE}/scale_ivf_pids_{SUFFIX}.npy', ivf_pids_np)
print(f"  저장: scale_ivf_pids_{SUFFIX}.npy")

print()
print("=" * 60)
print(f"Phase R 완료! (suffix={SUFFIX})")
