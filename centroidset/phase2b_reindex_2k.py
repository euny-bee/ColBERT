"""
Phase 2b. 2048 centroid 재구축
  scale_all_vectors.npy 재사용 (shard 복원 생략)
  k-means 2048 + IVF 재구축
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import time
import faiss

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'

N_CENTROIDS  = 2048
KMEANS_ITERS = 20
DIM          = 128

# ==========================================================================
# 데이터 로드
# ==========================================================================
print("=" * 60)
print("데이터 로드 중...")
t0 = time.time()

all_vectors      = np.load(f'{BASE}/scale_all_vectors.npy')       # (3378234, 128)
passage_of_token = np.load(f'{BASE}/scale_passage_of_token.npy')  # (3378234,)

total_tokens = len(all_vectors)
print(f"  all_vectors:      {all_vectors.shape}  ({time.time()-t0:.1f}s)")
print(f"  passage_of_token: {passage_of_token.shape}")

# ==========================================================================
# k-means 2048 centroid (FAISS GPU)
# ==========================================================================
print()
print("=" * 60)
print(f"k-means {N_CENTROIDS} centroid 구축 중 (FAISS GPU)...")
t0 = time.time()

kmeans = faiss.Kmeans(DIM, N_CENTROIDS, niter=KMEANS_ITERS,
                      verbose=True, gpu=True, seed=42)
kmeans.train(all_vectors)
centroids_2k = kmeans.centroids  # (2048, 128)

print(f"  k-means 완료 ({time.time()-t0:.1f}s)")
print(f"  centroids shape: {centroids_2k.shape}")

np.save(f'{BASE}/scale_centroids_2k.npy', centroids_2k)
print(f"  저장: scale_centroids_2k.npy")

# ==========================================================================
# IVF 구축: token -> centroid 할당
# ==========================================================================
print()
print("=" * 60)
print("IVF 구축 중 (token -> centroid 할당)...")
t0 = time.time()

index_flat = faiss.IndexFlatL2(DIM)
index_flat.add(centroids_2k.astype(np.float32))

BATCH = 100_000
all_assignments = np.zeros(total_tokens, dtype=np.int32)

for start in range(0, total_tokens, BATCH):
    end = min(start + BATCH, total_tokens)
    _, I = index_flat.search(all_vectors[start:end], 1)
    all_assignments[start:end] = I[:, 0]
    if start % 500_000 == 0:
        print(f"  {start:,} / {total_tokens:,} ({start/total_tokens*100:.1f}%)")

print(f"  할당 완료 ({time.time()-t0:.1f}s)")

# ==========================================================================
# IVF pids 생성: centroid_id -> unique passage list
# ==========================================================================
print()
print("IVF pids 생성 중...")
t0 = time.time()

ivf_pids = [[] for _ in range(N_CENTROIDS)]
for tok_i in range(total_tokens):
    c_id = int(all_assignments[tok_i])
    ivf_pids[c_id].append(int(passage_of_token[tok_i]))

# unique passage IDs per centroid
ivf_pids_np = np.empty(N_CENTROIDS, dtype=object)
for c in range(N_CENTROIDS):
    ivf_pids_np[c] = np.array(list(dict.fromkeys(ivf_pids[c])), dtype=np.int32)

print(f"  완료 ({time.time()-t0:.1f}s)")

# 통계
sizes = [len(ivf_pids_np[c]) for c in range(N_CENTROIDS)]
print(f"  centroid당 평균 unique passages: {np.mean(sizes):.1f}")
print(f"  centroid당 최대: {np.max(sizes)}")
print(f"  centroid당 최소: {np.min(sizes)}")

np.save(f'{BASE}/scale_ivf_pids_2k.npy', ivf_pids_np)
print(f"  저장: scale_ivf_pids_2k.npy")

# ==========================================================================
# 완료
# ==========================================================================
print()
print("=" * 60)
print("저장 완료:")
print(f"  scale_centroids_2k.npy  -> {centroids_2k.shape}")
print(f"  scale_ivf_pids_2k.npy   -> {N_CENTROIDS} centroid lists")
print()
print("Phase 2b 완료!")
