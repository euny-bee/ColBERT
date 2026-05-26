import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\nmdl-khb\ColBERT")

from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig

# ── 경로 설정 ──────────────────────────────────────────────────────────────
COLLECTION_PATH = r"D:\msmarco\full\collection.tsv"
QUERIES_PATH    = r"C:\Users\nmdl-khb\ColBERT\data\msmarco\queries.dev.tsv"
QRELS_PATH      = r"C:\Users\nmdl-khb\ColBERT\data\msmarco\qrels.dev.small.tsv"
OUT_DIR         = r"C:\Users\nmdl-khb\ColBERT\centroidset 만들기"

CHECKPOINT = "colbert-ir/colbertv2.0"
N_CENTROIDS = 100
N_PASSAGES  = 200
SEED        = 42
TOP_N       = 10   # centroid selection 검증용 top-n

random.seed(SEED)
np.random.seed(SEED)

# ── Step 1: qrels 로드 & 쿼리 3개 선택 ────────────────────────────────────
print("=== Step 1: 쿼리 선택 ===")
qid2pids = {}
with open(QRELS_PATH, encoding="utf-8") as f:
    for line in f:
        qid, _, pid, _ = line.strip().split()
        qid2pids.setdefault(qid, []).append(pid)

# 재현성을 위해 sorted 후 첫 3개
selected_qids = sorted(qid2pids.keys())[:3]
relevant_pids = set()
for qid in selected_qids:
    relevant_pids.update(qid2pids[qid])
print(f"선택된 qid: {selected_qids}")
print(f"정답 pid: {sorted(relevant_pids)}")

# ── Step 2: 쿼리 텍스트 로드 ───────────────────────────────────────────────
print("\n=== Step 2: 쿼리 텍스트 로드 ===")
qid2text = {}
with open(QUERIES_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            qid2text[parts[0]] = parts[1]

selected_queries = [qid2text[qid] for qid in selected_qids]
for qid, qt in zip(selected_qids, selected_queries):
    print(f"  [{qid}] {qt}")

# ── Step 3: Passage 수집 ───────────────────────────────────────────────────
print(f"\n=== Step 3: Passage 수집 (정답 {len(relevant_pids)}개 + 랜덤 ~{N_PASSAGES}개) ===")
pid2text = {}
all_pids_in_collection = []

with open(COLLECTION_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            pid, text = parts[0], parts[1]
            all_pids_in_collection.append(pid)
            if pid in relevant_pids:
                pid2text[pid] = text

print(f"  정답 passage {len(pid2text)}개 수집 완료")

# 랜덤 샘플링 (정답 제외)
remaining = [p for p in all_pids_in_collection if p not in relevant_pids]
n_random = N_PASSAGES - len(pid2text)
random_pids = random.sample(remaining, n_random)

# 랜덤 passage 텍스트 로드 (collection 재순회 대신 인덱스 활용)
random_pid_set = set(random_pids)
with open(COLLECTION_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2 and parts[0] in random_pid_set:
            pid2text[parts[0]] = parts[1]

# 정답 passage가 앞에 오도록 정렬
ordered_pids = sorted(relevant_pids) + random_pids
passages = [pid2text[p] for p in ordered_pids]
print(f"  총 passage 수: {len(passages)}")

# ── Step 4: 모델 로드 & Document 인코딩 ────────────────────────────────────
print(f"\n=== Step 4: Document 인코딩 ===")
config = ColBERTConfig(query_maxlen=32, doc_maxlen=220, dim=128)
checkpoint = Checkpoint(CHECKPOINT, colbert_config=config)

# keep_dims=False → (list_of_tensors,) 튜플 반환; colbert.py가 내부에서 CPU 이동 처리
D_list = checkpoint.docFromText(
    passages, bsize=32, keep_dims=False, to_cpu=False, showprogress=True
)[0]
doclens = [d.shape[0] for d in D_list]
doc_embs = torch.cat([d.float() for d in D_list], dim=0)
print(f"  doc_embs shape: {doc_embs.shape}")  # (~13400, 128)
print(f"  doclens 합계: {sum(doclens)}")

# ── Step 5: K-means (k=100) ────────────────────────────────────────────────
print(f"\n=== Step 5: K-means (k={N_CENTROIDS}) ===")
try:
    import faiss
    use_gpu = torch.cuda.is_available()
    kmeans = faiss.Kmeans(128, N_CENTROIDS, niter=20, gpu=use_gpu, verbose=True, seed=123)
    kmeans.train(doc_embs.numpy())
    centroids = torch.from_numpy(kmeans.centroids)
except ImportError:
    print("  FAISS 없음 → sklearn 사용")
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_CENTROIDS, n_init=5, max_iter=100, random_state=123, verbose=1)
    km.fit(doc_embs.numpy())
    centroids = torch.from_numpy(km.cluster_centers_.astype(np.float32))

centroids = F.normalize(centroids.float(), dim=-1)
print(f"  centroids shape: {centroids.shape}")  # (100, 128)

# ── Step 6: Query 인코딩 ───────────────────────────────────────────────────
print(f"\n=== Step 6: Query 인코딩 ===")
Q = checkpoint.queryFromText(selected_queries, bsize=3, to_cpu=True)
Q_flat = Q.view(-1, 128).float()
print(f"  Q shape: {Q.shape}")         # (3, 32, 128)
print(f"  Q_flat shape: {Q_flat.shape}")  # (96, 128)

# ── Step 7: Centroid Selection 검증 ───────────────────────────────────────
print(f"\n=== Step 7: Centroid Selection 검증 (top-{TOP_N}) ===")
scores = centroids @ Q_flat.T           # (100, 96)
topn_indices = scores.topk(TOP_N, dim=0).indices  # (TOP_N, 96)

# 정답 passage token들의 nearest centroid 계산
n_relevant = len(relevant_pids)
rel_doclens = doclens[:n_relevant]
rel_embs = doc_embs[:sum(rel_doclens)]

nearest_centroids = (centroids @ rel_embs.T).argmax(dim=0)  # (total_rel_tokens,)

# hit: nearest centroid가 어떤 query token의 top-n 안에 있는지
topn_set = set(topn_indices.flatten().tolist())
hits = sum(1 for c in nearest_centroids.tolist() if c in topn_set)
total = len(nearest_centroids)
hit_rate = hits / total if total > 0 else 0.0

print(f"  정답 passage token 수: {total}")
print(f"  top-{TOP_N} hit: {hits}/{total} = {hit_rate:.1%}")

# top-n별 hit rate
result = {
    "selected_qids": selected_qids,
    "selected_queries": selected_queries,
    "relevant_pids": sorted(relevant_pids),
    "n_passages": len(passages),
    "n_centroids": N_CENTROIDS,
    "doc_embs_shape": list(doc_embs.shape),
    "query_embs_shape": list(Q_flat.shape),
    "hit_rates": {}
}

for n in [5, 10, 20, 30, 50]:
    tn = scores.topk(min(n, N_CENTROIDS), dim=0).indices
    tn_set = set(tn.flatten().tolist())
    h = sum(1 for c in nearest_centroids.tolist() if c in tn_set)
    result["hit_rates"][f"top{n}"] = round(h / total, 4) if total > 0 else 0.0
    print(f"  top-{n:2d} hit rate: {h}/{total} = {h/total:.1%}")

# ── Step 8: 저장 ───────────────────────────────────────────────────────────
print(f"\n=== Step 8: 저장 ===")
os.makedirs(OUT_DIR, exist_ok=True)

np.save(os.path.join(OUT_DIR, "doc_embs_12919x128.npy"), doc_embs.numpy())
with open(os.path.join(OUT_DIR, "doclens.json"), "w") as f:
    json.dump(doclens, f)
np.save(os.path.join(OUT_DIR, "centroids_100x128.npy"), centroids.numpy())
np.savetxt(os.path.join(OUT_DIR, "centroids_100x128.csv"), centroids.numpy(), delimiter=",")

np.save(os.path.join(OUT_DIR, "query_embs_96x128.npy"), Q_flat.numpy())
np.savetxt(os.path.join(OUT_DIR, "query_embs_96x128.csv"), Q_flat.numpy(), delimiter=",")

with open(os.path.join(OUT_DIR, "centroid_selection_result.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"  저장 완료: {OUT_DIR}")
print(f"  - centroids_100x128.npy / .csv")
print(f"  - query_embs_96x128.npy / .csv")
print(f"  - centroid_selection_result.json")
print("\n=== 완료 ===")
