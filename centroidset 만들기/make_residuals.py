import os
import json
import random
import numpy as np
import torch
import pandas as pd

OUT_DIR         = r"C:\Users\nmdl-khb\ColBERT\centroidset 만들기"
QRELS_PATH      = r"C:\Users\nmdl-khb\ColBERT\data\msmarco\qrels.dev.small.tsv"
COLLECTION_PATH = r"D:\msmarco\full\collection.tsv"
SEED = 42
random.seed(SEED)

# ── Step 1: 데이터 로드 ────────────────────────────────────────────────────
print("=== Step 1: 데이터 로드 ===")
doc_embs  = torch.from_numpy(np.load(os.path.join(OUT_DIR, "doc_embs_12919x128.npy")))   # (12919, 128)
centroids = torch.from_numpy(np.load(os.path.join(OUT_DIR, "centroids_100x128.npy")))    # (100, 128)
with open(os.path.join(OUT_DIR, "doclens.json"), encoding="utf-8") as f:
    doclens = json.load(f)
print(f"  doc_embs:  {doc_embs.shape}")
print(f"  centroids: {centroids.shape}")
print(f"  doclens:   {len(doclens)}개, 합계={sum(doclens)}")

# ordered_pids 재현
qid2pids = {}
with open(QRELS_PATH, encoding="utf-8") as f:
    for line in f:
        qid, _, pid, _ = line.strip().split()
        qid2pids.setdefault(qid, []).append(pid)
selected_qids = sorted(qid2pids.keys())[:3]
relevant_pids = set()
for qid in selected_qids:
    relevant_pids.update(qid2pids[qid])
all_pids = []
with open(COLLECTION_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            all_pids.append(parts[0])
remaining = [p for p in all_pids if p not in relevant_pids]
random_pids = random.sample(remaining, 200 - len(relevant_pids))
ordered_pids = sorted(relevant_pids) + random_pids

# 행 이름
row_names = []
for pid_idx, length in enumerate(doclens):
    pid = ordered_pids[pid_idx]
    for t in range(length):
        row_names.append(f"{pid}_t{t}")
col_names = [f"dim_{i}" for i in range(128)]

# ── Step 2: token → nearest centroid ──────────────────────────────────────
print("\n=== Step 2: nearest centroid 계산 ===")
scores = centroids @ doc_embs.T              # (100, 12919)
token2centroid = scores.argmax(dim=0)        # (12919,)
nearest_centroids = centroids[token2centroid]  # (12919, 128)

# ── Step 3: float32 잔차 ──────────────────────────────────────────────────
print("\n=== Step 3: float32 잔차 계산 ===")
residuals_float = doc_embs - nearest_centroids   # (12919, 128)
print(f"  residuals_float shape: {residuals_float.shape}")
print(f"  mean abs residual: {residuals_float.abs().mean():.4f}")
print(f"  max abs residual:  {residuals_float.abs().max():.4f}")

# ── Step 4: 2-bit 양자화 (ColBERT 원본 방식) ──────────────────────────────
print("\n=== Step 4: 2-bit 양자화 ===")
# bucket_cutoffs: 25%, 50%, 75% 분위수 (3개 경계 → 4개 버킷)
# bucket_weights: 12.5%, 37.5%, 62.5%, 87.5% 분위수 (각 버킷 중심값)
num_options = 4  # 2^2
quantiles = torch.arange(0, num_options) * (1.0 / num_options)
cutoff_quantiles  = quantiles[1:]                          # [0.25, 0.50, 0.75]
weight_quantiles  = quantiles + (0.5 / num_options)        # [0.125, 0.375, 0.625, 0.875]

bucket_cutoffs  = residuals_float.float().quantile(cutoff_quantiles)   # (3,)
bucket_weights  = residuals_float.float().quantile(weight_quantiles)   # (4,)

print(f"  bucket_cutoffs:  {[round(x,4) for x in bucket_cutoffs.tolist()]}")
print(f"  bucket_weights:  {[round(x,4) for x in bucket_weights.tolist()]}")

# 각 차원값 → 버킷 코드 (0/1/2/3)
codes = torch.bucketize(residuals_float.float(), bucket_cutoffs)  # (12919, 128) uint8 범위
# 코드 → 역양자화값 (bucket_weights)
residuals_2bit = bucket_weights[codes]                              # (12919, 128)

print(f"  2-bit codes unique: {codes.unique().tolist()}")
print(f"  mean abs 양자화오차: {(residuals_float - residuals_2bit).abs().mean():.4f}")

# ── Step 5: 저장 ───────────────────────────────────────────────────────────
print("\n=== Step 5: 저장 ===")

# --- float32 잔차 ---
df_float = pd.DataFrame(residuals_float.numpy(), index=row_names, columns=col_names)
df_float.index.name = "token_id"
df_float.to_csv(os.path.join(OUT_DIR, "residuals_float32.csv"))
df_float.to_excel(os.path.join(OUT_DIR, "residuals_float32.xlsx"))
print(f"  residuals_float32.csv / .xlsx 저장 완료 {df_float.shape}")

# --- 2-bit 잔차 (역양자화된 버킷 중심값) ---
df_2bit = pd.DataFrame(residuals_2bit.numpy(), index=row_names, columns=col_names)
df_2bit.index.name = "token_id"
df_2bit.to_csv(os.path.join(OUT_DIR, "residuals_2bit.csv"))
df_2bit.to_excel(os.path.join(OUT_DIR, "residuals_2bit.xlsx"))
print(f"  residuals_2bit.csv / .xlsx 저장 완료 {df_2bit.shape}")

# 버킷 정보 저장
bucket_info = {
    "bucket_cutoffs": [round(x, 6) for x in bucket_cutoffs.tolist()],
    "bucket_weights": [round(x, 6) for x in bucket_weights.tolist()],
    "description": "2-bit quantization: cutoffs for bucketize(), weights are dequantized values (0->w0, 1->w1, 2->w2, 3->w3)"
}
with open(os.path.join(OUT_DIR, "bucket_info.json"), "w", encoding="utf-8") as f:
    json.dump(bucket_info, f, indent=2)
print(f"  bucket_info.json 저장 완료")

print("\n=== 완료 ===")
