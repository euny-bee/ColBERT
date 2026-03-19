#!/usr/bin/env python3
"""
encode_and_compare.py
---------------------
인코딩 1회 → k-means 1회 → residual 1회 계산
    ├─ float16 그대로 저장  → colbert_float16_correct.csv
    └─ 2-bit quantize 적용 → colbert_2bit_correct.csv

순서:
  Step 1. 샘플 패시지 인코딩 (k-means 학습용, ~5000개)
  Step 2. FAISS k-means → centroids 학습
  Step 3. 샘플 residual → bucket_cutoffs / bucket_weights 학습
  Step 4. 첫 500개 패시지 인코딩 (CSV 대상)
  Step 5. 동일 embedding 기반으로 float16 잔차 + 2-bit 잔차 동시 계산
  Step 6. CSV 저장
"""

import os
import sys
import csv
import math
import time
import random
import torch
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────
COLBERT_DIR  = os.path.expanduser("~/ColBERT")
COLLECTION   = os.path.join(COLBERT_DIR, "data/msmarco/subset/collection.tsv")
CHECKPOINT   = "colbert-ir/colbertv2.0"
OUT_DIR      = COLBERT_DIR

N_CSV        = 500   # CSV에 저장할 패시지 수
BSIZE        = 32    # 인코딩 배치 크기
NBITS        = 2     # 양자화 비트 수 (2-bit → 4개 bucket)
DIM          = 128   # 임베딩 차원
KMEANS_NITERS = 20   # k-means 반복

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 패시지 로드
# ============================================================
def load_passages(path, n=None):
    passages = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            parts = line.rstrip("\n").split("\t", 1)
            passages.append(parts[1] if len(parts) > 1 else "")
    return passages


# ============================================================
# 인코딩: passages → (total_tokens, 128) float16, doclens
# ============================================================
def encode(encoder, passages):
    embs, doclens = encoder.encode_passages(passages)
    return embs.half(), doclens   # float16 그대로 유지


# ============================================================
# k-means (FAISS) — FAISS는 float32 필수, 그 외는 float16
# ============================================================
def train_kmeans(sample_embs, num_partitions, niters=20):
    import faiss
    log(f"  k-means: {sample_embs.shape[0]:,} 샘플 → {num_partitions:,} centroids")
    sample_np = sample_embs.float().numpy().astype(np.float32)  # FAISS 전달용만 float32
    use_gpu = torch.cuda.is_available()
    km = faiss.Kmeans(DIM, num_partitions, niter=niters, verbose=False, gpu=use_gpu)
    km.train(sample_np)
    return torch.from_numpy(km.centroids).half()   # (num_partitions, 128) float16


# ============================================================
# bucket_cutoffs / bucket_weights 학습
# ============================================================
def train_buckets(residuals):
    """
    residuals: (N, 128) float16
    4분위 경계값(cutoffs)과 각 구간 중앙값(weights) → float16 반환
    """
    # quantile은 float32 필요, 통계 계산용으로만 변환
    flat = residuals.float().flatten()

    # quantile()은 2^24 이상 입력 불가 → 최대 5M 샘플링
    max_samples = 5_000_000
    if flat.numel() > max_samples:
        idx = torch.randperm(flat.numel())[:max_samples]
        flat = flat[idx]

    num_buckets = 2 ** NBITS   # 4

    cutoff_qs = torch.linspace(0, 1, num_buckets + 1)[1:-1]
    bucket_cutoffs = torch.quantile(flat, cutoff_qs).half()   # float16

    bounds = torch.cat([torch.tensor([-1e4], dtype=torch.float16),
                        bucket_cutoffs,
                        torch.tensor([1e4],  dtype=torch.float16)])
    flat16 = flat.half()
    bucket_weights = torch.stack([
        flat16[(flat16 >= bounds[i]) & (flat16 < bounds[i + 1])].median()
        for i in range(num_buckets)
    ])   # float16
    return bucket_cutoffs, bucket_weights


# ============================================================
# nearest centroid 탐색 (dot product, float16)
# ============================================================
def find_codes(embs, centroids):
    # embs, centroids 모두 float16
    codes = []
    bsize = 1 << 16
    for batch in embs.split(bsize):
        sim = centroids @ batch.T          # float16 @ float16
        codes.append(sim.max(dim=0).indices.cpu())
    return torch.cat(codes)               # (N,)


# ============================================================
# 2-bit 양자화 (pack into uint8)
# ============================================================
def binarize(residuals, bucket_cutoffs):
    """residuals: (N, 128) float16 → (N, 32) uint8"""
    # bucketize는 float32 필요
    res = torch.bucketize(residuals.float(), bucket_cutoffs.float()).to(torch.uint8)
    arange_bits = torch.arange(0, NBITS, dtype=torch.uint8)
    res = res.unsqueeze(-1).expand(*res.size(), NBITS)
    res = res >> arange_bits
    res = res & 1
    packed_np = np.packbits(np.asarray(res.contiguous().flatten()))
    packed = torch.as_tensor(packed_np, dtype=torch.uint8)
    return packed.reshape(residuals.size(0), DIM // 8 * NBITS)   # (N, 32)


# ============================================================
# 2-bit 디코딩 (uint8 → float16)
# ============================================================
def decode_2bit(packed, bucket_weights):
    """packed: (N, 32) uint8 → (N, 128) float16"""
    r = packed.int()
    v0 = ((r >> 7) & 1) | (((r >> 6) & 1) << 1)
    v1 = ((r >> 5) & 1) | (((r >> 4) & 1) << 1)
    v2 = ((r >> 3) & 1) | (((r >> 2) & 1) << 1)
    v3 = ((r >> 1) & 1) | (((r >> 0) & 1) << 1)
    indices = torch.stack([v0, v1, v2, v3], dim=-1).reshape(packed.size(0), DIM)
    return bucket_weights.half()[indices.long()]   # float16


# ============================================================
# CSV 저장
# ============================================================
def save_csv(path, header, rows):
    log(f"Saving {os.path.basename(path)} ({len(rows):,} rows)...")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    mb = os.path.getsize(path) / 1024 / 1024
    log(f"  → {path}  ({mb:.1f} MB)")


def build_header():
    h = ["passage_id", "token_idx"]
    h += [f"centroid_dim_{i}" for i in range(DIM)]
    h += [f"residual_dim_{i}" for i in range(DIM)]
    return h


def build_rows(passage_ids, token_idxs, centroid_vecs, residuals):
    rows = []
    for i in range(len(passage_ids)):
        row = [passage_ids[i], token_idxs[i]]
        row += centroid_vecs[i].tolist()
        row += residuals[i].tolist()
        rows.append(row)
    return rows


# ============================================================
# main
# ============================================================
def main():
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.indexing.collection_encoder import CollectionEncoder
    from colbert.infra import ColBERTConfig, Run, RunConfig

    config = ColBERTConfig(
        nbits=NBITS, doc_maxlen=220, query_maxlen=32,
        index_bsize=BSIZE, avoid_fork_if_possible=True,
    )

    # ── 전체 패시지 로드 ─────────────────────────────────────
    log("Loading all passages...")
    all_passages = load_passages(COLLECTION)
    n_total = len(all_passages)
    log(f"  {n_total:,} passages")

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        log("Loading checkpoint...")
        checkpoint = Checkpoint(CHECKPOINT, colbert_config=config)
        if torch.cuda.is_available():
            checkpoint = checkpoint.cuda()
        encoder = CollectionEncoder(config, checkpoint)

        # ── Step 1: 샘플 인코딩 (k-means 학습용) ─────────────
        # ColBERT 기본: sqrt(120 * N) 개 패시지 샘플
        n_sample_pids = min(int(math.sqrt(120 * n_total)) + 1, n_total)
        random.seed(42)
        sample_pids = sorted(random.sample(range(n_total), n_sample_pids))
        sample_passages = [all_passages[i] for i in sample_pids]

        log(f"Step 1: Encoding {len(sample_passages):,} sampled passages for k-means...")
        sample_embs, _ = encode(encoder, sample_passages)
        log(f"  embs: {sample_embs.shape} dtype={sample_embs.dtype}  ||emb||² mean={((sample_embs.float()**2).sum(1)).mean():.4f}")

        # ── Step 2: k-means ───────────────────────────────────
        num_partitions = int(2 ** np.floor(np.log2(16 * math.sqrt(sample_embs.size(0)))))
        log(f"Step 2: Training k-means ({num_partitions:,} centroids)...")
        centroids = train_kmeans(sample_embs, num_partitions, niters=KMEANS_NITERS)

        # ── Step 3: residual → bucket_weights ────────────────
        log("Step 3: Computing residuals → training bucket_weights...")
        codes_s = find_codes(sample_embs, centroids)
        res_s = sample_embs - centroids[codes_s.long()]
        log(f"  residuals: ||r||² mean={((res_s**2).sum(1)).mean():.4f}")

        bucket_cutoffs, bucket_weights = train_buckets(res_s)
        log(f"  bucket_cutoffs : {bucket_cutoffs.tolist()}")
        log(f"  bucket_weights : {bucket_weights.tolist()}")
        del sample_embs, res_s

        # ── Step 4: 첫 N_CSV개 패시지 인코딩 ─────────────────
        log(f"Step 4: Encoding first {N_CSV} passages (CSV target)...")
        csv_passages = all_passages[:N_CSV]
        csv_embs, csv_doclens = encode(encoder, csv_passages)
        total_tokens = sum(csv_doclens)
        log(f"  tokens: {total_tokens:,} dtype={csv_embs.dtype}  ||emb||² mean={((csv_embs.float()**2).sum(1)).mean():.4f}")

        # ── Step 5: 동일 embedding → float16 + 2-bit ─────────
        log("Step 5: Computing float16 and 2-bit residuals from SAME embeddings...")
        codes_csv    = find_codes(csv_embs, centroids)
        centroid_csv = centroids[codes_csv.long()]          # (total_tokens, 128) float16
        residuals_f16 = csv_embs - centroid_csv             # (total_tokens, 128) float16

        log(f"  float16 residuals: dtype={residuals_f16.dtype}  ||r||² mean={((residuals_f16.float()**2).sum(1)).mean():.4f}")

        packed_2bit   = binarize(residuals_f16, bucket_cutoffs)
        residuals_2bit = decode_2bit(packed_2bit, bucket_weights)
        log(f"  2-bit unique values: {residuals_2bit.unique().tolist()}")

        # ── passage_id / token_idx 매핑 ──────────────────────
        passage_ids, token_idxs = [], []
        for pid, doclen in enumerate(csv_doclens):
            for t in range(doclen):
                passage_ids.append(pid)
                token_idxs.append(t)

        header = build_header()

        # ── Step 6: CSV 저장 ──────────────────────────────────
        log("Step 6: Saving CSVs...")
        save_csv(
            os.path.join(OUT_DIR, "colbert_float16_correct.csv"),
            header,
            build_rows(passage_ids, token_idxs, centroid_csv, residuals_f16),
        )
        save_csv(
            os.path.join(OUT_DIR, "colbert_2bit_correct.csv"),
            header,
            build_rows(passage_ids, token_idxs, centroid_csv, residuals_2bit),
        )

    log("\nAll done.")


if __name__ == "__main__":
    main()
