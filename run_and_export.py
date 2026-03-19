#!/usr/bin/env python3
"""
run_and_export.py
-----------------
1. float16 (subset.analog) 와 2-bit (subset.quantized) 인덱스 각각:
   - 200K passages 전체 검색 → MRR@10, R@50, R@1K 계산
   - 첫 500 passages raw data (centroid 벡터 + residual 벡터) 추출
2. 결과를 CSV 4개로 저장:
   - colbert_metrics.csv
   - colbert_passages.csv
   - colbert_float16_raw.csv
   - colbert_2bit_raw.csv
"""

import os
import sys
import gc
import json
import time
import subprocess
import csv

import torch

COLBERT_DIR = os.path.expanduser("~/ColBERT")
DATA_DIR    = os.path.join(COLBERT_DIR, "data/msmarco")
RESULTS_DIR = os.path.join(COLBERT_DIR, "experiments/msmarco")

SUBSET_COLLECTION = os.path.join(DATA_DIR, "subset/collection.tsv")
SUBSET_QRELS      = os.path.join(DATA_DIR, "subset/qrels.tsv")
QUERIES_PATH      = os.path.join(DATA_DIR, "queries.dev.small.tsv")

OUTPUT_DIR = os.path.join(COLBERT_DIR, "csv_output")

NUM_PASSAGES = 500  # raw data 추출 대상 passage 수

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 공통 유틸
# ============================================================

def load_passages(n):
    """collection.tsv에서 첫 n개 passage 읽기. passage_id는 0부터 새로 매김."""
    passages = []
    with open(SUBSET_COLLECTION, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            parts = line.rstrip("\n").split("\t", 1)
            text = parts[1] if len(parts) > 1 else ""
            passages.append((i, text))  # (새 passage_id, text)
    return passages


def load_doclens(index_path, n_passages):
    """doclens.0.json에서 첫 n_passages개 토큰 수 로드."""
    with open(os.path.join(index_path, "doclens.0.json")) as f:
        doclens = json.load(f)
    doclens = doclens[:n_passages]
    total_tokens = sum(doclens)
    return doclens, total_tokens


def run_eval(ranking_path, qrels_path):
    """MRR@10, R@50, R@1K 계산."""
    log(f"Evaluating: {ranking_path}")
    result = subprocess.run(
        [sys.executable, "-m", "utility.evaluate.msmarco_passages",
         "--qrels", qrels_path, "--ranking", ranking_path],
        capture_output=True, text=True, cwd=COLBERT_DIR
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(result.stderr, flush=True)

    metrics = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        if "only for ranked" in line:
            continue
        if "MRR@10 =" in line:
            metrics["MRR@10"] = float(line.split("=")[-1].strip())
        elif "Recall@50 =" in line:
            metrics["R@50"] = float(line.split("=")[-1].strip())
        elif "Recall@1000 =" in line:
            metrics["R@1k"] = float(line.split("=")[-1].strip())
    return metrics


# ============================================================
# analog (float16) 인덱스 로드를 위한 패치/복원
# ============================================================

def _apply_analog_patches():
    """
    subset.analog 인덱스(float16 residuals)를 로드하기 위해
    ColBERT 내부 코드 3곳을 런타임에 패치.
    원본 함수는 반환값으로 보존해 나중에 복원 가능.
    """
    import colbert.indexing.codecs.residual_embeddings as re_mod
    import colbert.indexing.codecs.residual as r_mod

    # --- 1) ResidualEmbeddings.__init__ : dtype assert 우회 ---
    orig_init = re_mod.ResidualEmbeddings.__init__

    def patched_init(self, codes, residuals):
        assert codes.size(0) == residuals.size(0)
        assert codes.dim() == 1 and residuals.dim() == 2
        # dtype assert 제거 (float16 허용)
        self.codes = codes.to(torch.int32)
        self.residuals = residuals

    re_mod.ResidualEmbeddings.__init__ = patched_init

    # --- 2) ResidualEmbeddings.load_chunks : float16 버퍼 할당 ---
    orig_load_chunks = re_mod.ResidualEmbeddings.load_chunks

    @classmethod
    def patched_load_chunks(cls, index_path, chunk_idxs, num_embeddings, load_index_with_mmap=False):
        import tqdm, ujson
        num_embeddings += 512
        with open(os.path.join(index_path, 'metadata.json')) as f:
            meta = ujson.load(f)['config']
        dim = meta['dim']  # 128

        codes     = torch.empty(num_embeddings, dtype=torch.int32)
        residuals = torch.empty(num_embeddings, dim, dtype=torch.float16)  # float16, 128dim

        codes_offset = 0
        for chunk_idx in tqdm.tqdm(chunk_idxs):
            chunk = cls.load(index_path, chunk_idx)
            codes_endpos = codes_offset + chunk.codes.size(0)
            codes[codes_offset:codes_endpos]     = chunk.codes
            residuals[codes_offset:codes_endpos] = chunk.residuals
            codes_offset = codes_endpos

        return cls(codes, residuals)

    re_mod.ResidualEmbeddings.load_chunks = patched_load_chunks

    # --- 3) ResidualCodec.decompress : float16 residuals 직접 덧셈 ---
    orig_decompress = r_mod.ResidualCodec.decompress

    def patched_decompress(self, compressed_embs):
        codes, residuals = compressed_embs.codes, compressed_embs.residuals
        D = []
        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            if self.use_gpu:
                codes_     = codes_.cuda()
                residuals_ = residuals_.cuda().half()
            else:
                residuals_ = residuals_.float()
            centroids_ = self.lookup_centroids(codes_, out_device=codes_.device)
            centroids_ = centroids_ + residuals_
            if self.use_gpu:
                D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()
            else:
                D_ = torch.nn.functional.normalize(centroids_.float(), p=2, dim=-1)
            D.append(D_)
        return torch.cat(D)

    r_mod.ResidualCodec.decompress = patched_decompress

    return orig_init, orig_load_chunks, orig_decompress


def _restore_analog_patches(orig_init, orig_load_chunks, orig_decompress):
    import colbert.indexing.codecs.residual_embeddings as re_mod
    import colbert.indexing.codecs.residual as r_mod
    re_mod.ResidualEmbeddings.__init__       = orig_init
    re_mod.ResidualEmbeddings.load_chunks    = orig_load_chunks
    r_mod.ResidualCodec.decompress           = orig_decompress


# ============================================================
# float16 인덱스 처리
# ============================================================

def process_float16(doclens, total_tokens):
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    index_path  = os.path.join(RESULTS_DIR, "indexes/subset.analog")
    ranking_dir = os.path.join(RESULTS_DIR, "rankings")
    os.makedirs(ranking_dir, exist_ok=True)

    ranking_path = os.path.join(ranking_dir, "float16.ranking.tsv")
    for ext in ["", ".meta"]:
        if os.path.exists(ranking_path + ext):
            os.remove(ranking_path + ext)

    # analog 패치 적용
    orig_init, orig_load_chunks, orig_decompress = _apply_analog_patches()

    try:
        # --- 검색 + 평가 ---
        log("Loading float16 index (subset.analog)...")
        with Run().context(RunConfig(nranks=1, experiment="msmarco")):
            config = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32)
            searcher = Searcher(index="subset.analog", config=config)
            queries  = Queries(QUERIES_PATH)

            log(f"Searching {len(queries)} queries (200K passages)...")
            start = time.time()
            ranking = searcher.search_all(queries, k=1000)
            log(f"Search done in {time.time()-start:.0f}s")

            ranking.save(ranking_path)

    finally:
        # 패치 복원 (오류가 나도 반드시 복원)
        _restore_analog_patches(orig_init, orig_load_chunks, orig_decompress)

    metrics = run_eval(ranking_path, SUBSET_QRELS)
    log(f"float16 metrics: {metrics}")

    # --- raw data 추출 ---
    log(f"Extracting raw data for first {NUM_PASSAGES} passages (float16)...")

    codes     = torch.load(os.path.join(index_path, "0.codes.pt"),     map_location="cpu")
    residuals = torch.load(os.path.join(index_path, "0.residuals.pt"), map_location="cpu")
    centroids = torch.load(os.path.join(index_path, "centroids.pt"),   map_location="cpu")
    if isinstance(centroids, tuple):
        centroids = centroids[0]

    codes     = codes[:total_tokens]      # (total_tokens,)
    residuals = residuals[:total_tokens]  # (total_tokens, 128) float16
    centroid_vecs = centroids[codes.long()].float()  # (total_tokens, 128)
    residuals_f   = residuals.float()                # (total_tokens, 128)

    # passage_id, token_idx 컬럼 생성
    passage_ids = []
    token_idxs  = []
    for pid, doclen in enumerate(doclens):
        for t in range(doclen):
            passage_ids.append(pid)
            token_idxs.append(t)

    rows = []
    for i in range(total_tokens):
        row = [passage_ids[i], token_idxs[i]]
        row += centroid_vecs[i].tolist()
        row += residuals_f[i].tolist()
        rows.append(row)

    del codes, residuals, centroids, centroid_vecs, residuals_f
    gc.collect()

    return metrics, rows


# ============================================================
# 2-bit 인덱스 처리
# ============================================================

def decode_2bit_residuals(residuals_uint8, bucket_weights):
    """
    residuals_uint8: (N, 32) uint8  →  (N, 128) float
    ColBERT binarize() 인코딩 방식:
      - arange_bits=[0,1] → lsb 먼저 저장
      - np.packbits (MSB first) → byte의 bit7=lsb, bit6=msb
    따라서 올바른 값 복원: value = lsb + 2*msb = ((byte>>7)&1) | (((byte>>6)&1)<<1)
    """
    N = residuals_uint8.shape[0]
    r = residuals_uint8.int()  # (N, 32)

    # 각 byte에서 2bit씩 4개 추출 (lsb=높은 비트, msb=낮은 비트)
    v0 = ((r >> 7) & 1) | (((r >> 6) & 1) << 1)  # (N, 32)
    v1 = ((r >> 5) & 1) | (((r >> 4) & 1) << 1)  # (N, 32)
    v2 = ((r >> 3) & 1) | (((r >> 2) & 1) << 1)  # (N, 32)
    v3 = ((r >> 1) & 1) | (((r >> 0) & 1) << 1)  # (N, 32)

    # (N, 32, 4) → (N, 128) 로 재구성
    indices = torch.stack([v0, v1, v2, v3], dim=-1).reshape(N, 128)  # values 0~3

    bw = bucket_weights.float()
    decoded = bw[indices.long()]  # (N, 128) float
    return decoded


def process_2bit(doclens, total_tokens):
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    index_path  = os.path.join(RESULTS_DIR, "indexes/subset.quantized")
    ranking_dir = os.path.join(RESULTS_DIR, "rankings")
    os.makedirs(ranking_dir, exist_ok=True)

    ranking_path = os.path.join(ranking_dir, "2bit.ranking.tsv")
    for ext in ["", ".meta"]:
        if os.path.exists(ranking_path + ext):
            os.remove(ranking_path + ext)

    # --- 검색 + 평가 ---
    log("Loading 2-bit index (subset.quantized)...")
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32)
        searcher = Searcher(index="subset.quantized", config=config)
        queries  = Queries(QUERIES_PATH)

        log(f"Searching {len(queries)} queries (200K passages)...")
        start = time.time()
        ranking = searcher.search_all(queries, k=1000)
        log(f"Search done in {time.time()-start:.0f}s")

        ranking.save(ranking_path)

    metrics = run_eval(ranking_path, SUBSET_QRELS)
    log(f"2-bit metrics: {metrics}")

    # --- raw data 추출 ---
    log(f"Extracting raw data for first {NUM_PASSAGES} passages (2-bit)...")

    codes          = torch.load(os.path.join(index_path, "0.codes.pt"),     map_location="cpu")
    residuals_u8   = torch.load(os.path.join(index_path, "0.residuals.pt"), map_location="cpu")
    centroids      = torch.load(os.path.join(index_path, "centroids.pt"),   map_location="cpu")
    buckets        = torch.load(os.path.join(index_path, "buckets.pt"),     map_location="cpu")
    if isinstance(centroids, tuple):
        centroids = centroids[0]
    bucket_cutoffs, bucket_weights = buckets  # bucket_weights: (4,) float16

    codes        = codes[:total_tokens]        # (total_tokens,)
    residuals_u8 = residuals_u8[:total_tokens] # (total_tokens, 32) uint8
    centroid_vecs = centroids[codes.long()].float()          # (total_tokens, 128)
    residuals_f   = decode_2bit_residuals(residuals_u8, bucket_weights)  # (total_tokens, 128)

    # passage_id, token_idx 컬럼 생성
    passage_ids = []
    token_idxs  = []
    for pid, doclen in enumerate(doclens):
        for t in range(doclen):
            passage_ids.append(pid)
            token_idxs.append(t)

    rows = []
    for i in range(total_tokens):
        row = [passage_ids[i], token_idxs[i]]
        row += centroid_vecs[i].tolist()
        row += residuals_f[i].tolist()
        rows.append(row)

    del codes, residuals_u8, centroids, centroid_vecs, residuals_f
    gc.collect()

    return metrics, rows


# ============================================================
# CSV 저장
# ============================================================

def save_csv(filepath, header, rows):
    log(f"Saving {filepath} ({len(rows)} rows)...")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    log(f"Saved: {filepath}")


def build_raw_header():
    header = ["passage_id", "token_idx"]
    header += [f"centroid_dim_{i}" for i in range(128)]
    header += [f"residual_dim_{i}" for i in range(128)]
    return header


# ============================================================
# main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 공통 준비 ---
    log(f"Loading first {NUM_PASSAGES} passages from collection...")
    passages = load_passages(NUM_PASSAGES)

    float16_index_path = os.path.join(RESULTS_DIR, "indexes/subset.analog")
    doclens, total_tokens = load_doclens(float16_index_path, NUM_PASSAGES)
    log(f"Total tokens for {NUM_PASSAGES} passages: {total_tokens}")

    # passages (공통)
    save_csv(
        os.path.join(OUTPUT_DIR, "colbert_passages.csv"),
        ["passage_id", "passage_text"],
        passages
    )

    # --- float16 처리 ---
    log("=" * 50)
    log("Processing float16 index...")
    float16_metrics, float16_rows = process_float16(doclens, total_tokens)
    save_csv(
        os.path.join(OUTPUT_DIR, "colbert_float16_raw.csv"),
        build_raw_header(),
        float16_rows
    )
    del float16_rows
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2-bit 처리 ---
    log("=" * 50)
    log("Processing 2-bit index...")
    twobit_metrics, twobit_rows = process_2bit(doclens, total_tokens)
    save_csv(
        os.path.join(OUTPUT_DIR, "colbert_2bit_raw.csv"),
        build_raw_header(),
        twobit_rows
    )
    del twobit_rows
    gc.collect()
    torch.cuda.empty_cache()

    # --- metrics 저장 ---
    log("=" * 50)
    save_csv(
        os.path.join(OUTPUT_DIR, "colbert_metrics.csv"),
        ["index", "MRR@10", "R@50", "R@1k"],
        [
            ["float16", float16_metrics.get("MRR@10"), float16_metrics.get("R@50"), float16_metrics.get("R@1k")],
            ["2bit",    twobit_metrics.get("MRR@10"),  twobit_metrics.get("R@50"),  twobit_metrics.get("R@1k")],
        ]
    )

    log("=" * 50)
    log("All done. Output files:")
    for fname in ["colbert_metrics.csv", "colbert_passages.csv", "colbert_float16_raw.csv", "colbert_2bit_raw.csv"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        log(f"  {fname}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
