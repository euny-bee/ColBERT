#!/usr/bin/env python3
"""
step1_1m.py
-----------
MS MARCO 110만 서브셋에 대해 float16 + 2-bit 인덱스 구축.

순서:
  A. ColBERT Indexer 실행 → raw float16 embeddings 청크별 저장
  B. float16 index: (emb - centroid).half() → 1m.analog/
  C. 2-bit index: binarize(residual_f16) → 1m.2bit/

실행:
  python step1_1m.py
"""

import os
import sys
import glob
import shutil
import subprocess
import time
import torch
import numpy as np

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
COLLECTION   = "D:/msmarco/collection_1m_fair.tsv"
INDEX_ROOT   = os.path.join(COLBERT_DIR, "experiments/msmarco_1m/indexes")
RAW_EMBS_DIR = "D:/msmarco/raw_embs"

ANALOG_INDEX = "1m.analog"
BIT2_INDEX   = "1m.2bit"
CHECKPOINT   = "colbert-ir/colbertv2.0"

NBITS        = 2
DIM          = 128
INDEX_BSIZE  = 64   # RTX 3080 Ti 12GB

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# Step A: Indexer 실행 + raw embeddings 저장
# ============================================================
def _do_encode():
    """ColBERT Indexer로 인코딩. IndexSaver.save_chunk 패치로 raw embs 저장."""
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.indexing.index_saver import IndexSaver

    os.makedirs(RAW_EMBS_DIR, exist_ok=True)
    os.makedirs(INDEX_ROOT, exist_ok=True)

    _orig_save_chunk = IndexSaver.save_chunk

    def _patched_save_chunk(self, chunk_idx, offset, embs, doclens):
        raw_path = os.path.join(RAW_EMBS_DIR, f"{chunk_idx}.pt")
        torch.save(embs.half().cpu(), raw_path)
        log(f"  raw embs chunk {chunk_idx}: {embs.shape}")
        return _orig_save_chunk(self, chunk_idx, offset, embs, doclens)

    IndexSaver.save_chunk = _patched_save_chunk

    try:
        with Run().context(RunConfig(nranks=1, experiment="msmarco_1m",
                                     avoid_fork_if_possible=True)):
            config = ColBERTConfig(
                nbits=NBITS, doc_maxlen=220, query_maxlen=32,
                index_bsize=INDEX_BSIZE, avoid_fork_if_possible=True,
            )
            indexer = Indexer(checkpoint=CHECKPOINT, config=config)
            indexer.index(name=ANALOG_INDEX, collection=COLLECTION, overwrite=True)
    finally:
        IndexSaver.save_chunk = _orig_save_chunk


# ============================================================
# Step B: float16 (analog) index 구축
# ============================================================
def _build_analog():
    """raw embs → (emb - centroid).half() → 1m.analog 잔차 교체."""
    from colbert.indexing.codecs.residual import ResidualCodec

    index_path = os.path.join(INDEX_ROOT, ANALOG_INDEX)
    codec = ResidualCodec.load(index_path=index_path)

    chunk_files = sorted(
        glob.glob(os.path.join(RAW_EMBS_DIR, "*.pt")),
        key=lambda p: int(os.path.basename(p).replace(".pt", ""))
    )
    log(f"Building float16 index from {len(chunk_files)} chunks...")

    for chunk_file in chunk_files:
        chunk_idx = int(os.path.basename(chunk_file).replace(".pt", ""))
        embs = torch.load(chunk_file, map_location="cpu", weights_only=True)  # float16

        all_residuals = []
        for sub in embs.split(1 << 16):  # 65K 단위 처리
            if codec.use_gpu:
                sub_dev = sub.cuda().half()
            else:
                sub_dev = sub.float()
            codes     = codec.compress_into_codes(sub_dev, out_device="cpu")
            centroids = codec.lookup_centroids(codes, out_device="cpu")
            residuals = sub.half() - centroids.half()
            all_residuals.append(residuals.cpu())

        chunk_residuals = torch.cat(all_residuals)  # (N_tokens, 128) float16
        out_path = os.path.join(index_path, f"{chunk_idx}.residuals.pt")
        torch.save(chunk_residuals, out_path)
        log(f"  chunk {chunk_idx}: {chunk_residuals.shape} float16 saved")

        if codec.use_gpu:
            torch.cuda.empty_cache()

    log(f"float16 index done → {index_path}")


# ============================================================
# Step C: 2-bit-from-float16 index 구축
# ============================================================
def _build_2bit():
    """float16 residuals → binarize → 1m.2bit."""
    analog_path = os.path.join(INDEX_ROOT, ANALOG_INDEX)
    bit2_path   = os.path.join(INDEX_ROOT, BIT2_INDEX)

    if os.path.exists(bit2_path):
        shutil.rmtree(bit2_path)
    shutil.copytree(analog_path, bit2_path)
    log(f"Copied analog → {BIT2_INDEX}")

    residual_files = sorted(
        glob.glob(os.path.join(analog_path, "*.residuals.pt")),
        key=lambda p: int(os.path.basename(p).split(".")[0])
    )

    # 청크별 샘플링 → OOM 방지
    log("Computing bucket_cutoffs (sampling per chunk)...")
    SAMPLE_PER_CHUNK = 300_000
    all_samples = []
    for rf in residual_files:
        r = torch.load(rf, map_location="cpu", weights_only=True)
        flat = r.flatten()
        if flat.numel() > SAMPLE_PER_CHUNK:
            idx = torch.randperm(flat.numel())[:SAMPLE_PER_CHUNK]
            flat = flat[idx]
        all_samples.append(flat)
        del r

    all_flat = torch.cat(all_samples)
    max_samples = 5_000_000
    if all_flat.numel() > max_samples:
        idx = torch.randperm(all_flat.numel())[:max_samples]
        all_flat = all_flat[idx]

    num_buckets = 2 ** NBITS  # 4
    cutoff_qs = torch.linspace(0, 1, num_buckets + 1)[1:-1]
    bucket_cutoffs = torch.quantile(all_flat.float(), cutoff_qs)

    bounds = torch.cat([torch.tensor([-1e4]), bucket_cutoffs, torch.tensor([1e4])])
    flat_f32 = all_flat.float()
    bucket_weights = torch.stack([
        flat_f32[(flat_f32 >= bounds[i]) & (flat_f32 < bounds[i + 1])].median()
        for i in range(num_buckets)
    ])

    log(f"  bucket_cutoffs : {bucket_cutoffs.tolist()}")
    log(f"  bucket_weights : {bucket_weights.tolist()}")

    torch.save(
        (bucket_cutoffs, bucket_weights),
        os.path.join(bit2_path, "buckets.pt")
    )

    # 청크별 binarize
    arange_bits = torch.arange(0, NBITS, dtype=torch.uint8)

    for rf in residual_files:
        chunk_idx = int(os.path.basename(rf).split(".")[0])
        res_f16 = torch.load(rf, map_location="cpu", weights_only=True)

        res = torch.bucketize(res_f16.float(), bucket_cutoffs).to(torch.uint8)
        res = res.unsqueeze(-1).expand(*res.size(), NBITS)
        res = res >> arange_bits
        res = res & 1
        packed_np = np.packbits(np.asarray(res.contiguous().flatten()))
        packed    = torch.as_tensor(packed_np, dtype=torch.uint8)
        packed    = packed.reshape(res_f16.size(0), DIM // 8 * NBITS)  # (N, 32)

        out_path = os.path.join(bit2_path, f"{chunk_idx}.residuals.pt")
        torch.save(packed, out_path)
        log(f"  chunk {chunk_idx}: {packed.shape} uint8 (2-bit) saved")

    log(f"2-bit index done → {bit2_path}")


# ============================================================
# main
# ============================================================
def main():
    log("=" * 55)
    log("step1_1m.py - 110man MS MARCO index build")
    log("=" * 55)

    t_total = time.time()

    # A: 인코딩 (subprocess → multiprocessing 격리)
    log("\n[A] Encoding 1M passages + saving raw embeddings...")
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "encode"],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        log("ERROR: encoding failed")
        sys.exit(1)
    log(f"[A] Done in {(time.time()-t0)/3600:.1f}h")

    # B: float16 index
    log("\n[B] Building float16 (analog) index...")
    t0 = time.time()
    _build_analog()
    log(f"[B] Done in {(time.time()-t0)/60:.1f}min")

    # C: 2-bit index
    log("\n[C] Building 2-bit index...")
    t0 = time.time()
    _build_2bit()
    log(f"[C] Done in {(time.time()-t0)/60:.1f}min")

    # raw embeddings 정리
    shutil.rmtree(RAW_EMBS_DIR)
    log("\nRaw embeddings cleaned up.")
    log(f"\nAll done! 총 소요: {(time.time()-t_total)/3600:.1f}h")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "encode":
            _do_encode()
        elif cmd == "build_analog":
            _build_analog()
        elif cmd == "build_2bit":
            _build_2bit()
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        main()
