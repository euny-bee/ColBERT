#!/usr/bin/env python3
"""
step1_lotte_writing.py
------------------------
LoTTE Lifestyle 컬렉션 다운로드 + float16 + 2-bit 인덱스 구축.

[A] HuggingFace 다운로드 → collection.tsv, queries.search.tsv,
    queries.forum.tsv, qrels.search.tsv, qrels.forum.tsv
[B] BERT 인코딩 → raw_embs 저장
[C] float16 index 빌드 (raw_embs → residuals as float16)
[D] 2-bit index 빌드 (float16 residuals → 2-bit 재양자화)

실행:
  python step1_lotte_writing.py
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
TOPIC        = "writing"
TOPIC_HF     = "lifestyle"          # HuggingFace config 이름
DATA_DIR     = f"D:/beir/lotte/{TOPIC}"
COLLECTION   = f"{DATA_DIR}/collection.tsv"
QUERIES_SEARCH = f"{DATA_DIR}/queries.search.tsv"
QUERIES_FORUM  = f"{DATA_DIR}/queries.forum.tsv"
QRELS_SEARCH   = f"{DATA_DIR}/qrels.search.tsv"
QRELS_FORUM    = f"{DATA_DIR}/qrels.forum.tsv"

INDEX_ROOT   = os.path.join(COLBERT_DIR, f"experiments/lotte_writing/indexes")
RAW_EMBS_DIR = f"{DATA_DIR}/raw_embs"

ANALOG_INDEX = f"{TOPIC}.analog"
BIT2_INDEX   = f"{TOPIC}.2bit"
CHECKPOINT   = "colbert-ir/colbertv2.0"

NBITS        = 2
DIM          = 128
INDEX_BSIZE  = 64

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# [A] 다운로드
# ============================================================
def _download():
    import urllib.request
    import tarfile

    os.makedirs(DATA_DIR, exist_ok=True)

    qas_search = f"{DATA_DIR}/qas.search.jsonl"

    # 이미 다운로드된 경우 스킵
    if os.path.exists(COLLECTION) and os.path.exists(QUERIES_SEARCH)             and os.path.exists(qas_search):
        log(f"  Data already exists at {DATA_DIR}, skipping download.")
        return

    LOTTE_URL = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz"
    log(f"Downloading LoTTE from {LOTTE_URL} ...")
    log(f"  (Extracting {TOPIC}/test/ only)")

    target_prefix = f"lotte/{TOPIC}/test/"

    with urllib.request.urlopen(LOTTE_URL) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.name.startswith(target_prefix):
                    continue
                fname = os.path.basename(member.name)
                if not fname:
                    continue
                out_path = os.path.join(DATA_DIR, fname)
                log(f"  Extracting {member.name} -> {out_path}")
                f = tar.extractfile(member)
                if f is not None:
                    with open(out_path, "wb") as fout:
                        fout.write(f.read())

    # questions.*.tsv -> queries.*.tsv 로 이름 통일
    for qt in ["search", "forum"]:
        src = os.path.join(DATA_DIR, f"questions.{qt}.tsv")
        dst = os.path.join(DATA_DIR, f"queries.{qt}.tsv")
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)

    log(f"  Download done -> {DATA_DIR}")


# ============================================================
# [B] BERT 인코딩 + raw_embs 저장
# ============================================================
def _do_encode():
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.indexing.index_saver import IndexSaver

    os.makedirs(RAW_EMBS_DIR, exist_ok=True)
    os.makedirs(INDEX_ROOT, exist_ok=True)

    analog_path = os.path.join(INDEX_ROOT, ANALOG_INDEX)
    if os.path.exists(analog_path):
        shutil.rmtree(analog_path)
        log(f"Removed existing index: {analog_path}")

    _orig_save_chunk = IndexSaver.save_chunk

    def _patched_save_chunk(self, chunk_idx, offset, embs, doclens):
        raw_path = os.path.join(RAW_EMBS_DIR, f"{chunk_idx}.pt")
        torch.save(embs.half().cpu(), raw_path)
        log(f"  raw embs chunk {chunk_idx}: {embs.shape}")
        return _orig_save_chunk(self, chunk_idx, offset, embs, doclens)

    IndexSaver.save_chunk = _patched_save_chunk

    try:
        with Run().context(RunConfig(nranks=1, experiment=f"lotte_{TOPIC}",
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
# [C] float16 index 빌드
# ============================================================
def _build_analog():
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
        embs = torch.load(chunk_file, map_location="cpu", weights_only=True)

        all_residuals = []
        for sub in embs.split(1 << 16):
            if codec.use_gpu:
                sub_dev = sub.cuda().half()
            else:
                sub_dev = sub.float()
            codes     = codec.compress_into_codes(sub_dev, out_device="cpu")
            centroids = codec.lookup_centroids(codes, out_device="cpu")
            residuals = sub.half() - centroids.half()
            all_residuals.append(residuals.cpu())

        chunk_residuals = torch.cat(all_residuals)
        out_path = os.path.join(index_path, f"{chunk_idx}.residuals.pt")
        torch.save(chunk_residuals, out_path)
        log(f"  chunk {chunk_idx}: {chunk_residuals.shape} float16 saved")

        if codec.use_gpu:
            torch.cuda.empty_cache()

    log(f"float16 index done -> {index_path}")


# ============================================================
# [D] 2-bit index 빌드 (float16 residuals 재양자화)
# ============================================================
def _build_2bit():
    analog_path = os.path.join(INDEX_ROOT, ANALOG_INDEX)
    bit2_path   = os.path.join(INDEX_ROOT, BIT2_INDEX)

    if os.path.exists(bit2_path):
        shutil.rmtree(bit2_path)
    shutil.copytree(analog_path, bit2_path)
    log(f"Copied analog -> {BIT2_INDEX}")

    residual_files = sorted(
        glob.glob(os.path.join(analog_path, "*.residuals.pt")),
        key=lambda p: int(os.path.basename(p).split(".")[0])
    )

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
    cutoff_qs   = torch.linspace(0, 1, num_buckets + 1)[1:-1]
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

    arange_bits = torch.arange(0, NBITS, dtype=torch.uint8)

    for rf in residual_files:
        chunk_idx = int(os.path.basename(rf).split(".")[0])
        res_f16   = torch.load(rf, map_location="cpu", weights_only=True)

        res = torch.bucketize(res_f16.float(), bucket_cutoffs).to(torch.uint8)
        res = res.unsqueeze(-1).expand(*res.size(), NBITS)
        res = res >> arange_bits
        res = res & 1
        packed_np = np.packbits(np.asarray(res.contiguous().flatten()))
        packed    = torch.as_tensor(packed_np, dtype=torch.uint8)
        packed    = packed.reshape(res_f16.size(0), DIM // 8 * NBITS)

        out_path = os.path.join(bit2_path, f"{chunk_idx}.residuals.pt")
        torch.save(packed, out_path)
        log(f"  chunk {chunk_idx}: {packed.shape} uint8 (2-bit) saved")

    log(f"2-bit index done -> {bit2_path}")


# ============================================================
# main
# ============================================================
def main():
    log("=" * 60)
    log(f"step1_lotte_{TOPIC}.py - LoTTE {TOPIC.capitalize()} index build")
    log("=" * 60)

    t_total = time.time()

    log("\n[A] Downloading LoTTE data...")
    t0 = time.time()
    _download()
    log(f"[A] Done in {(time.time()-t0)/60:.1f}min")

    log("\n[B] Encoding passages + saving raw embeddings...")
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "encode"],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        log("ERROR: encoding failed")
        sys.exit(1)
    log(f"[B] Done in {(time.time()-t0)/60:.1f}min")

    log("\n[C] Building float16 (analog) index...")
    t0 = time.time()
    _build_analog()
    log(f"[C] Done in {(time.time()-t0)/60:.1f}min")

    log("\n[D] Building 2-bit index...")
    t0 = time.time()
    _build_2bit()
    log(f"[D] Done in {(time.time()-t0)/60:.1f}min")

    shutil.rmtree(RAW_EMBS_DIR)
    log("\nRaw embeddings cleaned up.")
    log(f"\nAll done! 총 소요: {(time.time()-t_total)/60:.1f}min")


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
