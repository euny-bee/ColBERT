#!/usr/bin/env python3
"""
step3_wiki_trivia.py
--------------------
Wikipedia TriviaQA: float16 vs 2-bit index 비교
- Corpus : D:/beir/wiki_trivia_500k/collection.tsv  (~946K passages)
- Queries: D:/beir/wiki_trivia_500k/queries.tsv     (8,837 queries)
- Eval   : Success@5, nDCG@10 (answer string matching)

사용:
  python step3_wiki_trivia.py              # 전체 파이프라인
  python step3_wiki_trivia.py encode       # 인코딩만 (subprocess 진입점)
  python step3_wiki_trivia.py search_f16   # float16 검색 (subprocess 진입점)
  python step3_wiki_trivia.py search_2bit  # 2-bit 검색 (subprocess 진입점)
"""

import os
import sys
import json
import shutil
import subprocess
import time
import glob

import torch

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
DATA_DIR     = "D:/beir/wiki_trivia_500k"
INDEX_BASE   = "D:/beir_indexes"          # same root as run_beir.py
EXPERIMENT   = "wiki_trivia"              # RunConfig experiment name
CHECKPOINT   = "colbert-ir/colbertv2.0"
NBITS        = 2
INDEX_BSIZE  = 64
SEARCH_K     = 1000
NDOCS        = 4096

COLLECTION   = os.path.join(DATA_DIR, "collection.tsv")
QUERIES_PATH = os.path.join(DATA_DIR, "queries.tsv")
ANSWERS_PATH = os.path.join(DATA_DIR, "answers.json")
RANKING_DIR  = os.path.join(COLBERT_DIR, f"experiments/{EXPERIMENT}/rankings")

ANALOG_NAME  = "wiki_trivia.analog"
BIT2_NAME    = "wiki_trivia.2bit"
RAW_DIR      = os.path.join(DATA_DIR, "raw_embs")
# Index files: D:/beir_indexes/wiki_trivia/indexes/wiki_trivia.{analog,2bit}

RESULTS_FILE = os.path.join(COLBERT_DIR, "experiments/beir_all_results.json")

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# A. 인코딩 (subprocess 진입점)
# ============================================================
def _do_encode():
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.indexing.index_saver import IndexSaver

    index_dir   = os.path.join(INDEX_BASE, EXPERIMENT, "indexes")
    analog_path = os.path.join(index_dir, ANALOG_NAME)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    if os.path.exists(analog_path):
        shutil.rmtree(analog_path)

    _orig_save_chunk = IndexSaver.save_chunk

    def _patched_save_chunk(self, chunk_idx, offset, embs, doclens):
        raw_path = os.path.join(RAW_DIR, f"{chunk_idx}.pt")
        torch.save(embs.half().cpu(), raw_path)
        return _orig_save_chunk(self, chunk_idx, offset, embs, doclens)

    IndexSaver.save_chunk = _patched_save_chunk

    try:
        with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT,
                                     root=INDEX_BASE,
                                     avoid_fork_if_possible=True)):
            config = ColBERTConfig(
                nbits=NBITS, doc_maxlen=220, query_maxlen=32,
                index_bsize=INDEX_BSIZE, avoid_fork_if_possible=True,
            )
            indexer = Indexer(checkpoint=CHECKPOINT, config=config)
            indexer.index(name=ANALOG_NAME, collection=COLLECTION, overwrite=True)
    finally:
        IndexSaver.save_chunk = _orig_save_chunk


# ============================================================
# B. float16 인덱스 빌드 (analog)
# ============================================================
def _build_analog():
    from colbert.indexing.codecs.residual import ResidualCodec

    index_root  = os.path.join(INDEX_BASE, EXPERIMENT, "indexes")
    analog_path = os.path.join(index_root, ANALOG_NAME)

    codec = ResidualCodec.load(index_path=analog_path)
    chunk_files = sorted(
        glob.glob(os.path.join(RAW_DIR, "*.pt")),
        key=lambda p: int(os.path.basename(p).replace(".pt", ""))
    )

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
            if codec.use_gpu:
                torch.cuda.empty_cache()

        all_residuals = torch.cat(all_residuals, dim=0)

        import numpy as np
        residuals_np = all_residuals.numpy()
        residuals_f16 = residuals_np.astype(np.float16)

        bucket_path = os.path.join(analog_path, f"buckets.{chunk_idx}.pt")
        residuals_path = os.path.join(analog_path, f"residuals.{chunk_idx}.pt")
        if os.path.exists(bucket_path):
            os.remove(bucket_path)

        torch.save(
            torch.from_numpy(residuals_f16),
            residuals_path
        )
        log(f"    chunk {chunk_idx}: float16 residuals saved")


# ============================================================
# C. 2-bit 인덱스 빌드
# ============================================================
def _build_2bit():
    from colbert.indexing.codecs.residual import ResidualCodec

    index_root   = os.path.join(INDEX_BASE, EXPERIMENT, "indexes")
    analog_path  = os.path.join(index_root, ANALOG_NAME)
    bit2_path    = os.path.join(index_root, BIT2_NAME)

    if os.path.exists(bit2_path):
        shutil.rmtree(bit2_path)
    shutil.copytree(analog_path, bit2_path)

    codec = ResidualCodec.load(index_path=bit2_path)
    chunk_files = sorted(
        glob.glob(os.path.join(RAW_DIR, "*.pt")),
        key=lambda p: int(os.path.basename(p).replace(".pt", ""))
    )

    for chunk_file in chunk_files:
        chunk_idx = int(os.path.basename(chunk_file).replace(".pt", ""))
        embs = torch.load(chunk_file, map_location="cpu", weights_only=True)

        all_codes     = []
        all_residuals = []
        for sub in embs.split(1 << 16):
            if codec.use_gpu:
                sub_dev = sub.cuda().half()
            else:
                sub_dev = sub.float()
            codes     = codec.compress_into_codes(sub_dev, out_device="cpu")
            centroids = codec.lookup_centroids(codes, out_device="cpu")
            residuals = sub.half() - centroids.half()
            all_codes.append(codes)
            all_residuals.append(residuals.cpu())
            if codec.use_gpu:
                torch.cuda.empty_cache()

        all_codes     = torch.cat(all_codes,     dim=0)
        all_residuals = torch.cat(all_residuals, dim=0)
        packed = codec.compress(all_residuals)

        residuals_path = os.path.join(bit2_path, f"residuals.{chunk_idx}.pt")
        torch.save(packed, residuals_path)
        log(f"    chunk {chunk_idx}: 2-bit residuals saved")


# ============================================================
# D. 검색 (subprocess 진입점)
# ============================================================
def _do_search(index_name, analog=False):
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    os.makedirs(RANKING_DIR, exist_ok=True)

    ranking_path = os.path.join(RANKING_DIR, f"{index_name}.tsv")
    if os.path.exists(ranking_path):
        os.remove(ranking_path)

    if analog:
        _apply_analog_patches()

    torch.cuda.empty_cache()

    index_path = os.path.join(INDEX_BASE, EXPERIMENT, "indexes", index_name)

    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT,
                                  root=INDEX_BASE)):
        config   = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32,
                                  ndocs=NDOCS)
        searcher = Searcher(index=index_path, config=config)
        queries  = Queries(QUERIES_PATH)
        ranking  = searcher.search_all(queries, k=SEARCH_K)

    with open(ranking_path, "w", encoding="utf-8") as f:
        for items in ranking.flat_ranking:
            f.write('\t'.join(str(x) for x in items) + '\n')
    log(f"  Saved -> {ranking_path}")


# ============================================================
# E. float16 analog patches
# ============================================================
def _apply_analog_patches():
    from colbert.indexing.codecs.residual import ResidualCodec
    import torch.nn.functional as F
    import numpy as np

    _orig_load = ResidualCodec.load

    @classmethod
    def _patched_load(cls, index_path):
        codec = _orig_load.__func__(cls, index_path)
        return codec

    def _decompress(self, codes_, residuals_):
        if residuals_.dtype == torch.float16:
            if self.use_gpu:
                D = []
                for codes_chunk, res_chunk in zip(
                    codes_.split(1 << 16), residuals_.split(1 << 16)
                ):
                    codes_chunk = codes_chunk.cuda()
                    res_chunk   = res_chunk.cuda().half()
                    centroids_  = self.lookup_centroids(codes_chunk, out_device=codes_chunk.device)
                    vec_        = centroids_ + res_chunk
                    D.append(F.normalize(vec_, p=2, dim=-1).half())
                return torch.cat(D)
            else:
                centroids_ = self.lookup_centroids(codes_, out_device="cpu")
                vec_       = centroids_.float() + residuals_.float()
                return F.normalize(vec_, p=2, dim=-1)
        D = []
        for codes_chunk, res_chunk in zip(
            codes_.split(1 << 16), residuals_.split(1 << 16)
        ):
            if self.use_gpu:
                codes_chunk = codes_chunk.cuda()
                res_chunk   = res_chunk.cuda().half()
            else:
                res_chunk = res_chunk.float()
            centroids_  = self.lookup_centroids(codes_chunk, out_device=codes_chunk.device)
            vec_        = centroids_ + res_chunk
            if self.use_gpu:
                D.append(F.normalize(vec_, p=2, dim=-1).half())
            else:
                D.append(F.normalize(vec_.float(), p=2, dim=-1))
        return torch.cat(D)

    ResidualCodec.decompress = _decompress


# ============================================================
# F. Success@5 + nDCG@10 평가
# ============================================================
def _normalize(s):
    """소문자 + 구두점 제거 (DPR 스타일 answer normalization)"""
    import re
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def evaluate(ranking_path):
    """
    ranking_path: qid \t pid \t rank (1-based) TSV
    Returns: {Success@5, Success@20, nDCG@10, num_queries}
    """
    with open(ANSWERS_PATH, encoding="utf-8") as f:
        answers = json.load(f)  # {str(qid): [answer, ...]}

    # Load collection text for evaluation
    log("  Loading collection for evaluation...")
    coll = {}
    with open(COLLECTION, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                coll[int(parts[0])] = _normalize(parts[1])

    # Load rankings per query
    rankings = {}
    with open(ranking_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, pid, rank = int(parts[0]), int(parts[1]), int(parts[2])
            rankings.setdefault(qid, []).append((rank, pid))

    s5 = s20 = ndcg10 = 0.0
    n = 0

    for qid_str, ans_list in answers.items():
        qid = int(qid_str)
        if qid not in rankings:
            continue
        norm_answers = [_normalize(a) for a in ans_list if a.strip()]
        if not norm_answers:
            continue

        ranked_pids = [pid for _, pid in sorted(rankings[qid])]

        def hit(pid):
            text = coll.get(pid, "")
            return any(a in text for a in norm_answers)

        # Success@5
        top5_hit  = any(hit(pid) for pid in ranked_pids[:5])
        top20_hit = any(hit(pid) for pid in ranked_pids[:20])

        # nDCG@10 (binary relevance from top-10)
        rels = [1 if hit(pid) else 0 for pid in ranked_pids[:10]]
        import math
        dcg   = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(sum(rels) + 1, 10)) if i < sum(rels))
        ndcg10_q = dcg / ideal if ideal > 0 else 0.0

        s5    += float(top5_hit)
        s20   += float(top20_hit)
        ndcg10 += ndcg10_q
        n += 1

    return {
        "Success@5":  round(100 * s5  / n, 2) if n else 0.0,
        "Success@20": round(100 * s20 / n, 2) if n else 0.0,
        "nDCG@10":    round(100 * ndcg10 / n, 2) if n else 0.0,
        "num_queries": n,
    }


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(RANKING_DIR, exist_ok=True)

    t0 = time.time()

    # 1. 인코딩
    log("[1] Encoding passages...")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "encode"],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        raise RuntimeError("Encoding failed")

    # 2. float16 인덱스 빌드
    log("[2] Building float16 (analog) index...")
    _build_analog()

    # 3. 2-bit 인덱스 빌드
    log("[3] Building 2-bit index...")
    _build_2bit()

    # raw embeddings 삭제
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
        log("  raw_embs cleaned.")

    # 4. float16 검색
    log("[4] Searching with float16 index...")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "search_f16"],
        cwd=COLBERT_DIR
    ).returncode
    f16_ranking = os.path.join(RANKING_DIR, f"{ANALOG_NAME}.tsv")
    log("[4] Evaluating float16...")
    f16_metrics = evaluate(f16_ranking) if rc == 0 else {}
    log(f"  float16: {f16_metrics}")

    # 5. 2-bit 검색
    log("[5] Searching with 2-bit index...")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "search_2bit"],
        cwd=COLBERT_DIR
    ).returncode
    b2_ranking = os.path.join(RANKING_DIR, f"{BIT2_NAME}.tsv")
    log("[5] Evaluating 2-bit...")
    b2_metrics = evaluate(b2_ranking) if rc == 0 else {}
    log(f"  2-bit  : {b2_metrics}")

    # 6. 인덱스 삭제
    log("[6] Cleaning up indexes...")
    index_dir = os.path.join(INDEX_BASE, EXPERIMENT)
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        log(f"  Deleted: {index_dir}")

    elapsed = time.time() - t0

    # 7. 결과 저장
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    all_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            all_results = json.load(f)

    all_results["wiki_trivia"] = {
        "float16": f16_metrics,
        "2bit":    b2_metrics,
        "elapsed_min": round(elapsed / 60, 1),
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log(f"\nDone! elapsed={elapsed/60:.1f} min")
    log(f"Results saved -> {RESULTS_FILE}")


# ============================================================
# subprocess entry points
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "encode":
            _do_encode()
        elif cmd == "search_f16":
            _do_search(ANALOG_NAME, analog=True)
        elif cmd == "search_2bit":
            _do_search(BIT2_NAME, analog=False)
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        main()
