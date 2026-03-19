#!/usr/bin/env python3
"""
step2_eval.py
-------------
float16 / 2-bit index 검색 + MRR@10, R@50 계산.

실행:
  python step2_eval.py
"""

import os
import sys
import json
import time
import subprocess
import torch

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
INDEX_ROOT   = os.path.join(COLBERT_DIR, "experiments/msmarco/indexes")
RANKING_DIR  = os.path.join(COLBERT_DIR, "experiments/msmarco/rankings")
RESULTS_FILE = os.path.join(COLBERT_DIR, "experiments/msmarco/results.json")

QUERIES      = os.path.join(COLBERT_DIR, "data/msmarco/queries.dev.small.tsv")
QRELS        = os.path.join(COLBERT_DIR, "data/msmarco/subset/qrels.tsv")

ANALOG_INDEX = "200k.analog"
BIT2_INDEX   = "200k.2bit"
SEARCH_K     = 50

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# float16 index 검색용 runtime monkey-patch
# ============================================================
def _apply_analog_patches():
    """float16 residuals를 처리할 수 있도록 런타임 패치."""
    import tqdm
    from colbert.indexing.codecs import residual_embeddings as rem_mod
    from colbert.indexing.codecs.residual import ResidualCodec
    from colbert.utils.utils import print_message

    ResidualEmbeddings = rem_mod.ResidualEmbeddings

    # Patch 1: __init__ — uint8 assert 제거
    def _init(self, codes, residuals):
        assert codes.size(0) == residuals.size(0)
        assert codes.dim() == 1 and residuals.dim() == 2
        self.codes     = codes.to(torch.int32)
        self.residuals = residuals

    ResidualEmbeddings.__init__ = _init

    # Patch 2: load_chunks — dtype을 첫 청크에서 감지
    @classmethod
    def _load_chunks(cls, index_path, chunk_idxs, num_embeddings,
                     load_index_with_mmap=False):
        num_embeddings += 512
        print_message("#> Loading codes and residuals (float16 mode)...")

        first = cls.load(index_path, chunk_idxs[0])
        res_dim   = first.residuals.shape[1]
        res_dtype = first.residuals.dtype

        codes     = torch.empty(num_embeddings, dtype=torch.int32)
        residuals = torch.empty(num_embeddings, res_dim, dtype=res_dtype)

        offset = 0
        for i, idx in enumerate(tqdm.tqdm(chunk_idxs)):
            chunk = first if i == 0 else cls.load(index_path, idx)
            end   = offset + chunk.codes.size(0)
            codes[offset:end]     = chunk.codes
            residuals[offset:end] = chunk.residuals
            offset = end

        return cls(codes, residuals)

    ResidualEmbeddings.load_chunks = _load_chunks

    # Patch 3: decompress — centroid + float16 residual 덧셈
    def _decompress(self, compressed_embs):
        codes, residuals = compressed_embs.codes, compressed_embs.residuals
        D = []
        for codes_, res_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            if self.use_gpu:
                codes_ = codes_.cuda()
                res_   = res_.cuda().half()
            else:
                res_ = res_.float()
            centroids_ = self.lookup_centroids(codes_, out_device=codes_.device)
            vec_ = centroids_ + res_
            if self.use_gpu:
                D.append(torch.nn.functional.normalize(vec_, p=2, dim=-1).half())
            else:
                D.append(torch.nn.functional.normalize(vec_.float(), p=2, dim=-1))
        return torch.cat(D)

    ResidualCodec.decompress = _decompress
    log("  Analog patches applied.")


# ============================================================
# 검색
# ============================================================
def _search(index_name, ranking_path, analog=False):
    """index_name으로 검색 후 ranking_path에 저장."""
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    if analog:
        _apply_analog_patches()

    torch.cuda.empty_cache()

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config  = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32)
        searcher = Searcher(index=index_name, config=config)
        queries  = Queries(QUERIES)

        log(f"  Searching {len(queries)} queries (k={SEARCH_K})...")
        t0 = time.time()
        ranking = searcher.search_all(queries, k=SEARCH_K)
        log(f"  Done in {time.time() - t0:.0f}s")

        os.makedirs(os.path.dirname(ranking_path), exist_ok=True)
        ranking.save(ranking_path)
        log(f"  Saved → {ranking_path}")


# ============================================================
# 평가: MRR@10, R@50
# ============================================================
def compute_metrics(ranking_path, qrels_path):
    """MRR@10, R@50 계산."""
    # qrels 로드: {qid -> set(pid)}
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            qid, pid = parts[0], int(parts[2])
            qrels.setdefault(qid, set()).add(pid)

    # ranking 로드: {qid -> [(rank, pid), ...]}
    rankings = {}
    with open(ranking_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            qid, pid, rank = parts[0], int(parts[1]), int(parts[2])
            rankings.setdefault(qid, []).append((rank, pid))

    mrr10 = 0.0
    r50   = 0.0
    n     = 0

    for qid, relevant in qrels.items():
        if qid not in rankings:
            continue
        n += 1
        ranked = [pid for _, pid in sorted(rankings[qid])]

        # MRR@10
        for rank, pid in enumerate(ranked[:10], 1):
            if pid in relevant:
                mrr10 += 1.0 / rank
                break

        # R@50
        if any(pid in relevant for pid in ranked[:50]):
            r50 += 1.0

    return {
        "MRR@10": round(mrr10 / n, 4),
        "R@50":   round(r50   / n, 4),
        "num_queries": n,
    }


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(RANKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    results = {}

    # --- float16 검색 (subprocess → 독립 프로세스에서 패치 적용)
    log("\n[1] Searching float16 (analog) index...")
    analog_ranking = os.path.join(RANKING_DIR, "200k.analog.tsv")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "search_analog"],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        log("ERROR: float16 search failed")
    elif os.path.exists(analog_ranking):
        m = compute_metrics(analog_ranking, QRELS)
        results["float16"] = m
        log(f"  float16  MRR@10={m['MRR@10']}  R@50={m['R@50']}  (n={m['num_queries']})")

    # --- 2-bit 검색
    log("\n[2] Searching 2-bit (from float16) index...")
    bit2_ranking = os.path.join(RANKING_DIR, "200k.2bit.tsv")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "search_2bit"],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        log("ERROR: 2-bit search failed")
    elif os.path.exists(bit2_ranking):
        m = compute_metrics(bit2_ranking, QRELS)
        results["2bit"] = m
        log(f"  2-bit    MRR@10={m['MRR@10']}  R@50={m['R@50']}  (n={m['num_queries']})")

    # --- 결과 저장
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved → {RESULTS_FILE}")

    # --- 비교 테이블
    f16 = results.get("float16", {})
    b2  = results.get("2bit",    {})

    print(f"\n{'='*52}")
    print(f"  200K MS MARCO — float16 vs 2-bit (from float16)")
    print(f"{'='*52}")
    print(f"  {'Metric':<10} {'float16':>10} {'2-bit':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    for metric in ["MRR@10", "R@50"]:
        v_f16 = f16.get(metric)
        v_b2  = b2.get(metric)
        if isinstance(v_f16, float) and isinstance(v_b2, float):
            delta = v_f16 - v_b2
            sign  = "+" if delta >= 0 else ""
            print(f"  {metric:<10} {v_f16:>10.4f} {v_b2:>10.4f} {sign}{delta:>9.4f}")
        else:
            print(f"  {metric:<10} {'N/A':>10} {'N/A':>10}")
    print(f"{'='*52}")
    print(f"  (논문 2-bit 기준: MRR@10=0.3970  R@50=0.8650)")
    print(f"  * 200K 서브셋이라 절대값은 높음. delta가 핵심.")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "search_analog":
            analog_ranking = os.path.join(RANKING_DIR, "200k.analog.tsv")
            _search(ANALOG_INDEX, analog_ranking, analog=True)
        elif cmd == "search_2bit":
            bit2_ranking = os.path.join(RANKING_DIR, "200k.2bit.tsv")
            _search(BIT2_INDEX, bit2_ranking, analog=False)
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        main()
