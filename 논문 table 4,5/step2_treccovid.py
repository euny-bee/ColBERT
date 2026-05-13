#!/usr/bin/env python3
"""
step2_treccovid.py
------------------
treccovid.analog / treccovid.2bit 인덱스 검색 + nDCG@10 계산.
TREC-COVID qrels는 graded relevance (0/1/2) -> gain 그대로 사용.

nDCG@10: graded relevance (0/1/2) 사용
  gain = score (0, 1, or 2)
  ideal DCG = 상위 k개 최대 가능 DCG

실행:
  python step2_treccovid.py
"""

import os
import sys
import json
import math
import time
import subprocess
import torch

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
INDEX_ROOT   = os.path.join(COLBERT_DIR, "experiments/treccovid/indexes")
RANKING_DIR  = os.path.join(COLBERT_DIR, "experiments/treccovid/rankings")
RESULTS_FILE = os.path.join(COLBERT_DIR, "experiments/treccovid/results.json")

QUERIES      = "D:/beir/trec-covid/queries.test.tsv"
QRELS        = "D:/beir/trec-covid/qrels.test.int.tsv"

ANALOG_INDEX = "treccovid.analog"
BIT2_INDEX   = "treccovid.2bit"
SEARCH_K     = 1000
NDOCS        = 4096

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# float16 index 검색용 monkey-patch
# ============================================================
def _apply_analog_patches():
    import tqdm
    from colbert.indexing.codecs import residual_embeddings as rem_mod
    from colbert.indexing.codecs.residual import ResidualCodec
    from colbert.utils.utils import print_message

    ResidualEmbeddings = rem_mod.ResidualEmbeddings

    def _init(self, codes, residuals):
        assert codes.size(0) == residuals.size(0)
        assert codes.dim() == 1 and residuals.dim() == 2
        self.codes     = codes.to(torch.int32)
        self.residuals = residuals

    ResidualEmbeddings.__init__ = _init

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
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    if os.path.exists(ranking_path):
        os.remove(ranking_path)
        log(f"  Removed existing ranking: {ranking_path}")

    if analog:
        _apply_analog_patches()

    torch.cuda.empty_cache()

    with Run().context(RunConfig(nranks=1, experiment="treccovid")):
        config   = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32,
                                 ndocs=NDOCS)
        searcher = Searcher(index=index_name, config=config)
        queries  = Queries(QUERIES)

        log(f"  Searching {len(queries)} queries (k={SEARCH_K}, ndocs={NDOCS})...")
        t0 = time.time()
        ranking = searcher.search_all(queries, k=SEARCH_K)
        log(f"  Done in {time.time() - t0:.1f}s")

    os.makedirs(RANKING_DIR, exist_ok=True)

    with open(ranking_path, "w") as f:
        for items in ranking.flat_ranking:
            f.write('\t'.join(str(x) for x in items) + '\n')
    log(f"  Saved -> {ranking_path}")


# ============================================================
# 평가: nDCG@10 (graded)
# ============================================================
def compute_ndcg10(ranking_path, qrels_path):
    # qrels: {qid -> {pid -> score}}  (score = 0/1/2)
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid, score = str(parts[0]), int(parts[1]), int(parts[2])
            qrels.setdefault(qid, {})[pid] = score

    # ranking: {qid -> [(rank, pid), ...]}
    rankings = {}
    with open(ranking_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid, rank = str(parts[0]), int(parts[1]), int(parts[2])
            rankings.setdefault(qid, []).append((rank, pid))

    ndcg10_sum = 0.0
    n = 0

    for qid, rel_dict in qrels.items():
        if qid not in rankings:
            continue
        # 이 쿼리에 relevant (score>=1) 이 하나도 없으면 skip
        if not any(s >= 1 for s in rel_dict.values()):
            continue
        n += 1

        ranked = [pid for _, pid in sorted(rankings[qid])]

        # DCG@10
        dcg = 0.0
        for rank, pid in enumerate(ranked[:10], 1):
            gain = rel_dict.get(pid, 0)
            if gain > 0:
                dcg += gain / math.log2(rank + 1)

        # ideal DCG@10: 상위 10개 gain을 내림차순 정렬
        all_gains = sorted(rel_dict.values(), reverse=True)[:10]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(all_gains) if g > 0)

        ndcg10_sum += dcg / idcg if idcg > 0 else 0.0

    return {
        "nDCG@10":    round(ndcg10_sum / n, 4) if n > 0 else 0.0,
        "num_queries": n,
    }


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(RANKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    results = {}

    # float16 검색
    log("\n[1] Searching float16 (analog) index...")
    analog_ranking = os.path.join(RANKING_DIR, "treccovid.analog.tsv")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "search_analog"],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        log("ERROR: float16 search failed")
    elif os.path.exists(analog_ranking):
        m = compute_ndcg10(analog_ranking, QRELS)
        results["float16"] = m
        log(f"  float16  nDCG@10={m['nDCG@10']}  (n={m['num_queries']})")

    # 2-bit 검색
    log("\n[2] Searching 2-bit index...")
    bit2_ranking = os.path.join(RANKING_DIR, "treccovid.2bit.tsv")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "search_2bit"],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        log("ERROR: 2-bit search failed")
    elif os.path.exists(bit2_ranking):
        m = compute_ndcg10(bit2_ranking, QRELS)
        results["2bit"] = m
        log(f"  2-bit    nDCG@10={m['nDCG@10']}  (n={m['num_queries']})")

    # 결과 저장
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved -> {RESULTS_FILE}")

    # 비교 테이블
    f16 = results.get("float16", {})
    b2  = results.get("2bit",    {})

    print(f"\n{'='*50}")
    print(f"  TREC-COVID - float16 vs 2-bit")
    print(f"{'='*50}")
    print(f"  {'Metric':<12} {'float16':>10} {'2-bit':>10} {'Delta':>10}")
    print(f"  {'-'*45}")
    v_f16 = f16.get("nDCG@10")
    v_b2  = b2.get("nDCG@10")
    if isinstance(v_f16, float) and isinstance(v_b2, float):
        delta = v_f16 - v_b2
        sign  = "+" if delta >= 0 else ""
        print(f"  {'nDCG@10':<12} {v_f16:>10.4f} {v_b2:>10.4f} {sign}{delta:>9.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "search_analog":
            analog_ranking = os.path.join(RANKING_DIR, "treccovid.analog.tsv")
            _search(ANALOG_INDEX, analog_ranking, analog=True)
        elif cmd == "search_2bit":
            bit2_ranking = os.path.join(RANKING_DIR, "treccovid.2bit.tsv")
            _search(BIT2_INDEX, bit2_ranking, analog=False)
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        main()
