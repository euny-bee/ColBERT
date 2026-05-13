#!/usr/bin/env python3
"""
step2_lotte_writing.py
------------------------
lifestyle.analog / lifestyle.2bit 인덱스 검색 + 평가.

메트릭: S@5, nDCG@10, MRR@10, R@50, R@1k
쿼리:   Search / Forum 각각 평가

실행:
  python step2_lotte_writing.py
"""

import os
import sys
import json
import math
import time
import subprocess
import torch

COLBERT_DIR    = os.path.expanduser("~/ColBERT")
TOPIC          = "writing"
DATA_DIR       = f"D:/beir/lotte/{TOPIC}"
QUERIES_SEARCH = f"{DATA_DIR}/queries.search.tsv"
QUERIES_FORUM  = f"{DATA_DIR}/queries.forum.tsv"
QAS_SEARCH     = f"{DATA_DIR}/qas.search.jsonl"
QAS_FORUM      = f"{DATA_DIR}/qas.forum.jsonl"

INDEX_ROOT   = os.path.join(COLBERT_DIR, "experiments/lotte_writing/indexes")
RANKING_DIR  = os.path.join(COLBERT_DIR, f"experiments/lotte/rankings/{TOPIC}")
RESULTS_FILE = os.path.join(COLBERT_DIR, f"experiments/lotte/results/{TOPIC}.json")

ANALOG_INDEX = f"{TOPIC}.analog"
BIT2_INDEX   = f"{TOPIC}.2bit"
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

        first     = cls.load(index_path, chunk_idxs[0])
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
def _search(index_name, queries_path, ranking_path, analog=False):
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    if os.path.exists(ranking_path):
        os.remove(ranking_path)

    if analog:
        _apply_analog_patches()

    torch.cuda.empty_cache()

    with Run().context(RunConfig(nranks=1, experiment=f"lotte_{TOPIC}")):
        config   = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32,
                                 ndocs=NDOCS)
        searcher = Searcher(index=index_name, config=config)
        queries  = Queries(queries_path)

        log(f"  Searching {len(queries)} queries (k={SEARCH_K}, ndocs={NDOCS})...")
        t0 = time.time()
        ranking = searcher.search_all(queries, k=SEARCH_K)
        log(f"  Done in {time.time()-t0:.1f}s")

    os.makedirs(RANKING_DIR, exist_ok=True)
    with open(ranking_path, "w", encoding="utf-8") as f:
        for items in ranking.flat_ranking:
            f.write('\t'.join(str(x) for x in items) + '\n')
    log(f"  Saved -> {ranking_path}")


# ============================================================
# 평가: S@5, nDCG@10, MRR@10, R@50, R@1k
# ============================================================
def compute_metrics(ranking_path, qas_path):
    import json

    # qas.jsonl: {qid, query, answer_pids}
    qrels = {}
    with open(qas_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["qid"])
            qrels[qid] = set(int(p) for p in row["answer_pids"])

    # ranking: {qid -> [(rank, pid), ...]}
    rankings = {}
    with open(ranking_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("	")
            if len(parts) < 3:
                continue
            qid, pid, rank = str(parts[0]), int(parts[1]), int(parts[2])
            rankings.setdefault(qid, []).append((rank, pid))

    s5    = 0.0
    ndcg  = 0.0
    mrr   = 0.0
    r50   = 0.0
    r1k   = 0.0
    n     = 0

    for qid, relevant in qrels.items():
        if qid not in rankings:
            continue
        n += 1
        ranked = [pid for _, pid in sorted(rankings[qid])]

        # S@5
        if any(pid in relevant for pid in ranked[:5]):
            s5 += 1.0

        # nDCG@10
        dcg = 0.0
        for rank, pid in enumerate(ranked[:10], 1):
            if pid in relevant:
                dcg += 1.0 / __import__("math").log2(rank + 1)
        ideal_n = min(len(relevant), 10)
        idcg = sum(1.0 / __import__("math").log2(i + 2) for i in range(ideal_n))
        ndcg += dcg / idcg if idcg > 0 else 0.0

        # MRR@10
        for rank, pid in enumerate(ranked[:10], 1):
            if pid in relevant:
                mrr += 1.0 / rank
                break

        # R@50
        if any(pid in relevant for pid in ranked[:50]):
            r50 += 1.0

        # R@1k
        if any(pid in relevant for pid in ranked[:1000]):
            r1k += 1.0

    return {
        "S@5":     round(s5   / n, 4) if n > 0 else 0.0,
        "nDCG@10": round(ndcg / n, 4) if n > 0 else 0.0,
        "MRR@10":  round(mrr  / n, 4) if n > 0 else 0.0,
        "R@50":    round(r50  / n, 4) if n > 0 else 0.0,
        "R@1k":    round(r1k  / n, 4) if n > 0 else 0.0,
        "num_queries": n,
    }


# ============================================================
# 결과 테이블 출력
# ============================================================
def _print_table(results):
    METRICS = ["S@5", "nDCG@10", "MRR@10", "R@50", "R@1k"]
    W = 80

    print(f"\n{'='*W}")
    print(f"  LoTTE [{TOPIC.capitalize()}] Results - float16 vs 2-bit")
    print(f"{'='*W}")

    for qtype in ["search", "forum"]:
        label = "Search Queries" if qtype == "search" else "Forum Queries"
        print(f"\n  ── {label} {'─'*(W-6-len(label))}")
        print(f"\n  {'Topic':<14} {'Metric':<10} {'float16':>10} {'2-bit':>10} {'Delta':>10}")
        print(f"  {'-'*(W-2)}")

        f16 = results.get(f"{qtype}_float16", {})
        b2  = results.get(f"{qtype}_2bit",    {})

        first = True
        for metric in METRICS:
            topic_label = TOPIC.capitalize() if first else ""
            first = False
            v_f16 = f16.get(metric)
            v_b2  = b2.get(metric)
            if isinstance(v_f16, float) and isinstance(v_b2, float):
                delta = v_f16 - v_b2
                sign  = "+" if delta >= 0 else ""
                print(f"  {topic_label:<14} {metric:<10} {v_f16:>10.4f} {v_b2:>10.4f} {sign}{delta:>9.4f}")
            else:
                print(f"  {topic_label:<14} {metric:<10} {'N/A':>10} {'N/A':>10}")

    print(f"\n{'='*W}\n")


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(RANKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    results = {}

    for qtype, queries_path, qrels_path in [
        ("search", QUERIES_SEARCH, QAS_SEARCH),
        ("forum",  QUERIES_FORUM,  QAS_FORUM),
    ]:
        # float16 검색
        log(f"\n[{qtype.upper()}] Searching float16 (analog) index...")
        analog_ranking = os.path.join(RANKING_DIR, f"{TOPIC}.analog.{qtype}.tsv")
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             f"search_analog_{qtype}"],
            cwd=COLBERT_DIR
        ).returncode
        if rc != 0:
            log(f"ERROR: float16 {qtype} search failed")
        elif os.path.exists(analog_ranking):
            m = compute_metrics(analog_ranking, qrels_path)
            results[f"{qtype}_float16"] = m
            log(f"  float16 [{qtype}]  S@5={m['S@5']}  nDCG@10={m['nDCG@10']}  "
                f"MRR@10={m['MRR@10']}  R@50={m['R@50']}  R@1k={m['R@1k']}  "
                f"(n={m['num_queries']})")

        # 2-bit 검색
        log(f"\n[{qtype.upper()}] Searching 2-bit index...")
        bit2_ranking = os.path.join(RANKING_DIR, f"{TOPIC}.2bit.{qtype}.tsv")
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             f"search_2bit_{qtype}"],
            cwd=COLBERT_DIR
        ).returncode
        if rc != 0:
            log(f"ERROR: 2-bit {qtype} search failed")
        elif os.path.exists(bit2_ranking):
            m = compute_metrics(bit2_ranking, qrels_path)
            results[f"{qtype}_2bit"] = m
            log(f"  2-bit   [{qtype}]  S@5={m['S@5']}  nDCG@10={m['nDCG@10']}  "
                f"MRR@10={m['MRR@10']}  R@50={m['R@50']}  R@1k={m['R@1k']}  "
                f"(n={m['num_queries']})")

    # 결과 저장
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved -> {RESULTS_FILE}")

    # 테이블 출력
    _print_table(results)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "search_analog_search":
            ranking = os.path.join(RANKING_DIR, f"{TOPIC}.analog.search.tsv")
            _search(ANALOG_INDEX, QUERIES_SEARCH, ranking, analog=True)
        elif cmd == "search_2bit_search":
            ranking = os.path.join(RANKING_DIR, f"{TOPIC}.2bit.search.tsv")
            _search(BIT2_INDEX, QUERIES_SEARCH, ranking, analog=False)
        elif cmd == "search_analog_forum":
            ranking = os.path.join(RANKING_DIR, f"{TOPIC}.analog.forum.tsv")
            _search(ANALOG_INDEX, QUERIES_FORUM, ranking, analog=True)
        elif cmd == "search_2bit_forum":
            ranking = os.path.join(RANKING_DIR, f"{TOPIC}.2bit.forum.tsv")
            _search(BIT2_INDEX, QUERIES_FORUM, ranking, analog=False)
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        main()
