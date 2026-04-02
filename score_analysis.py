#!/usr/bin/env python3
"""
score_analysis.py
-----------------
float16 vs 2-bit MaxSim 스코어 직접 비교.

방법:
  1. 100개 쿼리 샘플 (seed=42)
  2. 각 쿼리 → float16으로 상위 200개 검색
  3. 동일한 쿼리 → 2-bit로 상위 200개 검색
  4. 교집합 PID에 대해 스코어 비교
  5. 분포(mean, std), Spearman ρ, r², top-50 overlap 계산

실행:
  python score_analysis.py
"""

import os, sys, json, random, subprocess
import numpy as np
from scipy import stats

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
QUERIES_FILE = os.path.join(COLBERT_DIR, "data/msmarco/queries.dev.small.tsv")
RESULTS_DIR  = os.path.join(COLBERT_DIR, "experiments/msmarco")
ANALOG_INDEX = "200k.analog"
BIT2_INDEX   = "200k.2bit"
N_QUERIES    = 100
K            = 200
SEED         = 42

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def load_queries(path, n, seed=SEED):
    queries = {}
    with open(path) as f:
        for line in f:
            qid, text = line.strip().split('\t', 1)
            queries[qid] = text
    random.seed(seed)
    return dict(random.sample(list(queries.items()), n))


def _apply_analog_patches():
    import tqdm, torch
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
    def _load_chunks(cls, index_path, chunk_idxs, num_embeddings, load_index_with_mmap=False):
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


def _run_search(index_name, queries, out_path, analog=False):
    """단일 인덱스로 검색 후 {qid: {pid: score}} JSON 저장."""
    import torch
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig

    if analog:
        _apply_analog_patches()

    torch.cuda.empty_cache()
    results = {}

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config   = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32,
                                 ncells=2, ndocs=8192)
        searcher = Searcher(index=index_name, config=config)
        for qid, text in queries.items():
            pids, _, scores = searcher.search(text, k=K)
            results[qid] = {str(p): float(s) for p, s in zip(pids, scores)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} queries → {out_path}")


def analyze(res_f16, res_2bit, queries):
    score_diffs    = []
    spearman_rhos  = []
    r2_vals        = []
    top50_overlaps = []

    for qid in queries:
        f16 = res_f16.get(qid, {})
        b2  = res_2bit.get(qid, {})
        common = sorted(set(f16) & set(b2), key=lambda p: -f16[p])
        if len(common) < 10:
            continue

        s_f16 = np.array([f16[p] for p in common])
        s_b2  = np.array([b2[p]  for p in common])

        score_diffs.extend((s_f16 - s_b2).tolist())

        rho, _ = stats.spearmanr(s_f16, s_b2)
        spearman_rhos.append(rho)

        r, _ = stats.pearsonr(s_f16, s_b2)
        r2_vals.append(r ** 2)

        top50_f16 = set(sorted(f16, key=f16.get, reverse=True)[:50])
        top50_b2  = set(sorted(b2,  key=b2.get,  reverse=True)[:50])
        top50_overlaps.append(len(top50_f16 & top50_b2) / 50)

    d = np.array(score_diffs)

    print(f"\n{'='*60}")
    print(f"  float16 vs 2-bit  MaxSim Score 비교")
    print(f"  (샘플: {len(queries)} queries, k={K} candidates each)")
    print(f"{'='*60}")
    print(f"  분석된 쿼리 수          : {len(spearman_rhos)}")
    print(f"  교집합 (query,doc) 쌍   : {len(d):,}")
    print(f"  Score diff  mean        : {d.mean():+.4f}")
    print(f"  Score diff  std         : {d.std():.4f}")
    print(f"  Score diff  |max|       : {np.abs(d).max():.4f}")
    print(f"  Spearman ρ  (mean±std)  : {np.mean(spearman_rhos):.4f} ± {np.std(spearman_rhos):.4f}")
    print(f"  r²          (mean±std)  : {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
    print(f"  Top-50 overlap (mean)   : {np.mean(top50_overlaps):.1%}")
    print(f"{'='*60}")

    return {
        "n_queries":          len(queries),
        "n_analyzed":         len(spearman_rhos),
        "n_pairs":            len(d),
        "score_diff_mean":    float(d.mean()),
        "score_diff_std":     float(d.std()),
        "score_diff_abs_max": float(np.abs(d).max()),
        "spearman_rho_mean":  float(np.mean(spearman_rhos)),
        "spearman_rho_std":   float(np.std(spearman_rhos)),
        "r2_mean":            float(np.mean(r2_vals)),
        "r2_std":             float(np.std(r2_vals)),
        "top50_overlap_mean": float(np.mean(top50_overlaps)),
    }


def main():
    queries = load_queries(QUERIES_FILE, N_QUERIES)
    print(f"Sampled {len(queries)} queries (seed={SEED})")

    queries_path = os.path.join(RESULTS_DIR, "sample_queries.json")
    analog_out   = os.path.join(RESULTS_DIR, "scores_analog.json")
    bit2_out     = os.path.join(RESULTS_DIR, "scores_2bit.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(queries_path, "w") as f:
        json.dump(queries, f)

    # float16 검색 (서브프로세스 — analog 패치 격리)
    print("\n[1] Searching float16 index (100 queries × k=200)...")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         "search_analog", queries_path, analog_out],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        print("ERROR: float16 search failed"); sys.exit(1)

    # 2-bit 검색 (서브프로세스)
    print("\n[2] Searching 2-bit index (100 queries × k=200)...")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         "search_2bit", queries_path, bit2_out],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        print("ERROR: 2-bit search failed"); sys.exit(1)

    # 분석
    print("\n[3] Computing statistics...")
    with open(analog_out) as f: res_f16  = json.load(f)
    with open(bit2_out)   as f: res_2bit = json.load(f)

    result = analyze(res_f16, res_2bit, queries)

    out_path = os.path.join(RESULTS_DIR, "score_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd, queries_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
        with open(queries_path) as f:
            queries = json.load(f)
        if cmd == "search_analog":
            _run_search(ANALOG_INDEX, queries, out_path, analog=True)
        elif cmd == "search_2bit":
            _run_search(BIT2_INDEX,   queries, out_path, analog=False)
        else:
            print(f"Unknown command: {cmd}"); sys.exit(1)
    else:
        main()
