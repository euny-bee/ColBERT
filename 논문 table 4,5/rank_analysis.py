#!/usr/bin/env python3
"""
rank_analysis.py
----------------
float16 vs 2-bit: R@k comparison and rank displacement analysis.

Steps:
  1. Search both indexes with k=1000 (500 queries)
  2. Load qrels (subset)
  3. Compute R@k for k=1,3,5,10,20,50,100,200,500,1000
  4. Rank displacement: rank_f16 vs rank_2bit for relevant docs

Usage:
  python rank_analysis.py
  python rank_analysis.py search_analog <queries_json> <out_json>
  python rank_analysis.py search_2bit   <queries_json> <out_json>
"""

import os, sys, json, random, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
QUERIES_FILE = os.path.join(COLBERT_DIR, "data/msmarco/queries.dev.small.tsv")
QRELS_FILE   = os.path.join(COLBERT_DIR, "data/msmarco/subset/qrels.tsv")
RESULTS_DIR  = os.path.join(COLBERT_DIR, "experiments/msmarco")
ANALOG_INDEX = "200k.analog"
BIT2_INDEX   = "200k.2bit"
N_QUERIES    = 500
K            = 1000
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


def load_qrels(path):
    """Returns {qid: set(pid)} for relevant docs."""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                parts = line.strip().split()
            qid, _, pid, rel = parts[0], parts[1], parts[2], parts[3]
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(pid)
    return qrels


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
    print(f"Saved {len(results)} queries -> {out_path}")


def compute_recall_at_k(results, qrels, ks):
    """
    Compute R@k for each k.
    Returns {k: mean_recall}.
    """
    recall = {k: [] for k in ks}
    for qid, pid_scores in results.items():
        rel = qrels.get(qid, set())
        if not rel:
            continue
        ranked = sorted(pid_scores.keys(), key=lambda p: -pid_scores[p])
        for k in ks:
            top_k = set(ranked[:k])
            r = len(rel & top_k) / len(rel)
            recall[k].append(r)
    return {k: np.mean(v) for k, v in recall.items() if v}


def compute_rank_displacement(res_f16, res_2bit, qrels, queries):
    """
    For each (qid, relevant_pid) pair:
      rank_f16, rank_2bit = position in top-K results (1-indexed)
    Only counts pairs where relevant doc appears in BOTH top-K lists.
    Returns list of dicts.
    """
    pairs = []
    for qid in queries:
        f16 = res_f16.get(qid, {})
        b2  = res_2bit.get(qid, {})
        rel = qrels.get(qid, set())
        if not rel:
            continue

        ranked_f16 = sorted(f16.keys(), key=lambda p: -f16[p])
        ranked_b2  = sorted(b2.keys(),  key=lambda p: -b2[p])

        rank_map_f16 = {p: i+1 for i, p in enumerate(ranked_f16)}
        rank_map_b2  = {p: i+1 for i, p in enumerate(ranked_b2)}

        for pid in rel:
            if pid in rank_map_f16 and pid in rank_map_b2:
                pairs.append({
                    "qid":       qid,
                    "pid":       pid,
                    "rank_f16":  rank_map_f16[pid],
                    "rank_2bit": rank_map_b2[pid],
                })
    return pairs


def main():
    queries = load_queries(QUERIES_FILE, N_QUERIES)
    print(f"Sampled {len(queries)} queries (seed={SEED})")

    queries_path = os.path.join(RESULTS_DIR, "sample_queries_500q.json")
    analog_out   = os.path.join(RESULTS_DIR, "scores_analog_k1000_500q.json")
    bit2_out     = os.path.join(RESULTS_DIR, "scores_2bit_k1000_500q.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(queries_path, "w") as f:
        json.dump(queries, f)

    # float16 search
    print("\n[1] Searching float16 index (k=1000)...")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         "search_analog", queries_path, analog_out],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        print("ERROR: float16 search failed"); sys.exit(1)

    # 2-bit search
    print("\n[2] Searching 2-bit index (k=1000)...")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         "search_2bit", queries_path, bit2_out],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        print("ERROR: 2-bit search failed"); sys.exit(1)

    # Load results
    print("\n[3] Loading results and qrels...")
    with open(analog_out) as f: res_f16  = json.load(f)
    with open(bit2_out)   as f: res_2bit = json.load(f)
    qrels = load_qrels(QRELS_FILE)
    print(f"  qrels loaded: {len(qrels)} queries with relevant docs")

    # R@k
    ks = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    recall_f16  = compute_recall_at_k(res_f16,  qrels, ks)
    recall_2bit = compute_recall_at_k(res_2bit, qrels, ks)

    print(f"\n{'='*55}")
    print(f"  R@k comparison  (float16 vs 2-bit)")
    print(f"{'='*55}")
    print(f"  {'k':>6}  {'float16':>10}  {'2-bit':>10}  {'delta':>10}")
    for k in ks:
        r16 = recall_f16.get(k, float('nan'))
        r2  = recall_2bit.get(k, float('nan'))
        print(f"  {k:>6}  {r16:>10.4f}  {r2:>10.4f}  {r2-r16:>+10.4f}")

    # Rank displacement
    pairs = compute_rank_displacement(res_f16, res_2bit, qrels, queries)
    print(f"\n  Rank displacement: {len(pairs)} (query, relevant_doc) pairs")

    disp = np.array([p["rank_2bit"] - p["rank_f16"] for p in pairs])
    rank_f16_arr = np.array([p["rank_f16"]  for p in pairs])
    rank_2bit_arr = np.array([p["rank_2bit"] for p in pairs])

    print(f"  Displacement mean : {disp.mean():+.2f}")
    print(f"  Displacement std  : {disp.std():.2f}")
    print(f"  Displacement |max|: {np.abs(disp).max():.0f}")
    print(f"  2-bit ranks worse (>0): {(disp > 0).mean():.1%}")
    print(f"  2-bit ranks same  (=0): {(disp == 0).mean():.1%}")
    print(f"  2-bit ranks better(<0): {(disp < 0).mean():.1%}")
    print(f"{'='*55}")

    # Save JSON
    result = {
        "n_queries":    N_QUERIES,
        "k":            K,
        "recall_f16":   {str(k): v for k, v in recall_f16.items()},
        "recall_2bit":  {str(k): v for k, v in recall_2bit.items()},
        "n_pairs":      len(pairs),
        "disp_mean":    float(disp.mean()),
        "disp_std":     float(disp.std()),
        "disp_abs_max": float(np.abs(disp).max()),
        "pct_worse":    float((disp > 0).mean()),
        "pct_same":     float((disp == 0).mean()),
        "pct_better":   float((disp < 0).mean()),
    }
    out_json = os.path.join(RESULTS_DIR, "rank_analysis_500q.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_json}")

    # ── Plots ────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "float16 vs 2-bit  Rank Analysis  (100 queries, k=1000)",
        fontsize=13, fontweight="bold"
    )

    # [0] R@k line chart
    ax = axes[0]
    ks_plot = sorted(recall_f16.keys())
    ax.plot(ks_plot, [recall_f16[k]  for k in ks_plot],
            marker="o", lw=2.0, color="steelblue",  label="float16")
    ax.plot(ks_plot, [recall_2bit[k] for k in ks_plot],
            marker="s", lw=2.0, color="darkorange", label="2-bit",
            linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("k (cutoff)")
    ax.set_ylabel("Recall@k")
    ax.set_title("R@k Curve")
    ax.legend(fontsize=10)
    ax.set_xticks(ks_plot)
    ax.set_xticklabels([str(k) for k in ks_plot], rotation=45, fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))

    # [1] Rank scatter
    ax = axes[1]
    ax.scatter(rank_f16_arr, rank_2bit_arr,
               alpha=0.3, s=8, color="seagreen", rasterized=True)
    lo, hi = 1, K
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x  (no change)")
    ax.set_xlabel("Rank (float16)")
    ax.set_ylabel("Rank (2-bit)")
    ax.set_title(f"Rank Displacement Scatter\n({len(pairs)} relevant-doc pairs)")
    ax.legend(fontsize=9)

    # [2] Displacement histogram
    ax = axes[2]
    clip = 200  # clip extreme outliers for readability
    disp_clipped = np.clip(disp, -clip, clip)
    ax.hist(disp_clipped, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0,           color="black",  lw=1.0, linestyle="-",  alpha=0.6)
    ax.axvline(disp.mean(), color="red",    lw=1.6, linestyle="--",
               label=f"mean = {disp.mean():+.1f}")
    ax.set_xlabel(f"rank_2bit − rank_f16  (clipped to ±{clip})")
    ax.set_ylabel("Count")
    ax.set_title("Rank Displacement Distribution\n"
                 f"worse: {(disp>0).mean():.1%}  same: {(disp==0).mean():.1%}  "
                 f"better: {(disp<0).mean():.1%}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "rank_analysis_500q.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_png}")


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
