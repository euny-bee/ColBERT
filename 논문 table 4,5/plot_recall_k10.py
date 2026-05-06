#!/usr/bin/env python3
"""
plot_recall_k10.py
------------------
R@k (k=1..10) 그래프만 재생성. 기존 500q 검색 결과 재사용.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

COLBERT_DIR = os.path.expanduser("~/ColBERT")
RESULTS_DIR = os.path.join(COLBERT_DIR, "experiments/msmarco")
QRELS_FILE  = os.path.join(COLBERT_DIR, "data/msmarco/subset/qrels.tsv")

with open(os.path.join(RESULTS_DIR, "scores_analog_k1000_500q.json")) as f: res_f16  = json.load(f)
with open(os.path.join(RESULTS_DIR, "scores_2bit_k1000_500q.json"))   as f: res_2bit = json.load(f)
with open(os.path.join(RESULTS_DIR, "sample_queries_500q.json"))       as f: queries  = json.load(f)

qrels = {}
with open(QRELS_FILE) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 4:
            parts = line.strip().split()
        qid, pid, rel = parts[0], parts[2], parts[3]
        if int(rel) > 0:
            qrels.setdefault(qid, set()).add(pid)

ks = list(range(1, 11))

def recall_at_k(results, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, pid_scores in results.items():
        rel = qrels.get(qid, set())
        if not rel:
            continue
        ranked = sorted(pid_scores.keys(), key=lambda p: -pid_scores[p])
        for k in ks:
            top_k = set(ranked[:k])
            scores[k].append(len(rel & top_k) / len(rel))
    return {k: np.mean(v) for k, v in scores.items() if v}

def mrr_at_k(results, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, pid_scores in results.items():
        rel = qrels.get(qid, set())
        if not rel:
            continue
        ranked = sorted(pid_scores.keys(), key=lambda p: -pid_scores[p])
        rr = 0.0
        for rank, pid in enumerate(ranked, 1):
            if pid in rel:
                rr = 1.0 / rank
                break
        for k in ks:
            scores[k].append(rr if (rr == 0 or 1/rr <= k) else 0.0)
    return {k: np.mean(v) for k, v in scores.items() if v}

def ndcg_at_k(results, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, pid_scores in results.items():
        rel = qrels.get(qid, set())
        if not rel:
            continue
        ranked = sorted(pid_scores.keys(), key=lambda p: -pid_scores[p])
        # IDCG@k: ideal = all relevant docs at top ranks
        for k in ks:
            dcg = sum(1.0 / np.log2(rank + 1)
                      for rank, pid in enumerate(ranked[:k], 1)
                      if pid in rel)
            idcg = sum(1.0 / np.log2(rank + 1)
                       for rank in range(1, min(k, len(rel)) + 1))
            scores[k].append(dcg / idcg if idcg > 0 else 0.0)
    return {k: np.mean(v) for k, v in scores.items() if v}

r_f16    = recall_at_k(res_f16,  qrels, ks)
r_2bit   = recall_at_k(res_2bit, qrels, ks)
mrr_f16  = mrr_at_k(res_f16,  qrels, ks)
mrr_2bit = mrr_at_k(res_2bit, qrels, ks)
ndcg_f16  = ndcg_at_k(res_f16,  qrels, ks)
ndcg_2bit = ndcg_at_k(res_2bit, qrels, ks)

print(f"{'k':>4}  {'R@k f16':>9}  {'R@k 2bit':>9}  {'MRR@k f16':>11}  {'MRR@k 2bit':>11}  {'nDCG@k f16':>12}  {'nDCG@k 2bit':>12}")
for k in ks:
    print(f"{k:>4}  {r_f16[k]:>9.4f}  {r_2bit[k]:>9.4f}  {mrr_f16[k]:>11.4f}  {mrr_2bit[k]:>11.4f}  {ndcg_f16[k]:>12.4f}  {ndcg_2bit[k]:>12.4f}")

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle("float16 vs 2-bit  (k=1..10)  |  500 queries, 200k MS MARCO subset",
             fontsize=12, fontweight="bold")

for ax, vals_f16, vals_2bit, ylabel, title in zip(
    axes,
    [r_f16,   mrr_f16,   ndcg_f16],
    [r_2bit,  mrr_2bit,  ndcg_2bit],
    ["Recall@k", "MRR@k", "nDCG@k"],
    ["R@k",      "MRR@k", "nDCG@k"],
):
    ax.plot(ks, [vals_f16[k]  for k in ks], marker="o", lw=2.0,
            color="steelblue",  label="float16")
    ax.plot(ks, [vals_2bit[k] for k in ks], marker="s", lw=2.0,
            color="darkorange", label="2-bit", linestyle="--")
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(ks)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "recall_mrr_ndcg_k10.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
