#!/usr/bin/env python3
"""
plot_covid_ndcg50.py
--------------------
TREC-COVID: nDCG@k (k=1..50) float16 vs 2-bit
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
TABLE5_DIR   = os.path.join(COLBERT_DIR, "experiments/table5")
RANKING_DIR  = os.path.join(TABLE5_DIR, "rankings")
QRELS_FILE   = os.path.join(COLBERT_DIR, "data/table5/beir/trec-covid/qrels.tsv")
F16_RANKING  = os.path.join(RANKING_DIR, "trec-covid.analog.tsv")
BIT2_RANKING = os.path.join(RANKING_DIR, "trec-covid.quantized.tsv")

def load_qrels(path, min_rel=1):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            qid, pid, label = parts[0], parts[2], int(parts[3])
            if label >= min_rel:
                qrels.setdefault(qid, set()).add(pid)
    return qrels

def load_ranking(path):
    rankings = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            qid, pid, rank, score = parts[0], parts[1], int(parts[2]), float(parts[3])
            rankings.setdefault(qid, []).append((rank, pid, score))
    for qid in rankings:
        rankings[qid].sort(key=lambda x: x[0])
    return rankings

def ndcg_at_k(rankings, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, results in rankings.items():
        rel = qrels.get(qid, set())
        if not rel: continue
        ranked_pids = [pid for _, pid, _ in results]
        for k in ks:
            dcg  = sum(1.0/np.log2(r+1) for r,p in enumerate(ranked_pids[:k],1) if p in rel)
            idcg = sum(1.0/np.log2(r+1) for r in range(1, min(k,len(rel))+1))
            scores[k].append(dcg/idcg if idcg > 0 else 0.0)
    return {k: np.mean(v) for k, v in scores.items() if v}

qrels     = load_qrels(QRELS_FILE)
rank_f16  = load_ranking(F16_RANKING)
rank_2bit = load_ranking(BIT2_RANKING)

ks = list(range(1, 31))
ndcg_f16  = ndcg_at_k(rank_f16,  qrels, ks)
ndcg_2bit = ndcg_at_k(rank_2bit, qrels, ks)

y_f16  = np.array([ndcg_f16[k]  for k in ks]) * 100
y_2bit = np.array([ndcg_2bit[k] for k in ks]) * 100
delta  = y_f16 - y_2bit

# ── 출력 ─────────────────────────────────────────────────────
print(f"{'k':>4}  {'float16':>10}  {'2-bit':>10}  {'delta':>10}")
print("-" * 44)
for k in ks:
    print(f"{k:>4}  {y_f16[k-1]:>10.4f}  {y_2bit[k-1]:>10.4f}  {delta[k-1]:>+10.4f}")

# ── 그래프 ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                          gridspec_kw={"height_ratios": [3, 1.5]})
fig.suptitle("nDCG@k  (k=1..30)  |  TREC-COVID  |  float16 vs 2-bit",
             fontsize=13, fontweight="bold")

# ── 상단: nDCG@k 곡선 ─────────────────────────────────────────
ax = axes[0]
ax.plot(ks, y_f16,  "o-",  color="steelblue",   lw=2,   ms=4, label="float16")
ax.plot(ks, y_2bit, "s--", color="darkorange",  lw=2,   ms=4, label="2-bit")
ax.fill_between(ks, y_2bit, y_f16, alpha=0.15, color="steelblue", label="gap (float16 − 2-bit)")

for k_mark in [1, 5, 10, 20, 30]:
    ax.annotate(f"k={k_mark}\nΔ={delta[k_mark-1]:+.2f}",
                xy=(k_mark, y_f16[k_mark-1]),
                xytext=(k_mark + 0.8, y_f16[k_mark-1] + 0.3),
                fontsize=8, color="steelblue",
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8))

ax.set_ylabel("nDCG@k (%)", fontsize=12)
ax.set_xticks(range(0, 31, 5))
ax.legend(fontsize=10)
ax.set_xlim(0.5, 30.5)

# ── 하단: delta 곡선 ──────────────────────────────────────────
ax2 = axes[1]
ax2.plot(ks, delta, color="crimson", lw=2, label="Δ = float16 − 2-bit")
ax2.fill_between(ks, 0, delta,
                 where=(delta >= 0), alpha=0.2, color="steelblue",  label="float16 better")
ax2.fill_between(ks, 0, delta,
                 where=(delta <  0), alpha=0.2, color="darkorange", label="2-bit better")
ax2.axhline(0, color="black", lw=0.8, linestyle="--")

ax2.set_xlabel("k", fontsize=12)
ax2.set_ylabel("Δ nDCG@k (%)", fontsize=12)
ax2.set_xticks(range(0, 31, 5))
ax2.legend(fontsize=9)
ax2.set_xlim(0.5, 30.5)

plt.tight_layout()
out = os.path.join(TABLE5_DIR, "covid_ndcg_k30.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
