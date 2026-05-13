#!/usr/bin/env python3
"""
plot_covid_analysis.py
----------------------
TREC-COVID: float16 vs 2-bit 비교 그래프
  Fig 1: Top-50 Result Overlap histogram
  Fig 2: R@k / MRR@k / nDCG@k (k=1..10) 3-panel
  Fig 3: Per-query Score Margin Scatter
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
TABLE5_DIR   = os.path.join(COLBERT_DIR, "experiments/table5")
RANKING_DIR  = os.path.join(TABLE5_DIR, "rankings")
QRELS_FILE   = os.path.join(COLBERT_DIR, "data/table5/beir/trec-covid/qrels.tsv")
ANALYSIS_JSON = os.path.join(TABLE5_DIR, "trec_covid_analysis.json")

F16_RANKING  = os.path.join(RANKING_DIR, "trec-covid.analog.tsv")
BIT2_RANKING = os.path.join(RANKING_DIR, "trec-covid.quantized.tsv")

# ── 데이터 로드 ───────────────────────────────────────────────
with open(ANALYSIS_JSON) as f:
    analysis = json.load(f)

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

qrels     = load_qrels(QRELS_FILE)
rank_f16  = load_ranking(F16_RANKING)
rank_2bit = load_ranking(BIT2_RANKING)

# ── 지표 계산 ─────────────────────────────────────────────────
ks = list(range(1, 11))

def recall_at_k(rankings, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, results in rankings.items():
        rel = qrels.get(qid, set())
        if not rel: continue
        ranked_pids = [pid for _, pid, _ in results]
        for k in ks:
            top_k = set(ranked_pids[:k])
            scores[k].append(len(rel & top_k) / len(rel))
    return {k: np.mean(v) for k, v in scores.items() if v}

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

r_f16    = recall_at_k(rank_f16,  qrels, ks)
r_2bit   = recall_at_k(rank_2bit, qrels, ks)
mrr_f16  = {int(k): v for k, v in analysis["mrr_float16"].items()}
mrr_2bit = {int(k): v for k, v in analysis["mrr_2bit"].items()}
ndcg_f16  = ndcg_at_k(rank_f16,  qrels, ks)
ndcg_2bit = ndcg_at_k(rank_2bit, qrels, ks)

top50_vals = np.array(analysis["top50_overlap"]["values"])

mg_f16  = analysis["score_margin_float16"]
mg_2bit = analysis["score_margin_2bit"]
common_qids = sorted(set(mg_f16) & set(mg_2bit))
f16_mg  = np.array([mg_f16[q]  for q in common_qids])
bit2_mg = np.array([mg_2bit[q] for q in common_qids])

# ── 스타일 ────────────────────────────────────────────────────
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.05)
COLOR_F16  = "steelblue"
COLOR_2BIT = "darkorange"

# ════════════════════════════════════════════════════════════
# Figure 1: Top-50 Overlap Histogram
# ════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(8, 5))

ax.hist(top50_vals * 100, bins=20, color="darkorange", alpha=0.75,
        edgecolor="white", linewidth=0.5)

# KDE 곡선
kde_x = np.linspace(0, 100, 300)
try:
    kde = gaussian_kde(top50_vals * 100, bw_method=0.3)
    kde_y = kde(kde_x)
    kde_y = kde_y / kde_y.max() * ax.get_ylim()[1] * 0.9
    ax.plot(kde_x, kde_y, color="darkorange", lw=2)
except Exception:
    pass

mean_val = top50_vals.mean() * 100
ax.axvline(mean_val, color="crimson", linestyle="--", lw=1.5,
           label=f"mean = {mean_val:.1f}%")

ax.set_xlabel("Top-50 overlap (%)", fontsize=12)
ax.set_ylabel("Query count", fontsize=12)
ax.set_title("[Method 2]  Top-50 Result Overlap per-query\n"
             "TREC-COVID  |  float16 vs 2-bit  |  50 queries",
             fontsize=12, fontweight="bold", color="steelblue")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(fontsize=10)
plt.tight_layout()
out1 = os.path.join(TABLE5_DIR, "covid_top50_overlap.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved -> {out1}")

# ════════════════════════════════════════════════════════════
# Figure 2: R@k / MRR@k / nDCG@k
# ════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("float16 vs 2-bit  (k=1..10)  |  50 queries, TREC-COVID",
              fontsize=12, fontweight="bold")

panels = [
    ("R@k",    r_f16,    r_2bit,   "Recall@k (%)"),
    ("MRR@k",  mrr_f16,  mrr_2bit, "MRR@k (%)"),
    ("nDCG@k", ndcg_f16, ndcg_2bit,"nDCG@k (%)"),
]

for ax, (title, vals_f16, vals_2bit, ylabel) in zip(axes, panels):
    y_f16  = [vals_f16[k]  * 100 for k in ks]
    y_2bit = [vals_2bit[k] * 100 for k in ks]
    ax.plot(ks, y_f16,  "o-", color=COLOR_F16,  lw=2, ms=5, label="float16")
    ax.plot(ks, y_2bit, "s--",color=COLOR_2BIT, lw=2, ms=5, label="2-bit")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(ks)
    ax.legend(fontsize=9)
    # 값 범위 여백
    all_vals = y_f16 + y_2bit
    margin = (max(all_vals) - min(all_vals)) * 0.5 + 0.05
    ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

plt.tight_layout()
out2 = os.path.join(TABLE5_DIR, "covid_recall_mrr_ndcg.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved -> {out2}")

# ════════════════════════════════════════════════════════════
# Figure 3: Per-query Score Margin Scatter
# ════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(7, 6))

f16_better  = f16_mg > bit2_mg
f16_worse   = ~f16_better
pct_better  = f16_better.sum() / len(f16_mg)

ax.scatter(f16_mg[f16_better],  bit2_mg[f16_better],
           c=COLOR_F16,  alpha=0.7, s=40, label=f"float16 larger ({pct_better:.1%})")
ax.scatter(f16_mg[f16_worse], bit2_mg[f16_worse],
           c=COLOR_2BIT, alpha=0.7, s=40, label=f"2-bit larger ({1-pct_better:.1%})")

lo = min(f16_mg.min(), bit2_mg.min()) - 0.5
hi = max(f16_mg.max(), bit2_mg.max()) + 0.5
ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, alpha=0.7, label="y = x  (equal margin)")
ax.axhline(0, color="gray", lw=0.7, linestyle=":")
ax.axvline(0, color="gray", lw=0.7, linestyle=":")

n_fail_f16  = (f16_mg  < 0).sum()
n_fail_2bit = (bit2_mg < 0).sum()

ax.text(0.03, 0.97,
        f"float16 failures: {n_fail_f16}/50 ({n_fail_f16/50:.1%})\n"
        f"2-bit   failures: {n_fail_2bit}/50 ({n_fail_2bit/50:.1%})",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.set_xlabel("Score Margin (float16)", fontsize=11)
ax.set_ylabel("Score Margin (2-bit)",   fontsize=11)
ax.set_title("Per-query Score Margin Scatter\n"
             "(blue=float16 larger, orange=2-bit larger)\n"
             "TREC-COVID  |  50 queries",
             fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
out3 = os.path.join(TABLE5_DIR, "covid_score_margin_scatter.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved -> {out3}")

# ── 수치 요약 ─────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  TREC-COVID Summary")
print(f"{'='*55}")
print(f"  Top-50 Overlap:  mean={top50_vals.mean():.4f}")
print(f"  MRR@10:  float16={mrr_f16[10]:.4f}  2-bit={mrr_2bit[10]:.4f}  Δ={mrr_f16[10]-mrr_2bit[10]:+.4f}")
print(f"  R@10:    float16={r_f16[10]:.4f}   2-bit={r_2bit[10]:.4f}   Δ={r_f16[10]-r_2bit[10]:+.4f}")
print(f"  nDCG@10: float16={ndcg_f16[10]:.4f}  2-bit={ndcg_2bit[10]:.4f}  Δ={ndcg_f16[10]-ndcg_2bit[10]:+.4f}")
print(f"  Score Margin: float16={f16_mg.mean():+.4f}  2-bit={bit2_mg.mean():+.4f}")
print(f"  float16 > 2-bit margin: {f16_better.sum()}/{len(f16_mg)} ({pct_better:.1%})")
print(f"{'='*55}")
