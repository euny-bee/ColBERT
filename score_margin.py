#!/usr/bin/env python3
"""
score_margin.py
---------------
Score Margin comparison: float16 vs 2-bit.
Uses existing 500q search results (no re-search needed).

Plots (2x2):
  [0,0] Violin+box: margin distribution side-by-side
  [0,1] CDF: cumulative distribution of margins
  [1,0] Per-query difference histogram (margin_f16 - margin_2bit)
  [1,1] Scatter with quadrant coloring
"""

import os, json
import numpy as np
from scipy import stats
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

# ── 계산 ─────────────────────────────────────────────────────
margins_f16, margins_2bit = [], []
ndcg10_f16,  ndcg10_2bit  = [], []
hard_rank_f16, hard_rank_2bit = [], []

for qid in queries:
    f16 = res_f16.get(qid, {})
    b2  = res_2bit.get(qid, {})
    rel = qrels.get(qid, set())
    if not rel or not f16 or not b2:
        continue

    ranked_f16 = sorted(f16.keys(), key=lambda p: -f16[p])
    ranked_b2  = sorted(b2.keys(),  key=lambda p: -b2[p])

    # ── Score Margin ─────────────────────────────────────────
    # 첫 번째 관련 문서의 score - 1등 비관련 문서의 score
    def get_margin(ranked, scores, rel):
        rel_score     = next((scores[p] for p in ranked if p in rel),     None)
        nonrel_top    = next((scores[p] for p in ranked if p not in rel), None)
        if rel_score is None or nonrel_top is None:
            return None
        return rel_score - nonrel_top

    m_f16 = get_margin(ranked_f16, f16, rel)
    m_b2  = get_margin(ranked_b2,  b2,  rel)
    if m_f16 is not None and m_b2 is not None:
        margins_f16.append(m_f16)
        margins_2bit.append(m_b2)

    # ── nDCG@10 per-query ────────────────────────────────────
    def dcg(ranked, scores, rel, k=10):
        gain = sum(1.0 / np.log2(r + 1)
                   for r, p in enumerate(ranked[:k], 1) if p in rel)
        ideal = sum(1.0 / np.log2(r + 1)
                    for r in range(1, min(k, len(rel)) + 1))
        return gain / ideal if ideal > 0 else 0.0

    ndcg10_f16.append(dcg(ranked_f16, f16, rel))
    ndcg10_2bit.append(dcg(ranked_b2,  b2,  rel))

    # ── Hard Query: 관련 문서가 float16에서 rank 2~10 ────────
    rank_map_f16 = {p: i+1 for i, p in enumerate(ranked_f16)}
    rank_map_b2  = {p: i+1 for i, p in enumerate(ranked_b2)}
    for pid in rel:
        r_f16 = rank_map_f16.get(pid)
        r_b2  = rank_map_b2.get(pid)
        if r_f16 is not None and 2 <= r_f16 <= 10 and r_b2 is not None:
            hard_rank_f16.append(r_f16)
            hard_rank_2bit.append(r_b2)

margins_f16  = np.array(margins_f16)
margins_2bit = np.array(margins_2bit)
ndcg10_f16   = np.array(ndcg10_f16)
ndcg10_2bit  = np.array(ndcg10_2bit)
hard_rank_f16  = np.array(hard_rank_f16)
hard_rank_2bit = np.array(hard_rank_2bit)

# ── 통계 ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Score Margin  (relevant - top_nonrelevant)")
print(f"{'='*60}")
print(f"  float16  mean={margins_f16.mean():+.4f}  std={margins_f16.std():.4f}")
print(f"  2-bit    mean={margins_2bit.mean():+.4f}  std={margins_2bit.std():.4f}")
print(f"  float16 > 2-bit: {(margins_f16 > margins_2bit).mean():.1%} of queries")

print(f"\n{'='*60}")
print(f"  Statistical Significance  (nDCG@10 per-query, n={len(ndcg10_f16)})")
print(f"{'='*60}")
t_stat, t_pval = stats.ttest_rel(ndcg10_f16, ndcg10_2bit)
w_stat, w_pval = stats.wilcoxon(ndcg10_f16, ndcg10_2bit)
print(f"  Paired t-test:            t={t_stat:+.4f}  p={t_pval:.4f}")
print(f"  Wilcoxon signed-rank:     W={w_stat:.0f}    p={w_pval:.4f}")
print(f"  float16 mean nDCG@10 = {ndcg10_f16.mean():.4f}")
print(f"  2-bit   mean nDCG@10 = {ndcg10_2bit.mean():.4f}")
sig = "YES (p<0.05)" if t_pval < 0.05 else "NO (p>=0.05)"
print(f"  Statistically significant: {sig}")

print(f"\n{'='*60}")
print(f"  Hard Queries (relevant doc at rank 2~10 in float16, n={len(hard_rank_f16)})")
print(f"{'='*60}")
if len(hard_rank_f16) > 0:
    print(f"  float16 rank  mean={hard_rank_f16.mean():.2f}")
    print(f"  2-bit   rank  mean={hard_rank_2bit.mean():.2f}")
    print(f"  2-bit worse (rank↑): {(hard_rank_2bit > hard_rank_f16).mean():.1%}")
    print(f"  2-bit same          : {(hard_rank_2bit == hard_rank_f16).mean():.1%}")
    print(f"  2-bit better (rank↓): {(hard_rank_2bit < hard_rank_f16).mean():.1%}")

# ── 그래프 ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "Score Margin Comparison: float16 vs 2-bit\n"
    "margin = score(relevant doc) − score(top non-relevant doc)  |  500 queries",
    fontsize=12, fontweight="bold"
)

# [0,0] Violin + box side-by-side
ax = axes[0, 0]
data = [margins_f16, margins_2bit]
labels = [f"float16\nmean={margins_f16.mean():+.2f}", f"2-bit\nmean={margins_2bit.mean():+.2f}"]
parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=False)
parts["bodies"][0].set_facecolor("steelblue");  parts["bodies"][0].set_alpha(0.6)
parts["bodies"][1].set_facecolor("darkorange"); parts["bodies"][1].set_alpha(0.6)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
ax.boxplot(data, positions=[1, 2], widths=0.1,
           patch_artist=False, medianprops=dict(color="black", lw=2),
           whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2), flierprops=dict(ms=3))
ax.axhline(0, color="gray", lw=1.0, linestyle="--", alpha=0.6, label="margin=0")
ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Score Margin")
ax.set_title("Margin Distribution (Violin + Box)", fontweight="bold")
ax.legend(fontsize=9)

# [0,1] CDF 비교
ax = axes[0, 1]
for arr, color, label in [
    (margins_f16,  "steelblue",  f"float16  mean={margins_f16.mean():+.2f}"),
    (margins_2bit, "darkorange", f"2-bit    mean={margins_2bit.mean():+.2f}"),
]:
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    ax.plot(sorted_arr, cdf, lw=2.0, color=color, label=label)
ax.axvline(0, color="gray", lw=1.0, linestyle="--", alpha=0.6)
ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.5)
ax.set_xlabel("Score Margin")
ax.set_ylabel("Cumulative Probability")
ax.set_title("CDF of Score Margin\n(curve shifted right = better)", fontweight="bold")
ax.legend(fontsize=9)

# [1,0] Per-query difference histogram (margin_f16 - margin_2bit)
diff = margins_f16 - margins_2bit
ax = axes[1, 0]
ax.hist(diff, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
ax.axvline(0,           color="black", lw=1.0, linestyle="-",  alpha=0.6, label="no difference")
ax.axvline(diff.mean(), color="red",   lw=1.8, linestyle="--",
           label=f"mean = {diff.mean():+.3f}")
pct_pos = (diff > 0).mean()
ax.set_xlabel("margin_f16 − margin_2bit  (per query)")
ax.set_ylabel("Query count")
ax.set_title(f"Per-query Margin Difference\n"
             f"float16 larger: {pct_pos:.1%}  |  2-bit larger: {1-pct_pos:.1%}",
             fontweight="bold")
ax.legend(fontsize=9)
# 좌/우 영역 색칠
ylim = ax.get_ylim()
ax.axvspan(ax.get_xlim()[0], 0, alpha=0.07, color="darkorange", label="2-bit better")
ax.axvspan(0, ax.get_xlim()[1], alpha=0.07, color="steelblue",  label="float16 better")

# [1,1] Scatter with quadrant coloring
ax = axes[1, 1]
color_arr = np.where(margins_f16 > margins_2bit, "steelblue", "darkorange")
ax.scatter(margins_f16, margins_2bit, c=color_arr, alpha=0.4, s=10, rasterized=True)
lo = min(margins_f16.min(), margins_2bit.min())
hi = max(margins_f16.max(), margins_2bit.max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x  (equal margin)")
ax.axhline(0, color="gray", lw=0.7, linestyle=":")
ax.axvline(0, color="gray", lw=0.7, linestyle=":")
# 사분면 레이블
ax.text(0.97, 0.03, f"float16 better\n{(margins_f16 > margins_2bit).mean():.1%}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="steelblue", fontweight="bold")
ax.text(0.03, 0.97, f"2-bit better\n{(margins_f16 < margins_2bit).mean():.1%}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        color="darkorange", fontweight="bold")
ax.set_xlabel("Score Margin (float16)")
ax.set_ylabel("Score Margin (2-bit)")
ax.set_title("Per-query Scatter\n(blue=float16 larger, orange=2-bit larger)", fontweight="bold")
ax.legend(fontsize=9)

# p-value 하단 주석
pval_str = (f"Statistical test on nDCG@10 per-query (n={len(ndcg10_f16)}):  "
            f"Paired t-test p={t_pval:.4f}  |  Wilcoxon p={w_pval:.4f}  "
            f"→ {'significant ✓' if t_pval < 0.05 else 'not significant (difference is real but small)'}")
fig.text(0.5, -0.02, pval_str, ha="center", fontsize=9, color="darkred",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "score_margin.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
