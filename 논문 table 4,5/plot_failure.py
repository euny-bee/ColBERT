#!/usr/bin/env python3
"""
plot_failure.py
---------------
Score Margin < 0 (검색 실패) 비율 비교: float16 vs 2-bit
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def get_margin(ranked, scores, rel):
    rel_score  = next((scores[p] for p in ranked if p in rel),     None)
    nonrel_top = next((scores[p] for p in ranked if p not in rel), None)
    if rel_score is None or nonrel_top is None:
        return None
    return rel_score - nonrel_top

margins_f16, margins_2bit = [], []
for qid in queries:
    f16 = res_f16.get(qid, {})
    b2  = res_2bit.get(qid, {})
    rel = qrels.get(qid, set())
    if not rel or not f16 or not b2:
        continue
    ranked_f16 = sorted(f16.keys(), key=lambda p: -f16[p])
    ranked_b2  = sorted(b2.keys(),  key=lambda p: -b2[p])
    m_f16 = get_margin(ranked_f16, f16, rel)
    m_b2  = get_margin(ranked_b2,  b2,  rel)
    if m_f16 is not None and m_b2 is not None:
        margins_f16.append(m_f16)
        margins_2bit.append(m_b2)

margins_f16  = np.array(margins_f16)
margins_2bit = np.array(margins_2bit)
n = len(margins_f16)

# ── 분류 ──────────────────────────────────────────────────────
fail_f16 = margins_f16 < 0
fail_2bit = margins_2bit < 0

both_fail    = fail_f16 & fail_2bit          # 둘 다 실패
only_f16     = fail_f16 & ~fail_2bit         # float16만 실패
only_2bit    = ~fail_f16 & fail_2bit         # 2-bit만 실패
both_succeed = ~fail_f16 & ~fail_2bit        # 둘 다 성공

counts = {
    "Both succeed": both_succeed.sum(),
    "Only 2-bit fails": only_2bit.sum(),
    "Only float16 fails": only_f16.sum(),
    "Both fail": both_fail.sum(),
}

print(f"{'='*55}")
print(f"  Search Failure Analysis  (margin < 0 = 관련문서가 밀림)")
print(f"{'='*55}")
print(f"  Total queries: {n}")
print(f"  float16 failures : {fail_f16.sum():>4} ({fail_f16.mean():.1%})")
print(f"  2-bit   failures : {fail_2bit.sum():>4} ({fail_2bit.mean():.1%})")
print()
print(f"  Both succeed      : {both_succeed.sum():>4} ({both_succeed.mean():.1%})")
print(f"  Only 2-bit fails  : {only_2bit.sum():>4} ({only_2bit.mean():.1%})")
print(f"  Only float16 fails: {only_f16.sum():>4} ({only_f16.mean():.1%})")
print(f"  Both fail         : {both_fail.sum():>4} ({both_fail.mean():.1%})")
print(f"{'='*55}")

# ── 그래프 ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Search Failure Comparison: margin < 0  (relevant doc ranked below top non-relevant)\n"
    f"500 queries, 200k MS MARCO subset",
    fontsize=12, fontweight="bold"
)

# ── [0] Scatter: 4가지 분류를 색상으로 ────────────────────────
ax = axes[0]
color_map = {
    "both_succeed": ("steelblue",  "Both succeed"),
    "only_2bit":    ("darkorange", "Only 2-bit fails"),
    "only_f16":     ("gold",       "Only float16 fails"),
    "both_fail":    ("crimson",    "Both fail"),
}
for mask, (color, label) in zip(
    [both_succeed, only_2bit, only_f16, both_fail],
    color_map.values()
):
    ax.scatter(margins_f16[mask], margins_2bit[mask],
               c=color, alpha=0.65, s=18, label=label, rasterized=True)

lo = min(margins_f16.min(), margins_2bit.min()) - 0.5
hi = max(margins_f16.max(), margins_2bit.max()) + 0.5
ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5, label="y = x")
ax.axhline(0, color="gray", lw=0.8, linestyle=":")
ax.axvline(0, color="gray", lw=0.8, linestyle=":")
ax.axhspan(lo, 0, alpha=0.04, color="crimson")
ax.axvspan(lo, 0, alpha=0.04, color="crimson")
ax.set_xlabel("Score Margin (float16)", fontsize=11)
ax.set_ylabel("Score Margin (2-bit)", fontsize=11)
ax.set_title("Per-query Scatter\n(colored by failure type)", fontweight="bold")
ax.legend(fontsize=8, loc="upper left")

# ── [1] Stacked bar: 실패 분류 ──────────────────────────────
ax = axes[1]
labels  = ["float16", "2-bit"]
succeed = [both_succeed.sum() + only_2bit.sum(),   both_succeed.sum() + only_f16.sum()]
fail    = [only_f16.sum()     + both_fail.sum(),   only_2bit.sum()    + both_fail.sum()]

bars_s = ax.bar(labels, succeed, color="steelblue", alpha=0.8, label="Success (margin ≥ 0)")
bars_f = ax.bar(labels, fail, bottom=succeed, color="crimson", alpha=0.75, label="Failure (margin < 0)")

for bar, cnt in zip(bars_s, succeed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f"{cnt}\n({cnt/n:.1%})", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
for bar, bot, cnt in zip(bars_f, succeed, fail):
    ax.text(bar.get_x() + bar.get_width()/2, bot + cnt/2,
            f"{cnt}\n({cnt/n:.1%})", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")

ax.set_ylabel("Query count", fontsize=11)
ax.set_ylim(0, n * 1.08)
ax.set_title("Success vs Failure Count\n(per method)", fontweight="bold")
ax.legend(fontsize=9)

# ── [2] Donut: 4가지 조합 비율 ─────────────────────────────
ax = axes[2]
wedge_labels = [
    f"Both succeed\n{both_succeed.sum()} ({both_succeed.mean():.1%})",
    f"Only 2-bit fails\n{only_2bit.sum()} ({only_2bit.mean():.1%})",
    f"Only float16 fails\n{only_f16.sum()} ({only_f16.mean():.1%})",
    f"Both fail\n{both_fail.sum()} ({both_fail.mean():.1%})",
]
wedge_sizes  = [both_succeed.sum(), only_2bit.sum(), only_f16.sum(), both_fail.sum()]
wedge_colors = ["steelblue", "darkorange", "gold", "crimson"]
wedge_explode = (0, 0.05, 0.05, 0.08)

wedges, texts, autotexts = ax.pie(
    wedge_sizes, labels=wedge_labels, colors=wedge_colors,
    explode=wedge_explode, autopct="%1.1f%%",
    startangle=140, pctdistance=0.75,
    wedgeprops=dict(width=0.55)  # donut
)
for t in autotexts:
    t.set_fontsize(9)
for t in texts:
    t.set_fontsize(8)
ax.set_title("Query Breakdown by Failure Pattern", fontweight="bold")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "failure_analysis.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
