#!/usr/bin/env python3
"""
plot_score_comparison.py
------------------------
float16 vs 2-bit MaxSim 스코어 비교 시각화.
scores_analog.json, scores_2bit.json, sample_queries.json 을 읽어
2×2 서브플롯 PNG를 생성한다.

실행:
  python plot_score_comparison.py
"""

import os, json, random
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

RESULTS_DIR = os.path.expanduser("~/ColBERT/experiments/msmarco")
OUT_PNG     = os.path.join(RESULTS_DIR, "score_comparison.png")
SEED        = 42
SCATTER_N   = 5000   # scatter에 표시할 최대 점 수

# ── 데이터 로드 ────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "scores_analog.json"))   as f: res_f16  = json.load(f)
with open(os.path.join(RESULTS_DIR, "scores_2bit.json"))     as f: res_2bit = json.load(f)
with open(os.path.join(RESULTS_DIR, "sample_queries.json"))  as f: queries  = json.load(f)

# ── per-pair & per-query 통계 계산 ─────────────────────────
all_s_f16, all_s_b2 = [], []
spearman_rhos, r2_vals, top50_overlaps = [], [], []

for qid in queries:
    f16 = res_f16.get(qid, {})
    b2  = res_2bit.get(qid, {})
    common = sorted(set(f16) & set(b2), key=lambda p: -f16[p])
    if len(common) < 10:
        continue

    s_f16 = np.array([f16[p] for p in common])
    s_b2  = np.array([b2[p]  for p in common])

    all_s_f16.extend(s_f16.tolist())
    all_s_b2.extend(s_b2.tolist())

    rho, _ = stats.spearmanr(s_f16, s_b2)
    spearman_rhos.append(rho)

    r, _ = stats.pearsonr(s_f16, s_b2)
    r2_vals.append(r ** 2)

    top50_f16 = set(sorted(f16, key=f16.get, reverse=True)[:50])
    top50_b2  = set(sorted(b2,  key=b2.get,  reverse=True)[:50])
    top50_overlaps.append(len(top50_f16 & top50_b2) / 50 * 100)

all_s_f16 = np.array(all_s_f16)
all_s_b2  = np.array(all_s_b2)
diffs     = all_s_f16 - all_s_b2

# scatter 샘플링
rng = np.random.default_rng(SEED)
idx = rng.choice(len(all_s_f16), size=min(SCATTER_N, len(all_s_f16)), replace=False)
sc_f16 = all_s_f16[idx]
sc_b2  = all_s_b2[idx]

# ── 스타일 ─────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("float16 vs 2-bit  MaxSim Score Comparison\n"
             f"(100 queries × k=200 candidates, {len(diffs):,} pairs)",
             fontsize=14, fontweight="bold", y=1.01)

# ── [좌상] Scatter ─────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(sc_b2, sc_f16, alpha=0.25, s=6, color="steelblue", rasterized=True)
lo = min(all_s_b2.min(), all_s_f16.min())
hi = max(all_s_b2.max(), all_s_f16.max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x")
ax.set_xlabel("2-bit score")
ax.set_ylabel("float16 score")
ax.set_title(f"Score Scatter  (r² = {np.mean(r2_vals):.3f})")
ax.legend(fontsize=9)

# ── [우상] Score diff 히스토그램 ──────────────────────────
ax = axes[0, 1]
ax.hist(diffs, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(diffs.mean(), color="red",    lw=1.5, linestyle="--",
           label=f"mean = {diffs.mean():+.3f}")
ax.axvline(diffs.mean() + diffs.std(), color="orange", lw=1.2, linestyle=":",
           label=f"±std = {diffs.std():.3f}")
ax.axvline(diffs.mean() - diffs.std(), color="orange", lw=1.2, linestyle=":")
ax.axvline(0, color="black", lw=0.8, linestyle="-", alpha=0.5)
ax.set_xlabel("float16 score − 2-bit score")
ax.set_ylabel("Count")
ax.set_title("Score Difference Distribution")
ax.legend(fontsize=9)

# ── [좌하] Spearman ρ per-query ───────────────────────────
ax = axes[1, 0]
sns.histplot(spearman_rhos, bins=20, kde=True, color="seagreen", ax=ax)
ax.axvline(np.mean(spearman_rhos), color="red", lw=1.5, linestyle="--",
           label=f"mean = {np.mean(spearman_rhos):.3f}")
ax.set_xlabel("Spearman ρ")
ax.set_ylabel("Query count")
ax.set_title("Per-query Spearman ρ  (ranking preservation)")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)

# ── [우하] Top-50 overlap per-query ───────────────────────
ax = axes[1, 1]
sns.histplot(top50_overlaps, bins=20, kde=True, color="darkorange", ax=ax)
ax.axvline(np.mean(top50_overlaps), color="red", lw=1.5, linestyle="--",
           label=f"mean = {np.mean(top50_overlaps):.1f}%")
ax.set_xlabel("Top-50 overlap (%)")
ax.set_ylabel("Query count")
ax.set_title("Per-query Top-50 Result Overlap")
ax.legend(fontsize=9)
ax.set_xlim(0, 105)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
