#!/usr/bin/env python3
"""
plot_combined.py
----------------
방법 1 (Reconstruction Error) + 방법 2 (MaxSim Score 비교) 합친 시각화.

3×2 레이아웃:
  행 1 — 방법 1: L2 error 히스토그램 / Angular error 히스토그램
  행 2 — 방법 2: Score scatter / Score diff 히스토그램
  행 3 — 방법 2: Spearman ρ per-query / Top-50 overlap per-query

실행:
  python plot_combined.py
"""

import os, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import torch

RESULTS_DIR = os.path.expanduser("~/ColBERT/experiments/msmarco")
INDEX_DIR   = os.path.expanduser("~/ColBERT/experiments/msmarco/indexes")
OUT_PNG     = os.path.join(RESULTS_DIR, "comparison_combined.png")
SEED        = 42

# ── 데이터 로드 ────────────────────────────────────────────
# 방법 1
l2_sample  = np.load(os.path.join(RESULTS_DIR, "recon_l2_sample.npy"))
with open(os.path.join(RESULTS_DIR, "recon_error.json")) as f:
    re_stats = json.load(f)

# 방법 1 추가: residual 값 분포 (float16 연속 vs 2-bit 4개 버킷)
res_f16_chunk = torch.load(f"{INDEX_DIR}/200k.analog/0.residuals.pt",
                           map_location="cpu").float().flatten().numpy()
rng_res = np.random.default_rng(SEED)
res_f16_sample = rng_res.choice(res_f16_chunk, size=min(100_000, len(res_f16_chunk)), replace=False)
_, bucket_weights = torch.load(f"{INDEX_DIR}/200k.2bit/buckets.pt", map_location="cpu")
bucket_weights = bucket_weights.numpy()

# 방법 2
with open(os.path.join(RESULTS_DIR, "scores_analog.json"))  as f: res_f16  = json.load(f)
with open(os.path.join(RESULTS_DIR, "scores_2bit.json"))    as f: res_2bit = json.load(f)
with open(os.path.join(RESULTS_DIR, "sample_queries.json")) as f: queries  = json.load(f)

all_sf16, all_sb2 = [], []
spearman_rhos, r2_vals, top50_overlaps = [], [], []

for qid in queries:
    f16 = res_f16.get(qid, {})
    b2  = res_2bit.get(qid, {})
    common = sorted(set(f16) & set(b2), key=lambda p: -f16[p])
    if len(common) < 10:
        continue
    sf16 = np.array([f16[p] for p in common])
    sb2  = np.array([b2[p]  for p in common])
    all_sf16.extend(sf16.tolist())
    all_sb2.extend(sb2.tolist())
    rho, _ = stats.spearmanr(sf16, sb2)
    spearman_rhos.append(rho)
    r, _ = stats.pearsonr(sf16, sb2)
    r2_vals.append(r ** 2)
    top50_f16 = set(sorted(f16, key=f16.get, reverse=True)[:50])
    top50_b2  = set(sorted(b2,  key=b2.get,  reverse=True)[:50])
    top50_overlaps.append(len(top50_f16 & top50_b2) / 50 * 100)

all_sf16  = np.array(all_sf16)
all_sb2   = np.array(all_sb2)
diffs     = all_sf16 - all_sb2

rng = np.random.default_rng(SEED)
idx = rng.choice(len(all_sf16), size=min(5000, len(all_sf16)), replace=False)

# ── 레이아웃 ───────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(3, 2, figsize=(13, 14))
fig.suptitle("float16 vs 2-bit  ColBERT Residual Comparison\n"
             "Method 1: Vector Reconstruction Error  |  Method 2: MaxSim Score Difference",
             fontsize=13, fontweight="bold", y=1.01)

# ── 방법 1 라벨 ────────────────────────────────────────────
for ax, title in zip(axes[0], ["[Method 1]  L2 Reconstruction Error",
                               "[Method 1]  Residual Value Distribution"]):
    ax.set_title(title, fontweight="bold", color="#1a4f8a")

# ── [0,0] L2 Error 히스토그램 ─────────────────────────────
ax = axes[0, 0]
ax.hist(l2_sample, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
ax.axvline(re_stats["l2_mean"],  color="red",    lw=1.6, linestyle="--",
           label=f"mean = {re_stats['l2_mean']:.3f}")
ax.axvline(re_stats["l2_p95"],   color="orange", lw=1.3, linestyle=":",
           label=f"p95 = {re_stats['l2_p95']:.3f}")
ax.set_xlabel("L2 distance  (0 = identical, 2.0 = opposite)")
ax.set_ylabel("Count")
ax.legend(fontsize=9)
ax.text(0.97, 0.95, f"n = {re_stats['n_embeddings']:,} embeddings",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray")

# ── [0,1] Residual 값 분포: float16(연속) vs 2-bit(4개 버킷) ─
ax = axes[0, 1]
ax.hist(res_f16_sample, bins=120, color="steelblue", edgecolor="none",
        alpha=0.75, density=True, label="float16 (continuous)")
for i, bw in enumerate(bucket_weights):
    ax.axvline(bw, color="darkorange", lw=2.0,
               linestyle="--", label=f"2-bit bucket {i+1}: {bw:.4f}" if i == 0 else f"  bucket {i+1}: {bw:.4f}")
ax.set_xlabel("Residual value (per dimension)")
ax.set_ylabel("Density")
ax.legend(fontsize=8.5)
ax.text(0.97, 0.97, "2-bit: only 4 values\nfloat16: continuous",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        color="gray", linespacing=1.5)

# 방법 2 타이틀 색
for ax, title in zip(axes[1],
        ["[Method 2]  Score Scatter", "[Method 2]  Score Distribution: float16 vs 2-bit"]):
    ax.set_title(title, fontweight="bold", color="#1a4f8a")

# ── [1,0] Score Scatter ───────────────────────────────────
ax = axes[1, 0]
ax.scatter(all_sb2[idx], all_sf16[idx],
           alpha=0.25, s=6, color="seagreen", rasterized=True)
lo = min(all_sb2.min(), all_sf16.min())
hi = max(all_sb2.max(), all_sf16.max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x")
ax.set_xlabel("2-bit score")
ax.set_ylabel("float16 score")
ax.legend(fontsize=9)
ax.text(0.05, 0.93, f"r² = {np.mean(r2_vals):.3f}",
        transform=ax.transAxes, fontsize=10, color="darkred")

# ── [1,1] Score Distribution Overlay ─────────────────────
ax = axes[1, 1]
ax.hist(all_sf16, bins=80, color="steelblue",   edgecolor="none",
        alpha=0.55, density=True, label=f"float16  (mean={all_sf16.mean():.2f})")
ax.hist(all_sb2,  bins=80, color="darkorange", edgecolor="none",
        alpha=0.55, density=True, label=f"2-bit    (mean={all_sb2.mean():.2f})")
ax.axvline(all_sf16.mean(), color="steelblue",  lw=1.6, linestyle="--")
ax.axvline(all_sb2.mean(),  color="darkorange", lw=1.6, linestyle="--")
ax.set_xlabel("MaxSim score")
ax.set_ylabel("Density")
ax.legend(fontsize=9)

for ax, title in zip(axes[2],
        ["[Method 2]  Spearman ρ per-query", "[Method 2]  Top-50 Result Overlap per-query"]):
    ax.set_title(title, fontweight="bold", color="#1a4f8a")

# ── [2,0] Spearman ρ ─────────────────────────────────────
ax = axes[2, 0]
sns.histplot(spearman_rhos, bins=20, kde=True, color="darkorange", ax=ax)
ax.axvline(np.mean(spearman_rhos), color="red", lw=1.6, linestyle="--",
           label=f"mean = {np.mean(spearman_rhos):.3f}")
ax.set_xlabel("Spearman ρ")
ax.set_ylabel("Query count")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)

# ── [2,1] Top-50 Overlap ─────────────────────────────────
ax = axes[2, 1]
sns.histplot(top50_overlaps, bins=20, kde=True, color="darkorange", ax=ax)
ax.axvline(np.mean(top50_overlaps), color="red", lw=1.6, linestyle="--",
           label=f"mean = {np.mean(top50_overlaps):.1f}%")
ax.set_xlabel("Top-50 overlap (%)")
ax.set_ylabel("Query count")
ax.legend(fontsize=9)
ax.set_xlim(0, 105)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
