#!/usr/bin/env python3
"""
plot_residual_reconstruction.py
--------------------------------
2-bit 잔차의 '복원' 품질 시각화.

float16 잔차 (연속값) vs 2-bit 복원값 (4개 버킷 중 하나)

Figure 3-panel:
  [0] Scatter: x=float16 residual, y=2-bit residual  → 4개 수평선 (버킷)
  [1] 1차원 슬라이스 (dim=0): 두 분포 비교
  [2] 버킷 경계 + float16 분포 오버레이 (전체 dim 합산)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

COLBERT_DIR = os.path.expanduser("~/ColBERT")
F16_CSV     = os.path.join(COLBERT_DIR, "colbert_float16_correct.csv")
BIT2_CSV    = os.path.join(COLBERT_DIR, "colbert_2bit_correct.csv")
OUT_DIR     = os.path.join(COLBERT_DIR, "experiments/msmarco")
DIM         = 128

RESIDUAL_COLS = [f"residual_dim_{i}" for i in range(DIM)]

print("Loading CSVs...")
df_f16  = pd.read_csv(F16_CSV)
df_2bit = pd.read_csv(BIT2_CSV)

res_f16  = df_f16[RESIDUAL_COLS].values.astype(np.float32)   # (N, 128)
res_2bit = df_2bit[RESIDUAL_COLS].values.astype(np.float32)  # (N, 128)
N = len(res_f16)
print(f"  {N:,} tokens, {DIM} dims each")

# 버킷 4개 값 추출 (전체에서 unique 값)
bucket_values = np.unique(np.round(res_2bit.flatten(), 4))
# 4개에 가깝게 clustering — np.unique는 부동소수점 오차로 더 많이 나올 수 있으므로
from scipy.cluster.vq import kmeans
bucket_vals_clean, _ = kmeans(res_2bit.flatten().reshape(-1,1).astype(np.float64), 4)
bucket_vals_clean = np.sort(bucket_vals_clean.flatten())
print(f"  2-bit bucket values: {bucket_vals_clean}")

# ── 샘플링 (scatter용) ────────────────────────────────────────
MAX_SCATTER = 80_000
flat_f16  = res_f16.flatten()
flat_2bit = res_2bit.flatten()
if len(flat_f16) > MAX_SCATTER:
    idx = np.random.default_rng(42).choice(len(flat_f16), MAX_SCATTER, replace=False)
    sf16  = flat_f16[idx]
    s2bit = flat_2bit[idx]
else:
    sf16, s2bit = flat_f16, flat_2bit

# ── 그래프 ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "2-bit Residual Reconstruction: float16 (continuous) vs 2-bit (4 buckets)\n"
    f"{N:,} tokens from 500 passages, 128 dims each",
    fontsize=12, fontweight="bold"
)

# ── [0] Scatter: float16 vs 2-bit ────────────────────────────
ax = axes[0]
ax.scatter(sf16, s2bit, alpha=0.03, s=3, color="steelblue", rasterized=True)

# 버킷 수평선
colors_b = ["#e74c3c", "#e67e22", "#27ae60", "#8e44ad"]
for bv, bc in zip(bucket_vals_clean, colors_b):
    ax.axhline(bv, color=bc, lw=1.8, linestyle="--", alpha=0.9,
               label=f"bucket = {bv:+.4f}")

ax.set_xlabel("float16 residual value  (continuous)", fontsize=11)
ax.set_ylabel("2-bit reconstructed value  (discrete)", fontsize=11)
ax.set_title("Scatter: float16 vs 2-bit\n(each point = one (token, dim) pair)", fontweight="bold")
ax.legend(fontsize=9, title="2-bit buckets")

# 대각선 (완벽한 복원이라면 y=x)
lo, hi = flat_f16.min(), flat_f16.max()
ax.plot([lo, hi], [lo, hi], "k:", lw=1.0, alpha=0.4, label="y=x (perfect)")

# ── [1] 1D 히스토그램 (dim=0) ─────────────────────────────────
ax = axes[1]
dim_idx = 0
vals_f16_d  = res_f16[:, dim_idx]
vals_2bit_d = res_2bit[:, dim_idx]

ax.hist(vals_f16_d,  bins=60, density=True, alpha=0.6,
        color="steelblue",  label=f"float16 (dim {dim_idx})", edgecolor="none")
ax.hist(vals_2bit_d, bins=60, density=True, alpha=0.6,
        color="darkorange", label=f"2-bit   (dim {dim_idx})", edgecolor="none")

for bv, bc in zip(bucket_vals_clean, colors_b):
    ax.axvline(bv, color=bc, lw=2.0, linestyle="--", alpha=0.9)

ax.set_xlabel("Residual value", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Residual Distribution (dim={dim_idx})\nfloat16=continuous, 2-bit=4 spikes", fontweight="bold")
ax.legend(fontsize=9)

# ── [2] 전체 dim 합산 + 버킷 boundary ────────────────────────
ax = axes[2]
ax.hist(flat_f16,  bins=120, density=True, alpha=0.55,
        color="steelblue",  label="float16 (all dims)", edgecolor="none")
ax.hist(flat_2bit, bins=120, density=True, alpha=0.55,
        color="darkorange", label="2-bit   (all dims)", edgecolor="none")

for bv, bc in zip(bucket_vals_clean, colors_b):
    ax.axvline(bv, color=bc, lw=2.0, linestyle="--", alpha=0.9,
               label=f"bucket {bv:+.4f}")

# 버킷 경계 (Voronoi midpoint)
boundaries = [(bucket_vals_clean[i] + bucket_vals_clean[i+1]) / 2
              for i in range(len(bucket_vals_clean)-1)]
for b in boundaries:
    ax.axvline(b, color="black", lw=1.0, linestyle=":", alpha=0.5)

ax.set_xlabel("Residual value", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("All Dims: float16 vs 2-bit\n(dashed=bucket centers, dotted=bucket boundaries)", fontweight="bold")
ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(OUT_DIR, "residual_reconstruction.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")

# ── 통계 출력 ─────────────────────────────────────────────────
quant_err = flat_f16 - flat_2bit
print(f"\n{'='*50}")
print(f"  Quantization Error (float16 - 2bit)  per-dimension")
print(f"{'='*50}")
print(f"  mean  : {quant_err.mean():+.6f}")
print(f"  std   : {quant_err.std():.6f}")
print(f"  abs mean: {np.abs(quant_err).mean():.6f}")
print(f"  p95   : {np.percentile(np.abs(quant_err), 95):.6f}")
print(f"  Bucket values: {bucket_vals_clean}")
