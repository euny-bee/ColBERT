#!/usr/bin/env python3
"""
visualize_compression.py
-------------------------
float16 vs 2-bit residual 압축 비교 시각화 (6개 figure)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

# ── 경로 설정 ────────────────────────────────────────────────
COLBERT_DIR  = r"C:\Users\dmsdu\ColBERT"
F16_CSV      = os.path.join(COLBERT_DIR, "colbert_float16_raw_2.csv")
BIT2_CSV     = os.path.join(COLBERT_DIR, "colbert_2bit_raw.csv")
OUT_DIR      = os.path.join(COLBERT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 스타일 ────────────────────────────────────────────────────
BLUE   = "#4C8BF5"
ORANGE = "#F5844C"
ALPHA  = 0.6
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


# ============================================================
# 데이터 로드
# ============================================================
def load_data():
    print("Loading CSVs...", flush=True)
    f16  = pd.read_csv(F16_CSV)
    b2   = pd.read_csv(BIT2_CSV)

    c_cols = [f"centroid_dim_{i}" for i in range(128)]
    r_cols = [f"residual_dim_{i}" for i in range(128)]

    # centroid, residual, reconstructed (float32)
    c16  = f16[c_cols].values.astype(np.float32)
    r16  = f16[r_cols].values.astype(np.float32)
    c2   = b2[c_cols].values.astype(np.float32)
    r2   = b2[r_cols].values.astype(np.float32)

    v16  = c16 + r16   # (N, 128)
    v2   = c2  + r2    # (N, 128)

    print(f"  Loaded {len(f16):,} tokens.", flush=True)
    return r16, r2, v16, v2


# ============================================================
# Figure 1 — Residual 값 분포 히스토그램
# ============================================================
def fig1_residual_hist(r16, r2):
    print("Fig 1: Residual distribution histogram...", flush=True)
    # flat하게 펼침 (샘플링: 최대 2M 값)
    flat16 = r16.flatten()
    flat2  = r2.flatten()
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(flat16), size=min(2_000_000, len(flat16)), replace=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat16[idx], bins=200, color=BLUE,   alpha=ALPHA, density=True, label="float16")
    ax.hist(flat2[idx],  bins=200, color=ORANGE, alpha=ALPHA, density=True, label="2-bit")
    ax.set_xlabel("Residual value")
    ax.set_ylabel("Density")
    ax.set_title("Figure 1 — Residual Value Distribution")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_residual_hist.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ============================================================
# Figure 2 — PCA Scatter (재구성 벡터)
# ============================================================
def fig2_pca_scatter(v16, v2):
    print("Fig 2: PCA scatter...", flush=True)
    N = len(v16)
    rng = np.random.default_rng(42)
    # 최대 5000개 샘플 (PCA 시각화용)
    idx = rng.choice(N, size=min(5000, N), replace=False)

    all_vecs = np.vstack([v16[idx], v2[idx]])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_vecs)
    p16 = pca.transform(v16[idx])
    p2  = pca.transform(v2[idx])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(p16[:, 0], p16[:, 1], s=4, color=BLUE,   alpha=0.4, label="float16")
    ax.scatter(p2[:, 0],  p2[:, 1],  s=4, color=ORANGE, alpha=0.4, label="2-bit")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("Figure 2 — PCA of Reconstructed Vectors (centroid+residual)")
    ax.legend(markerscale=3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_pca_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ============================================================
# Figure 3 — 코사인 유사도 분포
# ============================================================
def fig3_cosine_dist(v16, v2):
    print("Fig 3: Cosine similarity distribution...", flush=True)
    # L2 정규화 후 dot product
    n16   = v16 / (np.linalg.norm(v16, axis=1, keepdims=True) + 1e-9)
    n2    = v2  / (np.linalg.norm(v2,  axis=1, keepdims=True) + 1e-9)
    cos   = (n16 * n2).sum(axis=1)   # (N,)

    mean_cos = cos.mean()
    med_cos  = np.median(cos)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cos, bins=100, color=BLUE, alpha=0.8, edgecolor="none")
    ax.axvline(mean_cos, color="red",    linestyle="--", label=f"mean={mean_cos:.4f}")
    ax.axvline(med_cos,  color="orange", linestyle=":",  label=f"median={med_cos:.4f}")
    ax.set_xlabel("Cosine similarity (float16 vs 2-bit)")
    ax.set_ylabel("Token count")
    ax.set_title("Figure 3 — Cosine Similarity: float16 vs 2-bit Reconstructed Vectors")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_cosine_dist.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)
    print(f"  Cosine sim  mean={mean_cos:.4f}  median={med_cos:.4f}", flush=True)


# ============================================================
# Figure 4 — Per-dimension 평균 비교 (Line Plot)
# ============================================================
def fig4_perdim_mean(v16, v2):
    print("Fig 4: Per-dimension mean...", flush=True)
    mean16 = v16.mean(axis=0)   # (128,)
    mean2  = v2.mean(axis=0)    # (128,)
    dims   = np.arange(128)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dims, mean16, color=BLUE,   alpha=0.8, linewidth=1.2, label="float16")
    ax.plot(dims, mean2,  color=ORANGE, alpha=0.8, linewidth=1.2, label="2-bit", linestyle="--")
    ax.set_xlabel("Dimension index")
    ax.set_ylabel("Mean (centroid + residual)")
    ax.set_title("Figure 4 — Per-dimension Mean of Reconstructed Vectors")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_perdim_mean.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ============================================================
# Figure 5 — 재구성값 1:1 Scatter (density)
# ============================================================
def fig5_value_scatter(v16, v2):
    print("Fig 5: Value-level scatter (hexbin density)...", flush=True)
    # 전체 (N×128) 쌍 → hexbin으로 density 표시
    x = v16.flatten()
    y = v2.flatten()

    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(x, y, gridsize=100, cmap="Blues", mincnt=1,
                   norm=mcolors.LogNorm())
    fig.colorbar(hb, ax=ax, label="log count")
    # y=x 기준선
    lim = max(abs(x).max(), abs(y).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("float16  (centroid + residual)")
    ax.set_ylabel("2-bit    (centroid + residual)")
    ax.set_title("Figure 5 — Per-value Scatter: float16 vs 2-bit")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_value_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ============================================================
# Figure 6 — Squared L2 Norm 비교  Σ(centroid+residual)²
# ============================================================
def fig6_norm_comparison(v16, v2):
    print("Fig 6: Squared L2 norm comparison...", flush=True)
    norm16 = (v16 ** 2).sum(axis=1)   # (N,)
    norm2  = (v2  ** 2).sum(axis=1)   # (N,)
    diff   = norm16 - norm2

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 6A — 히스토그램 overlay
    ax = axes[0]
    ax.hist(norm16, bins=100, color=BLUE,   alpha=ALPHA, density=True, label="float16")
    ax.hist(norm2,  bins=100, color=ORANGE, alpha=ALPHA, density=True, label="2-bit")
    ax.set_xlabel("||v||²")
    ax.set_ylabel("Density")
    ax.set_title("6A — ||v||² Distribution")
    ax.legend()

    # 6B — 토큰별 Scatter
    ax = axes[1]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(norm16), size=min(10000, len(norm16)), replace=False)
    ax.scatter(norm16[idx], norm2[idx], s=4, alpha=0.3, color=BLUE)
    lim = max(norm16.max(), norm2.max()) * 1.02
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("float16  ||v||²")
    ax.set_ylabel("2-bit    ||v||²")
    ax.set_title("6B — Token-level Scatter")
    ax.legend()

    # 6C — 차이값 분포
    ax = axes[2]
    ax.hist(diff, bins=100, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(0,          color="black", linestyle="-",  linewidth=1)
    ax.axvline(diff.mean(),color="red",   linestyle="--", label=f"mean={diff.mean():.4f}")
    ax.set_xlabel("||v_float16||² − ||v_2bit||²")
    ax.set_ylabel("Token count")
    ax.set_title("6C — Norm Difference Distribution")
    ax.legend()

    fig.suptitle("Figure 6 — Squared L2 Norm: float16 vs 2-bit", fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_norm_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)
    print(f"  ||v||² mean  float16={norm16.mean():.4f}  2-bit={norm2.mean():.4f}", flush=True)
    print(f"  diff  mean={diff.mean():.4f}  std={diff.std():.4f}", flush=True)


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    r16, r2, v16, v2 = load_data()

    fig1_residual_hist(r16, r2)
    fig2_pca_scatter(v16, v2)
    fig3_cosine_dist(v16, v2)
    fig4_perdim_mean(v16, v2)
    fig5_value_scatter(v16, v2)
    fig6_norm_comparison(v16, v2)

    print("\nAll figures saved to:", OUT_DIR)
