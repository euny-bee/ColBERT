#!/usr/bin/env python3
"""
visualize_quantization.py
--------------------------
float16 vs 2-bit residual 비교 시각화 (5 figures)

Figure 1: Residual 값 분포 비교 (continuous vs 4 discrete values)
Figure 2: Per-token L2 이동거리 분포
Figure 3: Per-dimension 평균 절대 오차 (128 dims)
Figure 4: PCA 2D scatter — 벡터 이동 방향 시각화
Figure 5: ||centroid + residual||² 분포 (unit sphere 이탈 정도)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLBERT_DIR = os.path.expanduser("~/ColBERT")
F16_CSV     = os.path.join(COLBERT_DIR, "colbert_float16_correct.csv")
BIT2_CSV    = os.path.join(COLBERT_DIR, "colbert_2bit_correct.csv")
OUT_DIR     = COLBERT_DIR
DIM         = 128

CENTROID_COLS = [f"centroid_dim_{i}" for i in range(DIM)]
RESIDUAL_COLS = [f"residual_dim_{i}" for i in range(DIM)]


# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
def load_data():
    print("Loading float16 CSV...", flush=True)
    df_f16  = pd.read_csv(F16_CSV)
    print("Loading 2-bit CSV...",   flush=True)
    df_2bit = pd.read_csv(BIT2_CSV)

    centroid      = df_f16[CENTROID_COLS].values.astype(np.float32)
    residual_f16  = df_f16[RESIDUAL_COLS].values.astype(np.float32)
    residual_2bit = df_2bit[RESIDUAL_COLS].values.astype(np.float32)

    v_f16  = centroid + residual_f16
    v_2bit = centroid + residual_2bit

    print(f"  {len(v_f16):,} tokens loaded", flush=True)
    return centroid, residual_f16, residual_2bit, v_f16, v_2bit


# ─────────────────────────────────────────────────────────────
# Figure 1: Residual 값 분포
# ─────────────────────────────────────────────────────────────
def fig1_residual_distribution(residual_f16, residual_2bit, n):
    flat_f16  = residual_f16.flatten()
    flat_2bit = residual_2bit.flatten()

    # 속도를 위해 최대 500K 샘플
    if len(flat_f16) > 500_000:
        idx = np.random.choice(len(flat_f16), 500_000, replace=False)
        flat_f16  = flat_f16[idx]
        flat_2bit = flat_2bit[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat_f16,  bins=300, alpha=0.6, color="steelblue",
            label="float16 (continuous)", density=True)
    ax.hist(flat_2bit, bins=20,  alpha=0.7, color="tomato",
            label="2-bit (4 discrete values)", density=True)

    unique_vals = np.unique(flat_2bit)
    for v in unique_vals:
        ax.axvline(v, color="tomato", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Residual value")
    ax.set_ylabel("Density")
    ax.set_title(f"Figure 1: Residual Value Distribution\nfloat16 (continuous) vs 2-bit (4 discrete values)  [N={n:,} tokens]")
    ax.set_xlim(-0.35, 0.35)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig1_residual_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# Figure 2a: ||r||² 분포 비교
# ─────────────────────────────────────────────────────────────
def fig2a_residual_norm(residual_f16, residual_2bit, n):
    norm_r_f16  = (residual_f16 ** 2).sum(axis=1)
    norm_r_2bit = (residual_2bit ** 2).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norm_r_f16,  bins=100, alpha=0.6, color="steelblue",
            label=f"float16 (mean={norm_r_f16.mean():.4f})",  density=True)
    ax.hist(norm_r_2bit, bins=100, alpha=0.6, color="tomato",
            label=f"2-bit   (mean={norm_r_2bit.mean():.4f})", density=True)
    ax.set_xlabel("||r||²  per token")
    ax.set_ylabel("Density")
    ax.set_title(f"Figure 2a: Per-token Residual Norm  ||r||²\nfloat16 vs 2-bit  [N={n:,} tokens]")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig2a_residual_norm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# Figure 2b: ||c||² 분포
# ─────────────────────────────────────────────────────────────
def fig2b_centroid_norm(centroid, n):
    norm_c = (centroid ** 2).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norm_c, bins=100, alpha=0.8, color="seagreen",
            label=f"centroid (mean={norm_c.mean():.4f})", density=True)
    ax.set_xlabel("||c||²  per token")
    ax.set_ylabel("Density")
    ax.set_title(f"Figure 2b: Per-token Centroid Norm  ||c||²\n[N={n:,} tokens]")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig2b_centroid_norm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# Figure 2c: ||c+r||² 분포 비교
# ─────────────────────────────────────────────────────────────
def fig2c_reconstruction_norm(v_f16, v_2bit, n):
    norm_v_f16  = (v_f16  ** 2).sum(axis=1)
    norm_v_2bit = (v_2bit ** 2).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norm_v_f16,  bins=100, alpha=0.6, color="steelblue",
            label=f"float16 (mean={norm_v_f16.mean():.4f})",  density=True)
    ax.hist(norm_v_2bit, bins=100, alpha=0.6, color="tomato",
            label=f"2-bit   (mean={norm_v_2bit.mean():.4f})", density=True)
    ax.axvline(1.0, color="black", linestyle=":", alpha=0.7, label="ideal = 1.0")
    ax.set_xlabel("||c+r||²  per token")
    ax.set_ylabel("Density")
    ax.set_title(f"Figure 2c: Per-token Reconstruction Norm  ||c+r||²\nfloat16 vs 2-bit  [N={n:,} tokens]")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig2c_reconstruction_norm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# Figure 3: Per-dimension std 비교
# ─────────────────────────────────────────────────────────────
def fig3_per_dim_mae(residual_f16, residual_2bit, n):
    std_f16  = residual_f16.std(axis=0)    # (128,)
    std_2bit = residual_2bit.std(axis=0)   # (128,)

    x = np.arange(DIM)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, std_f16,  color="steelblue", linewidth=1.2, alpha=0.9,
            label=f"float16  (mean std={std_f16.mean():.4f})")
    ax.plot(x, std_2bit, color="tomato",    linewidth=1.2, alpha=0.9,
            label=f"2-bit    (mean std={std_2bit.mean():.4f})")

    ax.set_xlabel("Dimension index")
    ax.set_ylabel("Std of residual values")
    ax.set_title(f"Figure 3: Per-dimension Residual Std\nfloat16 vs 2-bit  [N={n:,} tokens]")
    ax.set_xlim(0, DIM - 1)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig3_per_dim_mae.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# Figure 4: PCA 2D scatter — 벡터 이동
# ─────────────────────────────────────────────────────────────
def pca_numpy(X, n_components=2):
    """numpy로 구현한 PCA (sklearn 불필요)"""
    X = X - X.mean(axis=0)
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order[:n_components]]
    return X @ components, eigvals[order[:n_components]] / eigvals.sum()


def fig4_pca_scatter(v_f16, v_2bit):
    N_SAMPLE = 600
    np.random.seed(42)
    idx = np.random.choice(len(v_f16), N_SAMPLE, replace=False)

    combined = np.vstack([v_f16[idx], v_2bit[idx]])
    proj, var_ratio = pca_numpy(combined, n_components=2)

    p_f16  = proj[:N_SAMPLE]
    p_2bit = proj[N_SAMPLE:]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 화살표 (일부만, 너무 많으면 지저분)
    N_ARROW = 200
    for i in range(N_ARROW):
        ax.annotate(
            "", xy=(p_2bit[i, 0], p_2bit[i, 1]),
            xytext=(p_f16[i, 0], p_f16[i, 1]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.25, lw=0.6),
        )

    ax.scatter(p_f16[:, 0],  p_f16[:, 1],  c="steelblue", s=12, alpha=0.8,
               label="float16", zorder=4)
    ax.scatter(p_2bit[:, 0], p_2bit[:, 1], c="tomato",    s=12, alpha=0.8,
               label="2-bit",   zorder=4)

    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    ax.set_title(f"Figure 4: PCA 2D Projection — Vector Displacement\n"
                 f"float16 → 2-bit quantization  (n={N_SAMPLE} tokens)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig4_pca_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# Figure 5: ||centroid + residual||² 분포
# ─────────────────────────────────────────────────────────────
def fig5_norm_distribution(v_f16, v_2bit, n):
    # float16은 항상 정확히 1.0 (trivial) → 2-bit만 표시
    norm_2bit = (v_2bit ** 2).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norm_2bit, bins=100, alpha=0.8, color="tomato",
            label=f"2-bit (mean={norm_2bit.mean():.4f})", density=True)
    ax.axvline(1.0, color="steelblue", linestyle="--", linewidth=1.5,
               label="float16 = 1.0 (exact)")
    ax.axvline(norm_2bit.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"2-bit mean = {norm_2bit.mean():.4f}")

    ax.set_xlabel("||centroid + residual||²")
    ax.set_ylabel("Density")
    ax.set_title(f"Figure 5: 2-bit Reconstruction Norm Distribution  [N={n:,} tokens]\n"
                 "(float16 = exactly 1.0, 2-bit deviates from unit sphere)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig5_norm_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    centroid, residual_f16, residual_2bit, v_f16, v_2bit = load_data()
    n = len(v_f16)

    print("Figure 1: Residual distribution...", flush=True)
    fig1_residual_distribution(residual_f16, residual_2bit, n)

    print("Figure 2a: Residual norm...", flush=True)
    fig2a_residual_norm(residual_f16, residual_2bit, n)

    print("Figure 2b: Centroid norm...", flush=True)
    fig2b_centroid_norm(centroid, n)

    print("Figure 2c: Reconstruction norm...", flush=True)
    fig2c_reconstruction_norm(v_f16, v_2bit, n)

    print("Figure 3: Per-dim std...", flush=True)
    fig3_per_dim_mae(residual_f16, residual_2bit, n)

    print("Figure 4: PCA scatter...", flush=True)
    fig4_pca_scatter(v_f16, v_2bit)

    print("Figure 5: Norm distribution...", flush=True)
    fig5_norm_distribution(v_f16, v_2bit, n)

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    main()
