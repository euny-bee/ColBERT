#!/usr/bin/env python3
"""
phase1_vector_analysis.py
--------------------------
A (768-dim float32), B (128-dim float32), C (128-dim float16) 벡터 손실 분석

Phase 1-1: 차원 축소 손실 (A -> B)
  - PCA explained variance: 128개 축이 768-dim 분산의 몇 % 설명하는지
  - Cosine similarity 보존율: PCA 128-dim(A') vs ColBERT projection 128-dim(B)

Phase 1-2: Precision 손실 (B -> C)
  - MSE: ||B - C||^2 분포
  - Max absolute error 분포
"""

import os
import sys
import json
import time
import torch
import numpy as np

COLBERT_DIR = os.path.expanduser("~/ColBERT")
CORPUS_PATH = "D:/beir/trec-covid/corpus.jsonl"
CHECKPOINT  = "colbert-ir/colbertv2.0"
SAMPLE_N    = 2000  # 샘플 passage 수
BATCH_SIZE  = 32

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 1. 샘플 passage 로드 ──────────────────────────────────────────────────────

def load_sample_passages(path, n):
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            doc = json.loads(line)
            title = doc.get("title", "")
            text  = doc.get("text", "")
            passages.append((title + " " + text).strip())
    log(f"Loaded {len(passages)} passages")
    return passages


# ── 2. A, B, C 임베딩 추출 ────────────────────────────────────────────────────

def extract_embeddings(passages):
    """
    A: BERT encoder 출력 (768-dim, float32)  — linear projection 전
    B: Linear projection + L2 norm (128-dim, float32) — .half() 전
    C: B.half() (128-dim, float16)
    """
    from colbert.modeling.base_colbert import BaseColBERT
    from colbert.infra import ColBERTConfig

    config = ColBERTConfig(doc_maxlen=220, query_maxlen=32)
    model = BaseColBERT(CHECKPOINT, colbert_config=config)
    model.eval()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    device = next(model.parameters()).device

    log(f"Model loaded on {device}")
    log(f"BERT hidden size: {model.bert.config.hidden_size}")
    log(f"Linear projection: {model.linear.weight.shape}")

    all_A, all_B, all_C = [], [], []

    with torch.no_grad():
        for i in range(0, len(passages), BATCH_SIZE):
            batch = passages[i:i + BATCH_SIZE]

            # 토크나이즈
            enc = model.raw_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=220,
                return_tensors="pt"
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # A: BERT encoder 출력 (768-dim, float32)
            bert_out = model.bert(input_ids, attention_mask=attention_mask)[0]  # (B, seq, 768)
            # CLS 토큰만 사용 (position 0)
            A = bert_out[:, 0, :].float().cpu()  # (B, 768)

            # B: Linear projection + L2 norm (128-dim, float32)
            proj = model.linear(bert_out)         # (B, seq, 128)
            proj_cls = proj[:, 0, :].float()
            B = torch.nn.functional.normalize(proj_cls, p=2, dim=-1).cpu()  # (B, 128)

            # C: float16 변환
            C = B.half().cpu()  # (B, 128)

            all_A.append(A)
            all_B.append(B)
            all_C.append(C)

            if (i // BATCH_SIZE) % 5 == 0:
                log(f"  batch {i//BATCH_SIZE + 1}/{(len(passages)-1)//BATCH_SIZE + 1}")

    A = torch.cat(all_A)  # (N, 768) float32
    B = torch.cat(all_B)  # (N, 128) float32
    C = torch.cat(all_C)  # (N, 128) float16

    log(f"A shape: {A.shape}, dtype: {A.dtype}")
    log(f"B shape: {B.shape}, dtype: {B.dtype}")
    log(f"C shape: {C.shape}, dtype: {C.dtype}")

    return A, B, C


# ── 3. Phase 1-1: 차원 축소 손실 ─────────────────────────────────────────────

def analyze_dim_reduction(A, B):
    log("\n=== Phase 1-1: 차원 축소 손실 (A 768→B 128) ===")
    A_np = A.numpy()
    B_np = B.numpy()

    # --- PCA explained variance ---
    from sklearn.decomposition import PCA
    log("PCA fitting on A (768-dim)...")
    pca = PCA(n_components=128)
    pca.fit(A_np)
    explained = pca.explained_variance_ratio_.sum() * 100
    log(f"PCA 128 components explained variance: {explained:.2f}%")

    # 누적 분산 (32, 64, 128)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    for k in [32, 64, 128]:
        log(f"  Top-{k} PCA components: {cumvar[k-1]:.2f}%")

    # --- A' (PCA 128-dim) 생성 ---
    A_pca = pca.transform(A_np)  # (N, 128)
    # L2 normalize
    A_pca_norm = A_pca / (np.linalg.norm(A_pca, axis=1, keepdims=True) + 1e-9)

    # --- Cosine similarity 보존율: A' vs B ---
    # 쌍별 cosine sim (샘플 500쌍)
    np.random.seed(42)
    idx = np.random.choice(len(A_np), size=min(500, len(A_np)), replace=False)
    A_sub = A_pca_norm[idx]   # (500, 128)
    B_sub = B_np[idx]         # (500, 128)
    B_sub_norm = B_sub / (np.linalg.norm(B_sub, axis=1, keepdims=True) + 1e-9)

    # 쌍별 cosine sim (i vs j, i≠j)
    sim_A = A_sub @ A_sub.T   # (500, 500)
    sim_B = B_sub_norm @ B_sub_norm.T

    # 대각 제외
    mask = ~np.eye(len(idx), dtype=bool)
    sim_A_flat = sim_A[mask]
    sim_B_flat = sim_B[mask]

    from scipy.stats import spearmanr
    corr, pval = spearmanr(sim_A_flat, sim_B_flat)
    log(f"Cosine similarity Spearman rank correlation (A' vs B): {corr:.4f} (p={pval:.2e})")

    diff = sim_A_flat - sim_B_flat
    log(f"Cosine sim difference (A' - B): mean={diff.mean():.4f}, std={diff.std():.4f}, max={diff.max():.4f}, min={diff.min():.4f}")

    return pca, A_pca_norm, sim_A_flat, sim_B_flat


# ── 4. Phase 1-2: Precision 손실 ─────────────────────────────────────────────

def analyze_precision_loss(B, C):
    log("\n=== Phase 1-2: Precision 손실 (B float32 → C float16) ===")
    B_np = B.numpy()           # float32
    C_np = C.float().numpy()   # float16 → float32로 변환해서 비교

    diff = B_np - C_np  # (N, 128)

    # MSE per vector
    mse_per_vec = (diff ** 2).mean(axis=1)
    log(f"MSE per vector: mean={mse_per_vec.mean():.6e}, std={mse_per_vec.std():.6e}, max={mse_per_vec.max():.6e}")

    # Max absolute error per vector
    max_abs_per_vec = np.abs(diff).max(axis=1)
    log(f"Max abs error per vector: mean={max_abs_per_vec.mean():.6e}, std={max_abs_per_vec.std():.6e}, max={max_abs_per_vec.max():.6e}")

    # Cosine similarity between B and C
    B_norm = B_np / (np.linalg.norm(B_np, axis=1, keepdims=True) + 1e-9)
    C_norm = C_np / (np.linalg.norm(C_np, axis=1, keepdims=True) + 1e-9)
    cos_sim = (B_norm * C_norm).sum(axis=1)  # (N,)
    log(f"Cosine similarity B vs C: mean={cos_sim.mean():.6f}, min={cos_sim.min():.6f}, std={cos_sim.std():.6e}")

    # 전체 dimension 오차 분포
    all_diff = diff.flatten()
    log(f"Per-dimension error: mean={all_diff.mean():.6e}, std={all_diff.std():.6e}, "
        f"p99={np.percentile(np.abs(all_diff), 99):.6e}")

    return mse_per_vec, max_abs_per_vec, cos_sim, all_diff


# ── 5. 시각화 ─────────────────────────────────────────────────────────────────

def plot_results(pca, sim_A_flat, sim_B_flat, mse_per_vec, max_abs_per_vec, cos_sim, all_diff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Phase 1: Vector Loss Analysis\n(A=768-dim float32, B=128-dim float32, C=128-dim float16)", fontsize=13)

    # ── 1-1-a: PCA cumulative explained variance ──
    ax = axes[0, 0]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, 129), cumvar, color="steelblue", linewidth=2)
    ax.axhline(cumvar[31],  color="orange", linestyle="--", label=f"Top-32: {cumvar[31]:.1f}%")
    ax.axhline(cumvar[63],  color="green",  linestyle="--", label=f"Top-64: {cumvar[63]:.1f}%")
    ax.axhline(cumvar[127], color="red",    linestyle="--", label=f"Top-128: {cumvar[127]:.1f}%")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA Explained Variance\n(768-dim → 128-dim)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 1-1-b: Cosine sim scatter A' vs B ──
    ax = axes[0, 1]
    sample = np.random.choice(len(sim_A_flat), size=min(5000, len(sim_A_flat)), replace=False)
    ax.scatter(sim_A_flat[sample], sim_B_flat[sample], alpha=0.1, s=2, color="steelblue")
    lim = [min(sim_A_flat.min(), sim_B_flat.min()) - 0.05,
           max(sim_A_flat.max(), sim_B_flat.max()) + 0.05]
    ax.plot(lim, lim, "r--", linewidth=1, label="y=x (perfect)")
    ax.set_xlabel("Cosine Sim (A' = PCA 128-dim)")
    ax.set_ylabel("Cosine Sim (B = ColBERT projection 128-dim)")
    ax.set_title("Cosine Similarity: A' vs B\n(pair-wise, 500 passages)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 1-1-c: Cosine sim difference histogram ──
    ax = axes[0, 2]
    diff_ab = sim_A_flat - sim_B_flat
    ax.hist(diff_ab, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(diff_ab.mean(), color="red", linestyle="--", label=f"mean={diff_ab.mean():.3f}")
    ax.set_xlabel("Cosine Sim Difference (A' - B)")
    ax.set_ylabel("Count")
    ax.set_title("Cosine Sim Difference Distribution\nA'(PCA) - B(ColBERT projection)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 1-2-a: MSE per vector histogram ──
    ax = axes[1, 0]
    ax.hist(mse_per_vec, bins=60, color="coral", edgecolor="none", alpha=0.8)
    ax.axvline(mse_per_vec.mean(), color="red", linestyle="--", label=f"mean={mse_per_vec.mean():.2e}")
    ax.set_xlabel("MSE per Vector")
    ax.set_ylabel("Count")
    ax.set_title("MSE Distribution: B(f32) vs C(f16)\nper vector")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 1-2-b: Max abs error per vector histogram ──
    ax = axes[1, 1]
    ax.hist(max_abs_per_vec, bins=60, color="coral", edgecolor="none", alpha=0.8)
    ax.axvline(max_abs_per_vec.mean(), color="red", linestyle="--", label=f"mean={max_abs_per_vec.mean():.2e}")
    ax.set_xlabel("Max Absolute Error per Vector")
    ax.set_ylabel("Count")
    ax.set_title("Max Abs Error Distribution: B(f32) vs C(f16)\nper vector")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 1-2-c: Per-dimension error distribution ──
    ax = axes[1, 2]
    sample_diff = np.random.choice(all_diff, size=min(100000, len(all_diff)), replace=False)
    ax.hist(sample_diff, bins=100, color="coral", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", label="zero")
    ax.set_xlabel("Per-dimension Error (B - C)")
    ax.set_ylabel("Count")
    ax.set_title("Per-dimension Error Distribution\nB(f32) - C(f16)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(COLBERT_DIR, "phase1_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log(f"Plot saved: {out_path}")


# ── 6. main ───────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Phase 1: Vector Loss Analysis (A=768f32, B=128f32, C=128f16)")
    log("=" * 60)

    passages = load_sample_passages(CORPUS_PATH, SAMPLE_N)

    log("\nExtracting embeddings...")
    A, B, C = extract_embeddings(passages)

    pca, A_pca_norm, sim_A_flat, sim_B_flat = analyze_dim_reduction(A, B)
    mse_per_vec, max_abs_per_vec, cos_sim, all_diff = analyze_precision_loss(B, C)

    log("\nPlotting results...")
    plot_results(pca, sim_A_flat, sim_B_flat, mse_per_vec, max_abs_per_vec, cos_sim, all_diff)

    log("\nDone.")


if __name__ == "__main__":
    main()
