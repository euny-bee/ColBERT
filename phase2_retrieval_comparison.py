#!/usr/bin/env python3
"""
phase2_retrieval_comparison.py
-------------------------------
A (768-dim float32, msmarco-bert-base-dot-v5, MS MARCO 학습)
B (128-dim float32, colbertv2.0 projection, MS MARCO 학습)
C (128-dim float16, colbertv2.0 projection)
세 가지 검색 성능 비교 + ranking 보존율 분석

Phase 2: Recall@5/10/100, MRR@10, nDCG@10
Phase 3: Spearman rank correlation, Top-10 overlap, 쿼리별 손실 분석
Dataset: TREC-COVID (171k passages, 50 queries)
"""

import os
import sys
import json
import time
import torch
import numpy as np
from collections import defaultdict

COLBERT_DIR   = os.path.expanduser("~/ColBERT")
CORPUS_PATH   = "D:/beir/trec-covid/corpus.jsonl"
QUERIES_PATH  = "D:/beir/trec-covid/queries.jsonl"
QRELS_PATH    = "D:/beir/trec-covid/qrels/test.tsv"
COLBERT_CKPT  = "colbert-ir/colbertv2.0"
MSMARCO_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"
BATCH_SIZE    = 64
CACHE_DIR     = os.path.join(COLBERT_DIR, "emb_cache")

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────

def load_corpus(path):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc["_id"])
            title = doc.get("title", "")
            text  = doc.get("text", "")
            texts.append((title + " " + text).strip())
    log(f"Corpus: {len(ids)} passages")
    return ids, texts


def load_queries(path):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            ids.append(q["_id"])
            texts.append(q["text"])
    log(f"Queries: {len(ids)}")
    return ids, texts


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            qid, pid, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels[qid][pid] = score
    log(f"Qrels: {len(qrels)} queries")
    return qrels


# ── 2. A 인코딩: msmarco-bert-base-dot-v5 (768-dim) ──────────────────────────

def encode_A(texts, max_length, desc=""):
    from transformers import AutoTokenizer, AutoModel

    log(f"  Loading {MSMARCO_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MSMARCO_MODEL)
    model     = AutoModel.from_pretrained(MSMARCO_MODEL)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(input_ids, attention_mask=attention_mask)
            # CLS 토큰 (msmarco-bert-base-dot-v5 방식)
            emb = out.last_hidden_state[:, 0, :].float()
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu())

            if (i // BATCH_SIZE) % 20 == 0:
                log(f"  A {desc}: {i+len(batch)}/{len(texts)}")

    del model
    torch.cuda.empty_cache()
    return torch.cat(all_embs)  # (N, 768) float32


# ── 3. B/C 인코딩: colbertv2.0 projection (128-dim) ──────────────────────────

def encode_BC(texts, max_length, desc=""):
    from colbert.modeling.base_colbert import BaseColBERT
    from colbert.infra import ColBERTConfig

    log(f"  Loading {COLBERT_CKPT}...")
    config = ColBERTConfig(doc_maxlen=220, query_maxlen=32)
    model  = BaseColBERT(COLBERT_CKPT, colbert_config=config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = model.raw_tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            bert_out = model.bert(input_ids, attention_mask=attention_mask)[0]  # (B, seq, 768)
            proj     = model.linear(bert_out)                                   # (B, seq, 128)

            # mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            emb  = (proj * mask).sum(dim=1) / mask.sum(dim=1)
            emb  = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
            all_embs.append(emb.cpu())

            if (i // BATCH_SIZE) % 20 == 0:
                log(f"  B/C {desc}: {i+len(batch)}/{len(texts)}")

    del model
    torch.cuda.empty_cache()
    return torch.cat(all_embs)  # (N, 128) float32


# ── 4. 검색 메트릭 계산 ───────────────────────────────────────────────────────

def compute_metrics(scores_matrix, query_ids, corpus_ids, qrels, k_values=[5, 10, 50, 100]):
    results = {f"Recall@{k}": [] for k in k_values}
    results["MRR@10"]  = []
    results["nDCG@10"] = []
    per_query = {}

    for q_i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue

        rel_pids    = set(qrels[qid].keys())
        scores      = scores_matrix[q_i]
        ranked_idx  = scores.argsort(descending=True).tolist()
        ranked_pids = [corpus_ids[i] for i in ranked_idx]

        # Recall@k
        for k in k_values:
            top_k  = set(ranked_pids[:k])
            recall = len(top_k & rel_pids) / len(rel_pids) if rel_pids else 0.0
            results[f"Recall@{k}"].append(recall)

        # MRR@10
        mrr = 0.0
        for rank, pid in enumerate(ranked_pids[:10], 1):
            if pid in rel_pids:
                mrr = 1.0 / rank
                break
        results["MRR@10"].append(mrr)

        # nDCG@10
        dcg, idcg = 0.0, 0.0
        ideal_rels = sorted([qrels[qid].get(p, 0) for p in rel_pids], reverse=True)
        for rank, pid in enumerate(ranked_pids[:10], 1):
            rel = qrels[qid].get(pid, 0)
            dcg += rel / np.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels[:10], 1):
            idcg += rel / np.log2(rank + 1)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        results["nDCG@10"].append(ndcg)

        per_query[qid] = {
            "ranked_pids": ranked_pids[:100],
            "nDCG@10": ndcg,
            "MRR@10": mrr,
        }

    return {k: np.mean(v) for k, v in results.items()}, per_query


# ── 5. Phase 3: Ranking 보존율 분석 ──────────────────────────────────────────

def analyze_ranking(per_query_A, per_query_B, per_query_C, query_ids, query_texts):
    from scipy.stats import spearmanr
    log("\n=== Phase 3: Ranking 보존율 분석 ===")

    spearman_AB, spearman_AC, spearman_BC = [], [], []
    overlap_AB,  overlap_AC,  overlap_BC  = [], [], []

    for qid in query_ids:
        if qid not in per_query_A or qid not in per_query_B:
            continue

        pids_A = per_query_A[qid]["ranked_pids"]
        pids_B = per_query_B[qid]["ranked_pids"]
        pids_C = per_query_C[qid]["ranked_pids"]

        # Spearman rank correlation (top-100 공통 passage 기준)
        pairs = [(pids_A, pids_B, spearman_AB),
                 (pids_A, pids_C, spearman_AC),
                 (pids_B, pids_C, spearman_BC)]
        for (pids_X, pids_Y, sp_list) in pairs:
            common = list(set(pids_X) & set(pids_Y))
            if len(common) > 5:
                rank_X = {p: i for i, p in enumerate(pids_X)}
                rank_Y = {p: i for i, p in enumerate(pids_Y)}
                corr, _ = spearmanr([rank_X[p] for p in common],
                                     [rank_Y[p] for p in common])
                sp_list.append(corr)

        # Top-10 overlap
        top10_A = set(pids_A[:10])
        top10_B = set(pids_B[:10])
        top10_C = set(pids_C[:10])
        overlap_AB.append(len(top10_A & top10_B))
        overlap_AC.append(len(top10_A & top10_C))
        overlap_BC.append(len(top10_B & top10_C))

    log(f"Spearman corr A vs B (top-100): mean={np.mean(spearman_AB):.4f}, min={np.min(spearman_AB):.4f}")
    log(f"Spearman corr A vs C (top-100): mean={np.mean(spearman_AC):.4f}, min={np.min(spearman_AC):.4f}")
    log(f"Spearman corr B vs C (top-100): mean={np.mean(spearman_BC):.4f}, min={np.min(spearman_BC):.4f}")
    log(f"Top-10 overlap A vs B: mean={np.mean(overlap_AB):.2f}/10")
    log(f"Top-10 overlap A vs C: mean={np.mean(overlap_AC):.2f}/10")
    log(f"Top-10 overlap B vs C: mean={np.mean(overlap_BC):.2f}/10")

    # 쿼리별 nDCG@10 차이 (A → B)
    qid2text = dict(zip(query_ids, query_texts))
    ndcg_diff = {qid: per_query_B[qid]["nDCG@10"] - per_query_A[qid]["nDCG@10"]
                 for qid in query_ids if qid in per_query_A and qid in per_query_B}
    sorted_diff = sorted(ndcg_diff.items(), key=lambda x: x[1])

    log("\n가장 손실 큰 쿼리 (A→B nDCG@10 하락):")
    for qid, diff in sorted_diff[:3]:
        log(f"  [{qid}] {diff:+.4f} | {qid2text.get(qid,'')[:80]}")
    log("가장 이득 큰 쿼리 (A→B nDCG@10 상승):")
    for qid, diff in sorted_diff[-3:]:
        log(f"  [{qid}] {diff:+.4f} | {qid2text.get(qid,'')[:80]}")

    return spearman_AB, spearman_AC, spearman_BC, overlap_AB, overlap_AC, overlap_BC


# ── 6. 시각화 ─────────────────────────────────────────────────────────────────

def plot_results(metrics_A, metrics_B, metrics_C,
                 spearman_AB, spearman_AC, spearman_BC,
                 overlap_AB, overlap_AC, overlap_BC):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLOR_AB = "#2196F3"   # 파란색  — A vs B (768 vs 128 f32)
    COLOR_AC = "#FF5722"   # 진주황  — A vs C (768 vs 128 f16)
    COLOR_BC = "#4CAF50"   # 초록    — B vs C (128 f32 vs 128 f16)

    metric_names = ["nDCG@10", "MRR@10", "Recall@5", "Recall@10", "Recall@50", "Recall@100"]
    a_vals = [metrics_A[m] for m in metric_names]
    b_vals = [metrics_B[m] for m in metric_names]
    c_vals = [metrics_C[m] for m in metric_names]
    x      = np.arange(len(metric_names))
    width  = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Phase 2+3: Retrieval Analysis\n"
        "A = msmarco-bert-base-dot-v5 (768-dim f32)  |  "
        "B = colbertv2.0 (128-dim f32)  |  "
        "C = colbertv2.0 (128-dim f16)",
        fontsize=12
    )

    # ── (0,0) 절대 성능 ──
    ax = axes[0, 0]
    ba = ax.bar(x - width, a_vals, width, label="A (768-dim f32)", color="steelblue", alpha=0.85)
    bb = ax.bar(x,         b_vals, width, label="B (128-dim f32)", color=COLOR_AB,    alpha=0.85)
    bc = ax.bar(x + width, c_vals, width, label="C (128-dim f16)", color=COLOR_AC,    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Absolute Retrieval Scores")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bars in [ba, bb, bc]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    # ── (0,1) A 기준 차이 (B-A, C-A, B-C) ──
    ax = axes[0, 1]
    diff_ba = [b - a for a, b in zip(a_vals, b_vals)]
    diff_ca = [c - a for a, c in zip(a_vals, c_vals)]
    diff_bc = [c - b for b, c in zip(b_vals, c_vals)]
    w3 = width * 0.85
    bb2 = ax.bar(x - w3, diff_ba, w3, label="B - A (768→128 f32)", color=COLOR_AB, alpha=0.85)
    bc2 = ax.bar(x,      diff_ca, w3, label="C - A (768→128 f16)", color=COLOR_AC, alpha=0.85)
    bd2 = ax.bar(x + w3, diff_bc, w3, label="B - C (f32 vs f16)",  color=COLOR_BC, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.set_ylabel("Score Difference")
    ax.set_title("Pairwise Score Differences")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bars, diffs in [(bb2, diff_ba), (bc2, diff_ca), (bd2, diff_bc)]:
        for bar, d in zip(bars, diffs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.002 if d >= 0 else -0.007),
                    f"{d:+.3f}", ha="center", va="bottom", fontsize=7)

    # ── (1,0) Spearman rank correlation — A vs B / A vs C / B vs C ──
    bins = np.linspace(-0.6, 1.0, 20)
    ax = axes[1, 0]
    ax.hist(spearman_AB, bins=bins, alpha=0.75, color=COLOR_AB,
            label=f"A vs B  mean={np.mean(spearman_AB):.3f}")
    ax.hist(spearman_AC, bins=bins, alpha=0.75, color=COLOR_AC,
            label=f"A vs C  mean={np.mean(spearman_AC):.3f}")
    ax.hist(spearman_BC, bins=bins, alpha=0.75, color=COLOR_BC,
            label=f"B vs C  mean={np.mean(spearman_BC):.3f}")
    for val, col in [(np.mean(spearman_AB), COLOR_AB),
                     (np.mean(spearman_AC), COLOR_AC),
                     (np.mean(spearman_BC), COLOR_BC)]:
        ax.axvline(val, color=col, linestyle="--", linewidth=2)
    ax.set_xlabel("Spearman Rank Correlation")
    ax.set_ylabel("Count (queries)")
    ax.set_title("Ranking Preservation (top-100)\nA vs B  /  A vs C  /  B vs C")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── (1,1) Top-10 overlap — A vs B / A vs C / B vs C ──
    bins_ov = np.arange(-0.5, 11.5, 1)
    offsets = [-0.25, 0, 0.25]
    width_ov = 0.25
    ax = axes[1, 1]
    for vals, col, lbl, off in [
        (overlap_AB, COLOR_AB, f"A vs B  mean={np.mean(overlap_AB):.2f}/10", offsets[0]),
        (overlap_AC, COLOR_AC, f"A vs C  mean={np.mean(overlap_AC):.2f}/10", offsets[1]),
        (overlap_BC, COLOR_BC, f"B vs C  mean={np.mean(overlap_BC):.2f}/10", offsets[2]),
    ]:
        counts = np.bincount(vals, minlength=11).astype(float)
        ax.bar(np.arange(11) + off, counts, width=width_ov, color=col, alpha=0.8, label=lbl)
    ax.set_xlabel("Top-10 Overlap Count")
    ax.set_ylabel("Count (queries)")
    ax.set_title("Top-10 Result Overlap\nA vs B  /  A vs C  /  B vs C")
    ax.set_xticks(range(11))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(COLBERT_DIR, "phase2_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log(f"Plot saved: {out_path}")


# ── 7. main ───────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Phase 2+3: Retrieval Comparison")
    log("A=msmarco-bert-768f32 | B=colbertv2-128f32 | C=colbertv2-128f16")
    log("=" * 60)

    corpus_ids, corpus_texts = load_corpus(CORPUS_PATH)
    query_ids,  query_texts  = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── A 인코딩 (캐시 우선) ──
    cache_A_corpus = os.path.join(CACHE_DIR, "A_corpus.pt")
    cache_A_query  = os.path.join(CACHE_DIR, "A_query.pt")
    if os.path.exists(cache_A_corpus) and os.path.exists(cache_A_query):
        log("\n[A] Loading from cache...")
        A_corpus = torch.load(cache_A_corpus, weights_only=True)
        A_query  = torch.load(cache_A_query,  weights_only=True)
    else:
        log("\n[A] msmarco-bert-base-dot-v5 (768-dim)...")
        A_corpus = encode_A(corpus_texts, max_length=220, desc="corpus")
        A_query  = encode_A(query_texts,  max_length=64,  desc="query")
        torch.save(A_corpus, cache_A_corpus)
        torch.save(A_query,  cache_A_query)
        log(f"  Cached to {CACHE_DIR}")
    log(f"  A corpus: {A_corpus.shape}  A query: {A_query.shape}")

    # ── B/C 인코딩 (캐시 우선) ──
    cache_B_corpus = os.path.join(CACHE_DIR, "B_corpus.pt")
    cache_B_query  = os.path.join(CACHE_DIR, "B_query.pt")
    if os.path.exists(cache_B_corpus) and os.path.exists(cache_B_query):
        log("\n[B/C] Loading from cache...")
        B_corpus = torch.load(cache_B_corpus, weights_only=True)
        B_query  = torch.load(cache_B_query,  weights_only=True)
    else:
        log("\n[B/C] colbertv2.0 projection (128-dim)...")
        B_corpus = encode_BC(corpus_texts, max_length=220, desc="corpus")
        B_query  = encode_BC(query_texts,  max_length=32,  desc="query")
        torch.save(B_corpus, cache_B_corpus)
        torch.save(B_query,  cache_B_query)
        log(f"  Cached to {CACHE_DIR}")
    C_corpus = B_corpus.half()
    C_query  = B_query.half()
    log(f"  B corpus: {B_corpus.shape}  B query: {B_query.shape}")

    # ── 검색 ──
    log("\nSearching...")
    scores_A = A_query @ A_corpus.T
    scores_B = B_query @ B_corpus.T
    scores_C = (C_query @ C_corpus.T).float()
    log("  Done.")

    # ── Phase 2: 메트릭 ──
    log("\nComputing metrics...")
    metrics_A, per_query_A = compute_metrics(scores_A, query_ids, corpus_ids, qrels)
    metrics_B, per_query_B = compute_metrics(scores_B, query_ids, corpus_ids, qrels)
    metrics_C, per_query_C = compute_metrics(scores_C, query_ids, corpus_ids, qrels)

    log("\n=== Phase 2 Results ===")
    log(f"{'Metric':<15} {'A (768f32)':>12} {'B (128f32)':>12} {'C (128f16)':>12} {'B-A':>8} {'C-A':>8}")
    log("-" * 70)
    for metric in ["nDCG@10", "MRR@10", "Recall@5", "Recall@10", "Recall@50", "Recall@100"]:
        a = metrics_A[metric]
        b = metrics_B[metric]
        c = metrics_C[metric]
        log(f"{metric:<15} {a:>12.4f} {b:>12.4f} {c:>12.4f} {b-a:>+8.4f} {c-a:>+8.4f}")

    # ── Phase 3: Ranking 보존율 ──
    spearman_AB, spearman_AC, spearman_BC, overlap_AB, overlap_AC, overlap_BC = analyze_ranking(
        per_query_A, per_query_B, per_query_C, query_ids, query_texts
    )

    # ── 시각화 ──
    log("\nPlotting...")
    plot_results(metrics_A, metrics_B, metrics_C,
                 spearman_AB, spearman_AC, spearman_BC,
                 overlap_AB, overlap_AC, overlap_BC)

    log("\nAll done.")


if __name__ == "__main__":
    main()
