#!/usr/bin/env python3
"""
phase2_visualization.py
------------------------
Phase 2 결과 상세 시각화

1. t-SNE: A(768-dim) vs B(128-dim) 벡터 클러스터 구조
2. Score 분포: relevant vs non-relevant 문서 점수 분포 (A vs B)
3. 쿼리별 nDCG@10 scatter (A vs B)
4. Relevant 문서 rank 변화 (A → B)
5. Score gap: discriminability (relevant vs non-relevant)
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
SAMPLE_N      = 2000  # t-SNE용 샘플

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_corpus(path, n=None):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            doc = json.loads(line)
            ids.append(doc["_id"])
            title = doc.get("title", "")
            text  = doc.get("text", "")
            texts.append((title + " " + text).strip())
    return ids, texts


def load_queries(path):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            ids.append(q["_id"])
            texts.append(q["text"])
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
    return qrels


# ── 임베딩 추출 ───────────────────────────────────────────────────────────────

def encode_A(texts, max_length):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MSMARCO_MODEL)
    model     = AutoModel.from_pretrained(MSMARCO_MODEL)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            out = model(enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
            emb = out.last_hidden_state[:, 0, :].float()
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu())
            if (i // BATCH_SIZE) % 20 == 0:
                log(f"  A: {i+len(batch)}/{len(texts)}")
    del model; torch.cuda.empty_cache()
    return torch.cat(all_embs)


def encode_B(texts, max_length):
    from colbert.modeling.base_colbert import BaseColBERT
    from colbert.infra import ColBERTConfig
    config = ColBERTConfig(doc_maxlen=220, query_maxlen=32)
    model  = BaseColBERT(COLBERT_CKPT, colbert_config=config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = model.raw_tokenizer(batch, padding=True, truncation=True,
                                      max_length=max_length, return_tensors="pt")
            bert_out = model.bert(enc["input_ids"].to(device),
                                  attention_mask=enc["attention_mask"].to(device))[0]
            proj = model.linear(bert_out)
            mask = enc["attention_mask"].to(device).unsqueeze(-1).float()
            emb  = (proj * mask).sum(dim=1) / mask.sum(dim=1)
            emb  = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
            all_embs.append(emb.cpu())
            if (i // BATCH_SIZE) % 20 == 0:
                log(f"  B: {i+len(batch)}/{len(texts)}")
    del model; torch.cuda.empty_cache()
    return torch.cat(all_embs)


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_all(corpus_ids, query_ids, query_texts, qrels,
             A_corpus, B_corpus, A_query, B_query):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from scipy.stats import spearmanr

    scores_A = (A_query @ A_corpus.T).numpy()  # (50, N)
    scores_B = (B_query @ B_corpus.T).numpy()

    pid2idx = {pid: i for i, pid in enumerate(corpus_ids)}

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(
        "Phase 2: Vector & Retrieval Visualization\n"
        "A = msmarco-bert-base-dot-v5 (768-dim f32)  |  B = colbertv2.0 (128-dim f32)",
        fontsize=14, y=0.98
    )

    # ─────────────────────────────────────────────────────────────
    # Plot 1: t-SNE (A 768-dim)
    # ─────────────────────────────────────────────────────────────
    log("Computing t-SNE for A (768-dim)...")
    ax1 = fig.add_subplot(4, 2, 1)

    # 샘플: corpus 일부 + 쿼리
    np.random.seed(42)
    sample_idx = np.random.choice(len(corpus_ids), size=min(SAMPLE_N, len(corpus_ids)), replace=False)
    A_sample   = A_corpus[sample_idx].numpy()
    A_q_np     = A_query.numpy()

    # 어떤 샘플이 relevant인지 표시 (첫 번째 쿼리 기준)
    qid0       = query_ids[0]
    rel_set0   = set(qrels.get(qid0, {}).keys())
    is_rel     = np.array([corpus_ids[i] in rel_set0 for i in sample_idx])

    combined_A = np.vstack([A_sample, A_q_np])
    tsne_A     = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(combined_A)
    tsne_corpus_A = tsne_A[:len(sample_idx)]
    tsne_query_A  = tsne_A[len(sample_idx):]

    ax1.scatter(tsne_corpus_A[~is_rel, 0], tsne_corpus_A[~is_rel, 1],
                c="lightblue", s=8, alpha=0.5, label="Non-relevant")
    ax1.scatter(tsne_corpus_A[is_rel, 0],  tsne_corpus_A[is_rel, 1],
                c="red", s=30, alpha=0.8, label=f"Relevant (Q:{qid0})")
    ax1.scatter(tsne_query_A[:, 0], tsne_query_A[:, 1],
                c="black", s=60, marker="*", label="Queries")
    ax1.set_title("t-SNE: A (768-dim)\nCorpus + Queries")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.2)

    # ─────────────────────────────────────────────────────────────
    # Plot 2: t-SNE (B 128-dim)
    # ─────────────────────────────────────────────────────────────
    log("Computing t-SNE for B (128-dim)...")
    ax2 = fig.add_subplot(4, 2, 2)

    B_sample = B_corpus[sample_idx].numpy()
    B_q_np   = B_query.numpy()
    combined_B = np.vstack([B_sample, B_q_np])
    tsne_B     = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(combined_B)
    tsne_corpus_B = tsne_B[:len(sample_idx)]
    tsne_query_B  = tsne_B[len(sample_idx):]

    ax2.scatter(tsne_corpus_B[~is_rel, 0], tsne_corpus_B[~is_rel, 1],
                c="lightblue", s=8, alpha=0.5, label="Non-relevant")
    ax2.scatter(tsne_corpus_B[is_rel, 0],  tsne_corpus_B[is_rel, 1],
                c="red", s=30, alpha=0.8, label=f"Relevant (Q:{qid0})")
    ax2.scatter(tsne_query_B[:, 0], tsne_query_B[:, 1],
                c="black", s=60, marker="*", label="Queries")
    ax2.set_title("t-SNE: B (128-dim)\nCorpus + Queries")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)

    # ─────────────────────────────────────────────────────────────
    # Plot 3 & 4: Score 분포 (relevant vs non-relevant)
    # ─────────────────────────────────────────────────────────────
    log("Computing score distributions...")
    ax3 = fig.add_subplot(4, 2, 3)
    ax4 = fig.add_subplot(4, 2, 4)

    rel_scores_A, nrel_scores_A = [], []
    rel_scores_B, nrel_scores_B = [], []

    for q_i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel_pids = set(qrels[qid].keys())
        for p_i, pid in enumerate(corpus_ids):
            s_A = scores_A[q_i, p_i]
            s_B = scores_B[q_i, p_i]
            if pid in rel_pids:
                rel_scores_A.append(s_A)
                rel_scores_B.append(s_B)
            else:
                nrel_scores_A.append(s_A)
                nrel_scores_B.append(s_B)

    # 너무 많으면 샘플링
    if len(nrel_scores_A) > 50000:
        idx = np.random.choice(len(nrel_scores_A), 50000, replace=False)
        nrel_scores_A = [nrel_scores_A[i] for i in idx]
        nrel_scores_B = [nrel_scores_B[i] for i in idx]

    # relevant: A(파란실선) vs B(주황실선), non-relevant: A(파란점선) vs B(주황점선)
    ax3.hist(rel_scores_A,  bins=80, alpha=0.5, color="steelblue", density=True,
             label=f"A Relevant (mean={np.mean(rel_scores_A):.3f})")
    ax3.hist(rel_scores_B,  bins=80, alpha=0.5, color="coral",     density=True,
             label=f"B Relevant (mean={np.mean(rel_scores_B):.3f})")
    ax3.hist(nrel_scores_A, bins=80, alpha=0.3, color="steelblue", density=True,
             linestyle="--", histtype="step", linewidth=1.5,
             label=f"A Non-relevant (mean={np.mean(nrel_scores_A):.3f})")
    ax3.hist(nrel_scores_B, bins=80, alpha=0.3, color="coral",     density=True,
             linestyle="--", histtype="step", linewidth=1.5,
             label=f"B Non-relevant (mean={np.mean(nrel_scores_B):.3f})")
    ax3.axvline(np.mean(rel_scores_A),  color="steelblue", linestyle="-",  linewidth=2)
    ax3.axvline(np.mean(rel_scores_B),  color="coral",     linestyle="-",  linewidth=2)
    ax3.axvline(np.mean(nrel_scores_A), color="steelblue", linestyle="--", linewidth=1.5)
    ax3.axvline(np.mean(nrel_scores_B), color="coral",     linestyle="--", linewidth=1.5)
    ax3.set_xlabel("Similarity Score")
    ax3.set_ylabel("Density")
    ax3.set_title("Score Distribution: A vs B\nRelevant (filled) vs Non-relevant (outline)")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # ax4는 비워두고 gap 비교로 활용
    ax4.set_visible(False)

    # ─────────────────────────────────────────────────────────────
    # Plot 5: 쿼리별 nDCG@10 scatter (A vs B)
    # ─────────────────────────────────────────────────────────────
    log("Computing per-query nDCG@10...")
    ax5 = fig.add_subplot(4, 2, 5)

    ndcg_A_list, ndcg_B_list, q_labels = [], [], []
    for q_i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel_pids = set(qrels[qid].keys())
        for scores, ndcg_list in [(scores_A[q_i], ndcg_A_list),
                                   (scores_B[q_i], ndcg_B_list)]:
            ranked_pids = [corpus_ids[i] for i in np.argsort(-scores)]
            ideal_rels  = sorted([qrels[qid].get(p, 0) for p in rel_pids], reverse=True)
            dcg = sum(qrels[qid].get(ranked_pids[r], 0) / np.log2(r + 2)
                      for r in range(min(10, len(ranked_pids))))
            idcg = sum(ideal_rels[r] / np.log2(r + 2)
                       for r in range(min(10, len(ideal_rels))))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
        q_labels.append(qid)

    ndcg_A_arr = np.array(ndcg_A_list)
    ndcg_B_arr = np.array(ndcg_B_list)
    colors = ["red" if b < a else "green" for a, b in zip(ndcg_A_arr, ndcg_B_arr)]

    sc = ax5.scatter(ndcg_A_arr, ndcg_B_arr, c=colors, s=60, alpha=0.8, zorder=3)
    lim = [0, max(ndcg_A_arr.max(), ndcg_B_arr.max()) + 0.05]
    ax5.plot(lim, lim, "k--", linewidth=1, label="y=x (equal)")
    ax5.set_xlabel("nDCG@10 (A, 768-dim)")
    ax5.set_ylabel("nDCG@10 (B, 128-dim)")
    ax5.set_title("Per-query nDCG@10: A vs B\n(red=B worse, green=B better)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    # 쿼리 ID 라벨
    for i, (x, y, qid) in enumerate(zip(ndcg_A_arr, ndcg_B_arr, q_labels)):
        ax5.annotate(qid, (x, y), fontsize=6, alpha=0.7,
                     xytext=(3, 3), textcoords="offset points")

    # ─────────────────────────────────────────────────────────────
    # Plot 6: Relevant 문서 rank 변화 (A → B)
    # ─────────────────────────────────────────────────────────────
    log("Computing rank changes...")
    ax6 = fig.add_subplot(4, 2, 6)

    rank_A_all, rank_B_all = [], []
    for q_i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel_pids = set(qrels[qid].keys())
        ranked_A = [corpus_ids[i] for i in np.argsort(-scores_A[q_i])]
        ranked_B = [corpus_ids[i] for i in np.argsort(-scores_B[q_i])]
        rank_map_A = {p: r+1 for r, p in enumerate(ranked_A)}
        rank_map_B = {p: r+1 for r, p in enumerate(ranked_B)}
        for pid in rel_pids:
            if pid in rank_map_A and pid in rank_map_B:
                rank_A_all.append(rank_map_A[pid])
                rank_B_all.append(rank_map_B[pid])

    rank_A_arr = np.array(rank_A_all)
    rank_B_arr = np.array(rank_B_all)
    rank_diff  = rank_B_arr - rank_A_arr  # 양수 = B에서 순위 하락

    ax6.hist(rank_diff, bins=50, color="steelblue", alpha=0.8, edgecolor="none")
    ax6.axvline(0, color="black", linewidth=1)
    ax6.axvline(np.mean(rank_diff), color="red", linestyle="--",
                label=f"mean={np.mean(rank_diff):+.1f}")
    ax6.set_xlabel("Rank Change (B rank - A rank)\nPositive = rank dropped in B")
    ax6.set_ylabel("Count (relevant docs)")
    ax6.set_title("Relevant Document Rank Change\nA → B")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    # ─────────────────────────────────────────────────────────────
    # Plot 7: Score gap (discriminability)
    # ─────────────────────────────────────────────────────────────
    log("Computing score gaps...")
    ax7 = fig.add_subplot(4, 2, 7)

    gap_A, gap_B = [], []
    for q_i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel_pids = set(qrels[qid].keys())
        rel_idx  = [p_i for p_i, pid in enumerate(corpus_ids) if pid in rel_pids]
        nrel_idx = [p_i for p_i, pid in enumerate(corpus_ids) if pid not in rel_pids]
        if not rel_idx:
            continue
        mean_rel_A  = scores_A[q_i, rel_idx].mean()
        mean_nrel_A = scores_A[q_i, nrel_idx].mean()
        mean_rel_B  = scores_B[q_i, rel_idx].mean()
        mean_nrel_B = scores_B[q_i, nrel_idx].mean()
        gap_A.append(mean_rel_A - mean_nrel_A)
        gap_B.append(mean_rel_B - mean_nrel_B)

    x = np.arange(len(gap_A))
    qids_with_rel = [qid for qid in query_ids if qid in qrels]
    ax7.bar(x - 0.2, gap_A, 0.4, label="A (768-dim)", color="steelblue", alpha=0.8)
    ax7.bar(x + 0.2, gap_B, 0.4, label="B (128-dim)", color="coral",     alpha=0.8)
    ax7.axhline(np.mean(gap_A), color="steelblue", linestyle="--",
                label=f"A mean={np.mean(gap_A):.3f}")
    ax7.axhline(np.mean(gap_B), color="coral", linestyle="--",
                label=f"B mean={np.mean(gap_B):.3f}")
    ax7.set_xticks(x)
    ax7.set_xticklabels(qids_with_rel, rotation=90, fontsize=6)
    ax7.set_ylabel("Score Gap (Relevant - Non-relevant)")
    ax7.set_title("Discriminability: Score Gap per Query\nHigher = better separation")
    ax7.legend(fontsize=8)
    ax7.grid(axis="y", alpha=0.3)

    # ─────────────────────────────────────────────────────────────
    # Plot 8: Score gap 분포 (히스토그램)
    # ─────────────────────────────────────────────────────────────
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.hist(gap_A, bins=15, alpha=0.7, color="steelblue",
             label=f"A mean={np.mean(gap_A):.3f}")
    ax8.hist(gap_B, bins=15, alpha=0.7, color="coral",
             label=f"B mean={np.mean(gap_B):.3f}")
    ax8.axvline(np.mean(gap_A), color="steelblue", linestyle="--", linewidth=2)
    ax8.axvline(np.mean(gap_B), color="coral",     linestyle="--", linewidth=2)
    ax8.set_xlabel("Score Gap (Relevant - Non-relevant)")
    ax8.set_ylabel("Count (queries)")
    ax8.set_title("Score Gap Distribution\nA vs B")
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(COLBERT_DIR, "phase2_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log(f"Plot saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Phase 2 Visualization")
    log("=" * 60)

    log("Loading data...")
    corpus_ids, corpus_texts = load_corpus(CORPUS_PATH)
    query_ids,  query_texts  = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    log("\n[A] Encoding (msmarco-bert-base-dot-v5, 768-dim)...")
    A_corpus = encode_A(corpus_texts, max_length=220)
    A_query  = encode_A(query_texts,  max_length=64)

    log("\n[B] Encoding (colbertv2.0, 128-dim)...")
    B_corpus = encode_B(corpus_texts, max_length=220)
    B_query  = encode_B(query_texts,  max_length=32)

    log("\nPlotting all visualizations...")
    plot_all(corpus_ids, query_ids, query_texts, qrels,
             A_corpus, B_corpus, A_query, B_query)

    log("\nAll done.")


if __name__ == "__main__":
    main()
