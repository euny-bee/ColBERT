#!/usr/bin/env python3
"""
analyze_ko_datasets.py
----------------------
Mr.TyDi Korean / MIRACL Korean — float16 vs 2-bit 상세 분석

[Figure 1] R@k / MRR@k / nDCG@k  (k=1..10)
[Figure 2] Per-query Score Margin Scatter
[Figure 3] Top-50 Result Overlap per-query Histogram

실행:
  python analyze_ko_datasets.py
"""

import os
import math
import json
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLBERT_DIR = os.path.expanduser("~/ColBERT")
OUT_DIR = os.path.join(COLBERT_DIR, "experiments/plots")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    {
        "name":         "mrtydi_ko",
        "label":        "Mr.TyDi Korean",
        "analog_rank":  os.path.join(COLBERT_DIR, "experiments/beir_mrtydi_ko/rankings/mrtydi_ko.analog.tsv"),
        "bit2_rank":    os.path.join(COLBERT_DIR, "experiments/beir_mrtydi_ko/rankings/mrtydi_ko.2bit.tsv"),
        "qrels":        "D:/beir/mrtydi_ko/qrels.test.int.tsv",
        "graded":       False,
    },
    {
        "name":         "miracl_ko",
        "label":        "MIRACL Korean",
        "analog_rank":  os.path.join(COLBERT_DIR, "experiments/beir_miracl_ko/rankings/miracl_ko.analog.tsv"),
        "bit2_rank":    os.path.join(COLBERT_DIR, "experiments/beir_miracl_ko/rankings/miracl_ko.2bit.tsv"),
        "qrels":        "D:/beir/miracl_ko/qrels.test.int.tsv",
        "graded":       False,
    },
]


# ============================================================
# 파일 로더
# ============================================================
def load_qrels(path):
    """qrels: {qid(str) -> {pid(int) -> score(int)}}"""
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid, score = str(parts[0]), int(parts[1]), int(parts[2])
            qrels.setdefault(qid, {})[pid] = score
    return qrels


def load_ranking(path):
    """ranking: {qid(str) -> [(rank, pid, score), ...]}  sorted by rank"""
    rankings = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid  = str(parts[0])
            pid  = int(parts[1])
            rank = int(parts[2])
            score = float(parts[3]) if len(parts) >= 4 else 0.0
            rankings.setdefault(qid, []).append((rank, pid, score))
    for qid in rankings:
        rankings[qid].sort()
    return rankings


# ============================================================
# Figure 1: R@k / MRR@k / nDCG@k  (k=1..10)
# ============================================================
def compute_k_curves(rankings, qrels, graded=False, max_k=10):
    K = list(range(1, max_k + 1))
    r_at_k   = {k: [] for k in K}
    mrr_at_k = {k: [] for k in K}
    ndcg_at_k = {k: [] for k in K}

    for qid, rel_dict in qrels.items():
        if qid not in rankings:
            continue
        if not any(s >= 1 for s in rel_dict.values()):
            continue

        ranked_pids   = [pid   for _, pid, _     in rankings[qid]]
        ranked_scores = [score for _, _,   score in rankings[qid]]
        rel_set = {pid for pid, s in rel_dict.items() if s >= 1}

        for k in K:
            top_k = ranked_pids[:k]

            # R@k: fraction of relevant docs found in top-k
            r_at_k[k].append(1.0 if any(p in rel_set for p in top_k) else 0.0)

            # MRR@k
            mrr = 0.0
            for r, pid in enumerate(top_k, 1):
                if pid in rel_set:
                    mrr = 1.0 / r
                    break
            mrr_at_k[k].append(mrr)

            # nDCG@k
            dcg = 0.0
            for r, pid in enumerate(top_k, 1):
                gain = rel_dict.get(pid, 0) if graded else (1 if pid in rel_set else 0)
                if gain > 0:
                    dcg += gain / math.log2(r + 1)
            ideal_gains = sorted(rel_dict.values(), reverse=True)[:k] if graded else \
                          sorted([1 if pid in rel_set else 0 for pid in rel_dict], reverse=True)[:k]
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains) if g > 0)
            ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0.0)

    return (
        {k: np.mean(v) for k, v in r_at_k.items()},
        {k: np.mean(v) for k, v in mrr_at_k.items()},
        {k: np.mean(v) for k, v in ndcg_at_k.items()},
        len([q for q in qrels if q in rankings and any(s >= 1 for s in qrels[q].values())]),
    )


def plot_k_curves(analog_r, analog_mrr, analog_ndcg,
                  bit2_r,   bit2_mrr,   bit2_ndcg,
                  n_queries, label, out_path, max_k=10):
    K = list(range(1, max_k + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"float16 vs 2-bit  (k=1..{max_k})  |  {n_queries} queries, {label}",
                 fontsize=13, fontweight="bold")

    for ax, (title, f16, b2, scale) in zip(axes, [
        ("R@k",    analog_r,    bit2_r,    1),
        ("MRR@k",  analog_mrr,  bit2_mrr,  100),
        ("nDCG@k", analog_ndcg, bit2_ndcg, 100),
    ]):
        ax.plot(K, [f16[k]*scale for k in K], "o-",  color="#3a7abf", label="float16", linewidth=2)
        ax.plot(K, [b2[k]*scale  for k in K], "s--", color="#e07b39", label="2-bit",   linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("k")
        ax.set_xticks(K[::5] if max_k > 10 else K)
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ============================================================
# Figure 2: Per-query Score Margin Scatter
# ============================================================
def compute_score_margins(rankings, qrels):
    """Returns list of (f16_margin, b2_margin) per query, and failure counts."""
    margins = []
    f16_failures = 0
    b2_failures  = 0

    for qid, rel_dict in qrels.items():
        rel_set = {pid for pid, s in rel_dict.items() if s >= 1}
        if not rel_set:
            continue

        def margin_for(ranking):
            if qid not in ranking:
                return None
            ranked = ranking[qid]
            rel_scores    = [s for _, pid, s in ranked if pid in rel_set]
            nonrel_scores = [s for _, pid, s in ranked if pid not in rel_set]
            if not rel_scores or not nonrel_scores:
                return None
            return max(rel_scores) - max(nonrel_scores)

        return margins, f16_failures, b2_failures  # placeholder — filled below

    return margins, f16_failures, b2_failures


def compute_per_query_margins(analog_rankings, bit2_rankings, qrels):
    f16_margins = {}
    b2_margins  = {}
    f16_fail = 0
    b2_fail  = 0

    for qid, rel_dict in qrels.items():
        rel_set = {pid for pid, s in rel_dict.items() if s >= 1}
        if not rel_set:
            continue

        def _margin(ranking):
            if qid not in ranking:
                return None
            ranked = ranking[qid]
            rel_sc    = [s for _, pid, s in ranked if pid in rel_set]
            nonrel_sc = [s for _, pid, s in ranked if pid not in rel_set]
            if not rel_sc or not nonrel_sc:
                return None
            return max(rel_sc) - max(nonrel_sc)

        f16 = _margin(analog_rankings)
        b2  = _margin(bit2_rankings)

        if f16 is None:
            f16_fail += 1
        if b2 is None:
            b2_fail += 1

        if f16 is not None and b2 is not None:
            f16_margins[qid] = f16
            b2_margins[qid]  = b2

    return f16_margins, b2_margins, f16_fail, b2_fail


def plot_score_margin_scatter(f16_margins, b2_margins, f16_fail, b2_fail,
                               n_total, label, out_path):
    qids = list(f16_margins.keys())
    x = np.array([f16_margins[q] for q in qids])
    y = np.array([b2_margins[q]  for q in qids])

    f16_larger = x >= y
    colors = np.where(f16_larger, "#3a7abf", "#e07b39")
    n_f16 = f16_larger.sum()
    n_b2  = (~f16_larger).sum()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x[f16_larger],  y[f16_larger],  color="#3a7abf", alpha=0.7,
               label=f"float16 larger ({n_f16/len(qids):.1%})", zorder=3)
    ax.scatter(x[~f16_larger], y[~f16_larger], color="#e07b39", alpha=0.7,
               label=f"2-bit larger ({n_b2/len(qids):.1%})",   zorder=3)

    lim = max(abs(x).max(), abs(y).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5, label="y = x  (equal margin)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Score Margin (float16)")
    ax.set_ylabel("Score Margin (2-bit)")
    ax.set_title(f"Per-query Score Margin Scatter\n"
                 f"(blue=float16 larger, orange=2-bit larger)\n{label}  |  {len(qids)} queries",
                 fontsize=11, fontweight="bold")
    ax.text(0.02, 0.97,
            f"float16 failures: {f16_fail}/{n_total} ({f16_fail/n_total:.1%})\n"
            f"2-bit   failures: {b2_fail}/{n_total} ({b2_fail/n_total:.1%})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ============================================================
# Figure 3: Top-50 Result Overlap per-query
# ============================================================
def compute_top50_overlap(analog_rankings, bit2_rankings, qrels, top_k=50):
    overlaps = []
    for qid in qrels:
        if qid not in analog_rankings or qid not in bit2_rankings:
            continue
        f16_top = set(pid for _, pid, _ in analog_rankings[qid][:top_k])
        b2_top  = set(pid for _, pid, _ in bit2_rankings[qid][:top_k])
        overlap = len(f16_top & b2_top) / top_k
        overlaps.append(overlap)
    return overlaps


def plot_top50_overlap(overlaps, label, out_path, top_k=50):
    mean_overlap = np.mean(overlaps)
    pct = [o * 100 for o in overlaps]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, 101, 2)
    ax.hist(pct, bins=bins, color="#e07b39", alpha=0.85, edgecolor="white")

    # KDE 곡선
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(pct, bw_method=0.3)
    xs = np.linspace(0, 100, 300)
    ys = kde(xs)
    scale = len(pct) * 2  # bin width=2
    ax.plot(xs, ys * scale, color="#e07b39", linewidth=2)

    ax.axvline(mean_overlap * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_overlap:.1%}")
    ax.set_xlabel(f"Top-{top_k} overlap (%)")
    ax.set_ylabel("Query count")
    ax.set_title(f"Top-{top_k} Result Overlap per-query\n{label}  |  float16 vs 2-bit  |  {len(overlaps)} queries",
                 fontsize=11, fontweight="bold", color="#3a7abf")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ============================================================
# main
# ============================================================
def main():
    for ds in DATASETS:
        name  = ds["name"]
        label = ds["label"]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        qrels           = load_qrels(ds["qrels"])
        analog_rankings = load_ranking(ds["analog_rank"])
        bit2_rankings   = load_ranking(ds["bit2_rank"])
        n_total = len([q for q in qrels if any(s >= 1 for s in qrels[q].values())])
        print(f"  Queries with relevance: {n_total}")

        # --- Figure 1a: k-curves (k=1..10) ---
        print("  [1] Computing k-curves (k=1..10)...")
        f16_r, f16_mrr, f16_ndcg, n_q = compute_k_curves(analog_rankings, qrels, ds["graded"], max_k=10)
        b2_r,  b2_mrr,  b2_ndcg,  _   = compute_k_curves(bit2_rankings,   qrels, ds["graded"], max_k=10)
        plot_k_curves(f16_r, f16_mrr, f16_ndcg,
                      b2_r,  b2_mrr,  b2_ndcg,
                      n_q, label,
                      os.path.join(OUT_DIR, f"{name}_k_curve.png"), max_k=10)
        print(f"    k=10  float16: R={f16_r[10]:.4f}  MRR={f16_mrr[10]:.4f}  nDCG={f16_ndcg[10]:.4f}")
        print(f"    k=10  2-bit:   R={b2_r[10]:.4f}  MRR={b2_mrr[10]:.4f}  nDCG={b2_ndcg[10]:.4f}")

        # --- Figure 1b: MRR@k + nDCG@k (k=1..30) ---
        print("  [1b] Computing MRR@k + nDCG@k (k=1..30)...")
        f16_r30, f16_mrr30, f16_ndcg30, _ = compute_k_curves(analog_rankings, qrels, ds["graded"], max_k=30)
        b2_r30,  b2_mrr30,  b2_ndcg30,  _ = compute_k_curves(bit2_rankings,   qrels, ds["graded"], max_k=30)

        K30 = list(range(1, 31))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"float16 vs 2-bit  (k=1..30)  |  {n_q} queries, {label}",
                     fontsize=13, fontweight="bold")
        for ax, (title, f16, b2) in zip(axes, [
            ("MRR@k",  f16_mrr30, b2_mrr30),
            ("nDCG@k", f16_ndcg30, b2_ndcg30),
        ]):
            ax.plot(K30, [f16[k]*100 for k in K30], "o-",  color="#3a7abf", label="float16", linewidth=2, markersize=4)
            ax.plot(K30, [b2[k]*100  for k in K30], "s--", color="#e07b39", label="2-bit",   linewidth=2, markersize=4)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("k")
            ax.set_xticks(range(0, 31, 5))
            ax.legend()
            ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out30 = os.path.join(OUT_DIR, f"{name}_mrr_ndcg_k30.png")
        plt.savefig(out30, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out30}")
        print(f"    k=30  float16: MRR={f16_mrr30[30]:.4f}  nDCG={f16_ndcg30[30]:.4f}")
        print(f"    k=30  2-bit:   MRR={b2_mrr30[30]:.4f}  nDCG={b2_ndcg30[30]:.4f}")

        # --- Figure 1c: MRR@k + nDCG@k (k=1..50) ---
        print("  [1c] Computing MRR@k + nDCG@k (k=1..50)...")
        f16_r50, f16_mrr50, f16_ndcg50, _ = compute_k_curves(analog_rankings, qrels, ds["graded"], max_k=50)
        b2_r50,  b2_mrr50,  b2_ndcg50,  _ = compute_k_curves(bit2_rankings,   qrels, ds["graded"], max_k=50)
        K50 = list(range(1, 51))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"float16 vs 2-bit  (k=1..50)  |  {n_q} queries, {label}",
                     fontsize=13, fontweight="bold")
        for ax, (title, f16, b2) in zip(axes, [
            ("MRR@k",  f16_mrr50, b2_mrr50),
            ("nDCG@k", f16_ndcg50, b2_ndcg50),
        ]):
            ax.plot(K50, [f16[k]*100 for k in K50], "-",  color="#3a7abf", label="float16", linewidth=2)
            ax.plot(K50, [b2[k]*100  for k in K50], "--", color="#e07b39", label="2-bit",   linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("k")
            ax.set_xticks(range(0, 51, 10))
            ax.legend()
            ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out50 = os.path.join(OUT_DIR, f"{name}_mrr_ndcg_k50.png")
        plt.savefig(out50, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out50}")
        print(f"    k=50  float16: MRR={f16_mrr50[50]:.4f}  nDCG={f16_ndcg50[50]:.4f}")
        print(f"    k=50  2-bit:   MRR={b2_mrr50[50]:.4f}  nDCG={b2_ndcg50[50]:.4f}")

        # --- Figure 1d: MRR@k + nDCG@k (k=1..100) ---
        print("  [1d] Computing MRR@k + nDCG@k (k=1..100)...")
        f16_r100, f16_mrr100, f16_ndcg100, _ = compute_k_curves(analog_rankings, qrels, ds["graded"], max_k=100)
        b2_r100,  b2_mrr100,  b2_ndcg100,  _ = compute_k_curves(bit2_rankings,   qrels, ds["graded"], max_k=100)
        K100 = list(range(1, 101))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"float16 vs 2-bit  (k=1..100)  |  {n_q} queries, {label}",
                     fontsize=13, fontweight="bold")
        for ax, (title, f16, b2) in zip(axes, [
            ("MRR@k",  f16_mrr100, b2_mrr100),
            ("nDCG@k", f16_ndcg100, b2_ndcg100),
        ]):
            ax.plot(K100, [f16[k]*100 for k in K100], "-",  color="#3a7abf", label="float16", linewidth=2)
            ax.plot(K100, [b2[k]*100  for k in K100], "--", color="#e07b39", label="2-bit",   linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("k")
            ax.set_xticks(range(0, 101, 20))
            ax.legend()
            ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out100 = os.path.join(OUT_DIR, f"{name}_mrr_ndcg_k100.png")
        plt.savefig(out100, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out100}")
        print(f"    k=100 float16: MRR={f16_mrr100[100]:.4f}  nDCG={f16_ndcg100[100]:.4f}")
        print(f"    k=100 2-bit:   MRR={b2_mrr100[100]:.4f}  nDCG={b2_ndcg100[100]:.4f}")

        # --- Figure 1e: MRR@k + nDCG@k (k=1..200) ---
        print("  [1e] Computing MRR@k + nDCG@k (k=1..200)...")
        f16_r200, f16_mrr200, f16_ndcg200, _ = compute_k_curves(analog_rankings, qrels, ds["graded"], max_k=200)
        b2_r200,  b2_mrr200,  b2_ndcg200,  _ = compute_k_curves(bit2_rankings,   qrels, ds["graded"], max_k=200)
        K200 = list(range(1, 201))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"float16 vs 2-bit  (k=1..200)  |  {n_q} queries, {label}",
                     fontsize=13, fontweight="bold")
        for ax, (title, f16, b2) in zip(axes, [
            ("MRR@k",  f16_mrr200, b2_mrr200),
            ("nDCG@k", f16_ndcg200, b2_ndcg200),
        ]):
            ax.plot(K200, [f16[k]*100 for k in K200], "-",  color="#3a7abf", label="float16", linewidth=2)
            ax.plot(K200, [b2[k]*100  for k in K200], "--", color="#e07b39", label="2-bit",   linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("k")
            ax.set_xticks(range(0, 201, 50))
            ax.legend()
            ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out200 = os.path.join(OUT_DIR, f"{name}_mrr_ndcg_k200.png")
        plt.savefig(out200, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out200}")
        print(f"    k=200 float16: MRR={f16_mrr200[200]:.4f}  nDCG={f16_ndcg200[200]:.4f}")
        print(f"    k=200 2-bit:   MRR={b2_mrr200[200]:.4f}  nDCG={b2_ndcg200[200]:.4f}")

        # --- Figure 1f: MRR@k + nDCG@k (k=1..1000) ---
        print("  [1f] Computing MRR@k + nDCG@k (k=1..1000)...")
        f16_r1k, f16_mrr1k, f16_ndcg1k, _ = compute_k_curves(analog_rankings, qrels, ds["graded"], max_k=1000)
        b2_r1k,  b2_mrr1k,  b2_ndcg1k,  _ = compute_k_curves(bit2_rankings,   qrels, ds["graded"], max_k=1000)
        K1k = list(range(1, 1001))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"float16 vs 2-bit  (k=1..1000)  |  {n_q} queries, {label}",
                     fontsize=13, fontweight="bold")
        for ax, (title, f16, b2) in zip(axes, [
            ("MRR@k",  f16_mrr1k, b2_mrr1k),
            ("nDCG@k", f16_ndcg1k, b2_ndcg1k),
        ]):
            ax.plot(K1k, [f16[k]*100 for k in K1k], "-",  color="#3a7abf", label="float16", linewidth=2)
            ax.plot(K1k, [b2[k]*100  for k in K1k], "--", color="#e07b39", label="2-bit",   linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("k")
            ax.set_xticks(range(0, 1001, 200))
            ax.legend()
            ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out1k = os.path.join(OUT_DIR, f"{name}_mrr_ndcg_k1000.png")
        plt.savefig(out1k, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out1k}")
        print(f"    k=1000 float16: MRR={f16_mrr1k[1000]:.4f}  nDCG={f16_ndcg1k[1000]:.4f}")
        print(f"    k=1000 2-bit:   MRR={b2_mrr1k[1000]:.4f}  nDCG={b2_ndcg1k[1000]:.4f}")

        # --- Figure 2: Score Margin Scatter ---
        print("  [2] Computing score margin scatter...")
        f16_margins, b2_margins, f16_fail, b2_fail = \
            compute_per_query_margins(analog_rankings, bit2_rankings, qrels)
        plot_score_margin_scatter(f16_margins, b2_margins, f16_fail, b2_fail,
                                  n_total, label,
                                  os.path.join(OUT_DIR, f"{name}_score_margin.png"))

        # --- Figure 3: Top-50 Overlap ---
        print("  [3] Computing top-50 overlap...")
        overlaps = compute_top50_overlap(analog_rankings, bit2_rankings, qrels)
        plot_top50_overlap(overlaps, label,
                           os.path.join(OUT_DIR, f"{name}_top50_overlap.png"))
        print(f"    mean overlap = {np.mean(overlaps):.1%}")

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
