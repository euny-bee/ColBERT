#!/usr/bin/env python3
import os, json
import numpy as np
import csv

COLBERT_DIR = os.path.expanduser("~/ColBERT")
RESULTS_DIR = os.path.join(COLBERT_DIR, "experiments/msmarco")
QRELS_FILE  = os.path.join(COLBERT_DIR, "data/msmarco/subset/qrels.tsv")

with open(os.path.join(RESULTS_DIR, "scores_analog_k1000_500q.json")) as f: res_f16  = json.load(f)
with open(os.path.join(RESULTS_DIR, "scores_2bit_k1000_500q.json"))   as f: res_2bit = json.load(f)

qrels = {}
with open(QRELS_FILE) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 4:
            parts = line.strip().split()
        qid, pid, rel = parts[0], parts[2], parts[3]
        if int(rel) > 0:
            qrels.setdefault(qid, set()).add(pid)

ks = list(range(1, 11))

def recall_at_k(results, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, pid_scores in results.items():
        rel = qrels.get(qid, set())
        if not rel: continue
        ranked = sorted(pid_scores.keys(), key=lambda p: -pid_scores[p])
        for k in ks:
            top_k = set(ranked[:k])
            scores[k].append(len(rel & top_k) / len(rel))
    return {k: np.mean(v) for k, v in scores.items() if v}

def mrr_at_k(results, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, pid_scores in results.items():
        rel = qrels.get(qid, set())
        if not rel: continue
        ranked = sorted(pid_scores.keys(), key=lambda p: -pid_scores[p])
        rr = 0.0
        for rank, pid in enumerate(ranked, 1):
            if pid in rel:
                rr = 1.0 / rank
                break
        for k in ks:
            scores[k].append(rr if (rr == 0 or 1/rr <= k) else 0.0)
    return {k: np.mean(v) for k, v in scores.items() if v}

def ndcg_at_k(results, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, pid_scores in results.items():
        rel = qrels.get(qid, set())
        if not rel: continue
        ranked = sorted(pid_scores.keys(), key=lambda p: -pid_scores[p])
        for k in ks:
            dcg  = sum(1.0/np.log2(r+1) for r,p in enumerate(ranked[:k],1) if p in rel)
            idcg = sum(1.0/np.log2(r+1) for r in range(1, min(k,len(rel))+1))
            scores[k].append(dcg/idcg if idcg > 0 else 0.0)
    return {k: np.mean(v) for k, v in scores.items() if v}

r_f16    = recall_at_k(res_f16,  qrels, ks)
r_2bit   = recall_at_k(res_2bit, qrels, ks)
mrr_f16  = mrr_at_k(res_f16,  qrels, ks)
mrr_2bit = mrr_at_k(res_2bit, qrels, ks)
ndcg_f16  = ndcg_at_k(res_f16,  qrels, ks)
ndcg_2bit = ndcg_at_k(res_2bit, qrels, ks)

out_path = os.path.join(RESULTS_DIR, "recall_mrr_ndcg_k10.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k",
                     "Recall@k_float16", "Recall@k_2bit", "Recall@k_delta",
                     "MRR@k_float16",    "MRR@k_2bit",    "MRR@k_delta",
                     "nDCG@k_float16",   "nDCG@k_2bit",   "nDCG@k_delta"])
    for k in ks:
        writer.writerow([
            k,
            round(r_f16[k], 6),    round(r_2bit[k], 6),   round(r_f16[k]-r_2bit[k], 6),
            round(mrr_f16[k], 6),  round(mrr_2bit[k], 6), round(mrr_f16[k]-mrr_2bit[k], 6),
            round(ndcg_f16[k], 6), round(ndcg_2bit[k], 6),round(ndcg_f16[k]-ndcg_2bit[k], 6),
        ])

print("Saved ->", out_path)

# console table
header = ("k", "R_f16", "R_2bit", "R_delta", "MRR_f16", "MRR_2bit", "MRR_delta", "nDCG_f16", "nDCG_2bit", "nDCG_delta")
print("\n" + "  ".join(f"{h:>11}" for h in header))
for k in ks:
    rd  = r_f16[k]   - r_2bit[k]
    md  = mrr_f16[k] - mrr_2bit[k]
    nd  = ndcg_f16[k]- ndcg_2bit[k]
    row = [str(k),
           f"{r_f16[k]:.6f}",    f"{r_2bit[k]:.6f}",   f"{rd:+.6f}",
           f"{mrr_f16[k]:.6f}",  f"{mrr_2bit[k]:.6f}", f"{md:+.6f}",
           f"{ndcg_f16[k]:.6f}", f"{ndcg_2bit[k]:.6f}",f"{nd:+.6f}"]
    print("  ".join(f"{v:>11}" for v in row))
