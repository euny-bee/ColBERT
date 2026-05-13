#!/usr/bin/env python3
"""
compute_strict_recall.py
------------------------
R@k "strict" 버전: relevant passage 전부가 top-k에 포함되어야 1.0

기존(any):  쿼리당 relevant 중 하나라도 top-k에 있으면 1.0
변경(all):  쿼리당 relevant 전부가 top-k에 있어야 1.0

TSV 포맷 (ranking): qid \t pid \t rank \t score
Qrels 포맷:
  4컬럼 (TREC, MS MARCO): qid \t 0 \t pid \t rel
  3컬럼 (HotpotQA):       qid \t pid \t rel
"""

import os
import json
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_qrels(qrels_path):
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:          # TREC: qid 0 pid rel
                qid, pid = parts[0], int(parts[2])
            elif len(parts) >= 3:        # 3-col: qid pid rel
                qid, pid = parts[0], int(parts[1])
            else:
                continue
            qrels.setdefault(qid, set()).add(pid)
    return qrels


def compute_metrics(ranking_path, qrels_path, cutoffs=(50, 1000)):
    qrels = load_qrels(qrels_path)

    # ranking → {qid: set of pids in top-k (max cutoff)}
    max_k = max(cutoffs)
    top_pids = {}  # qid -> {k -> set(pid)}
    with open(ranking_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid  = str(parts[0])
            pid  = int(parts[1])
            rank = int(parts[2])
            if rank > max_k:
                continue
            if qid not in top_pids:
                top_pids[qid] = {k: set() for k in cutoffs}
            for k in cutoffs:
                if rank <= k:
                    top_pids[qid][k].add(pid)

    # 지표 계산
    counts_any = {k: 0 for k in cutoffs}  # 기존: any
    counts_all = {k: 0 for k in cutoffs}  # 신규: all
    n = 0

    for qid, relevant in qrels.items():
        if qid not in top_pids:
            continue
        n += 1
        for k in cutoffs:
            found = top_pids[qid][k]
            if any(p in found for p in relevant):
                counts_any[k] += 1
            if all(p in found for p in relevant):
                counts_all[k] += 1

    result = {"n_queries": n}
    for k in cutoffs:
        result[f"R@{k}_any"] = round(counts_any[k] / n, 4) if n else 0.0
        result[f"R@{k}_all"] = round(counts_all[k] / n, 4) if n else 0.0
    return result


def main():
    COLBERT_DIR = os.path.expanduser("~/ColBERT")

    tasks = [
        {
            "label":   "MS MARCO 1.1M - float16",
            "ranking": os.path.join(COLBERT_DIR, "experiments/msmarco_1m/rankings/1m.analog.tsv"),
            "qrels":   os.path.join(COLBERT_DIR, "data/msmarco/qrels.dev.small.fair.tsv"),
        },
        {
            "label":   "MS MARCO 1.1M - 2-bit",
            "ranking": os.path.join(COLBERT_DIR, "experiments/msmarco_1m/rankings/1m.2bit.tsv"),
            "qrels":   os.path.join(COLBERT_DIR, "data/msmarco/qrels.dev.small.fair.tsv"),
        },
        {
            "label":   "HotpotQA 2.6M - float16",
            "ranking": os.path.join(COLBERT_DIR, "experiments/hotpotqa_2.6m/rankings/hotpotqa.analog.tsv"),
            "qrels":   "D:/beir/hotpotqa/qrels.test.fair.int.tsv",
        },
        {
            "label":   "HotpotQA 2.6M - 2-bit",
            "ranking": os.path.join(COLBERT_DIR, "experiments/hotpotqa_2.6m/rankings/hotpotqa.2bit.tsv"),
            "qrels":   "D:/beir/hotpotqa/qrels.test.fair.int.tsv",
        },
    ]

    all_results = {}
    for task in tasks:
        log(f"Computing: {task['label']}")
        if not os.path.exists(task["ranking"]):
            log(f"  SKIP: {task['ranking']}")
            continue
        m = compute_metrics(task["ranking"], task["qrels"])
        all_results[task["label"]] = m
        log(f"  R@50  any={m['R@50_any']}  all={m['R@50_all']}")
        log(f"  R@1k  any={m['R@1000_any']}  all={m['R@1000_all']}  (n={m['n_queries']})")

    out_path = os.path.join(COLBERT_DIR, "experiments/strict_recall_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved → {out_path}")

    # 비교 테이블
    print(f"\n{'='*75}")
    print(f"  Strict Recall (all relevant must appear in top-k)")
    print(f"{'='*75}")
    print(f"  {'Dataset':<35} {'R@50_any':>9} {'R@50_all':>9} {'R@1k_any':>9} {'R@1k_all':>9}")
    print(f"  {'-'*70}")

    pairs = [
        ("MS MARCO 1.1M",  "MS MARCO 1.1M - float16",  "MS MARCO 1.1M - 2-bit"),
        ("HotpotQA 2.6M",  "HotpotQA 2.6M - float16",  "HotpotQA 2.6M - 2-bit"),
    ]
    for dataset, k_f16, k_b2 in pairs:
        for label, key in [(f"{dataset} float16", k_f16), (f"{dataset} 2-bit", k_b2)]:
            m = all_results.get(key, {})
            if m:
                print(f"  {label:<35} {m['R@50_any']:>9.4f} {m['R@50_all']:>9.4f} "
                      f"{m['R@1000_any']:>9.4f} {m['R@1000_all']:>9.4f}")
        f16 = all_results.get(k_f16, {})
        b2  = all_results.get(k_b2,  {})
        for suffix in ["R@50_all", "R@1000_all"]:
            if f16 and b2:
                delta = f16[suffix] - b2[suffix]
                sign = "+" if delta >= 0 else ""
                print(f"  {'  delta '+suffix:<35} {sign+f'{delta:.4f}':>9}")
        print()

    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
