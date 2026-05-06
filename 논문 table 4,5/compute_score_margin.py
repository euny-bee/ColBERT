#!/usr/bin/env python3
"""
compute_score_margin.py
-----------------------
기존 ranking TSV 파일에서 Score Margin을 계산.

Score Margin per query:
  margin = max_score(relevant docs) - score(top non-relevant doc)
  - 양수: relevant > 1등 비관련 → 정상
  - 음수: 비관련이 관련보다 위 → 검색 실패

TSV 포맷: qid \t pid \t rank \t score
Qrels 포맷: qid \t pid \t score (1이면 relevant)
"""

import sys
import json
import math


def compute_score_margin(ranking_path, qrels_path):
    # qrels 로드: {qid(str) -> set(pid)}
    # 3컬럼: qid\tpid\trel  (HotpotQA)
    # 4컬럼: qid\t0\tpid\trel  (MS MARCO TREC format)
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                qid = parts[0]
                pid = int(parts[2])
            elif len(parts) >= 3:
                qid = parts[0]
                pid = int(parts[1])
            else:
                continue
            qrels.setdefault(qid, set()).add(pid)

    # ranking 로드: {qid(str) -> [(rank, pid, score), ...]}
    rankings = {}
    with open(ranking_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            qid  = str(parts[0])
            pid  = int(parts[1])
            rank = int(parts[2])
            score = float(parts[3])
            rankings.setdefault(qid, []).append((rank, pid, score))

    margins = []
    n_positive = 0
    n_negative = 0
    n_no_relevant = 0
    n_no_nonrel = 0

    for qid, relevant in qrels.items():
        if qid not in rankings:
            continue

        ranked = sorted(rankings[qid])  # sort by rank

        rel_scores    = [score for (_, pid, score) in ranked if pid in relevant]
        nonrel_scores = [score for (_, pid, score) in ranked if pid not in relevant]

        if not rel_scores:
            n_no_relevant += 1
            continue
        if not nonrel_scores:
            n_no_nonrel += 1
            continue

        best_rel    = max(rel_scores)
        best_nonrel = max(nonrel_scores)
        margin = best_rel - best_nonrel

        margins.append(margin)
        if margin >= 0:
            n_positive += 1
        else:
            n_negative += 1

    n = len(margins)
    avg_margin = sum(margins) / n if n > 0 else 0.0

    return {
        "score_margin":     round(avg_margin, 4),
        "n_queries":        n,
        "n_positive":       n_positive,
        "n_negative":       n_negative,
        "positive_rate":    round(n_positive / n, 4) if n > 0 else 0.0,
        "n_no_relevant":    n_no_relevant,
        "n_no_nonrel":      n_no_nonrel,
    }


def main():
    import os, time

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    COLBERT_DIR = os.path.expanduser("~/ColBERT")

    tasks = [
        {
            "label":   "MS MARCO 1.1M - float16",
            "ranking": os.path.join(COLBERT_DIR, "experiments/msmarco_1m/rankings/1m.analog.tsv"),
            "qrels":   os.path.join(COLBERT_DIR, "data/msmarco/qrels.dev.small.fair.tsv"),
            "result_key": "msmarco_float16",
        },
        {
            "label":   "MS MARCO 1.1M - 2-bit",
            "ranking": os.path.join(COLBERT_DIR, "experiments/msmarco_1m/rankings/1m.2bit.tsv"),
            "qrels":   os.path.join(COLBERT_DIR, "data/msmarco/qrels.dev.small.fair.tsv"),
            "result_key": "msmarco_2bit",
        },
        {
            "label":   "HotpotQA 2.6M - float16",
            "ranking": os.path.join(COLBERT_DIR, "experiments/hotpotqa_2.6m/rankings/hotpotqa.analog.tsv"),
            "qrels":   "D:/beir/hotpotqa/qrels.test.fair.int.tsv",
            "result_key": "hotpotqa_float16",
        },
        {
            "label":   "HotpotQA 2.6M - 2-bit",
            "ranking": os.path.join(COLBERT_DIR, "experiments/hotpotqa_2.6m/rankings/hotpotqa.2bit.tsv"),
            "qrels":   "D:/beir/hotpotqa/qrels.test.fair.int.tsv",
            "result_key": "hotpotqa_2bit",
        },
    ]

    all_results = {}

    for task in tasks:
        log(f"Computing: {task['label']}")
        if not os.path.exists(task["ranking"]):
            log(f"  SKIP: ranking not found: {task['ranking']}")
            continue
        if not os.path.exists(task["qrels"]):
            log(f"  SKIP: qrels not found: {task['qrels']}")
            continue

        m = compute_score_margin(task["ranking"], task["qrels"])
        all_results[task["result_key"]] = m
        log(f"  Score Margin = {m['score_margin']:.4f}  "
            f"(positive={m['n_positive']}, negative={m['n_negative']}, "
            f"positive_rate={m['positive_rate']:.3f})")

    # 저장
    out_path = os.path.join(COLBERT_DIR, "experiments/score_margin_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved → {out_path}")

    # 비교 테이블 출력
    print(f"\n{'='*70}")
    print(f"  Score Margin 비교 (margin = max_score_rel - max_score_nonrel)")
    print(f"{'='*70}")
    print(f"  {'Dataset':<30} {'Margin':>10} {'Pos%':>8}")
    print(f"  {'-'*55}")

    pairs = [
        ("MS MARCO 1.1M",   "msmarco_float16",   "msmarco_2bit"),
        ("HotpotQA 2.6M",   "hotpotqa_float16",  "hotpotqa_2bit"),
    ]
    for dataset, k_f16, k_b2 in pairs:
        f16 = all_results.get(k_f16, {})
        b2  = all_results.get(k_b2,  {})
        if f16:
            print(f"  {dataset+' float16':<30} {f16['score_margin']:>10.4f} {f16['positive_rate']:>7.1%}")
        if b2:
            print(f"  {dataset+' 2-bit':<30} {b2['score_margin']:>10.4f} {b2['positive_rate']:>7.1%}")
        if f16 and b2:
            delta = f16['score_margin'] - b2['score_margin']
            sign = "+" if delta >= 0 else ""
            print(f"  {'  delta (f16 - 2bit)':<30} {sign+f'{delta:.4f}':>10}")
        print()

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
