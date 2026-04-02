#!/usr/bin/env python3
"""
run_covid_analog.py
--------------------
TREC-COVID float16 (analog) 인덱스 빌드 + 분석 파이프라인.

Step 1: 재인덱싱 (raw embeddings 저장)
Step 2: analog index 빌드 (float16 residuals)
Step 3: analog 검색 (k=100)
Step 4: 분석 — Top-50 Overlap, MRR@k (k=1..10), Score Margin

Usage:
  cd ~/ColBERT && source ~/colbert-env/bin/activate
  nohup python run_covid_analog.py > covid_analog.log 2>&1 &
"""

import os, sys, json, time, subprocess
import numpy as np

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
DATA_DIR     = os.path.join(COLBERT_DIR, "data/table5/beir/trec-covid")
RESULTS_DIR  = os.path.join(COLBERT_DIR, "experiments/table5")
RANKING_DIR  = os.path.join(RESULTS_DIR, "rankings")
RUN_TABLE5   = os.path.join(COLBERT_DIR, "run_table5.py")

COLLECTION   = os.path.join(DATA_DIR, "collection.tsv")
QUERIES      = os.path.join(DATA_DIR, "queries.tsv")
QRELS        = os.path.join(DATA_DIR, "qrels.tsv")

Q_INDEX      = "beir.trec-covid.quantized"
A_INDEX      = "beir.trec-covid.analog"
Q_RANKING    = os.path.join(RANKING_DIR, "trec-covid.quantized.tsv")
A_RANKING    = os.path.join(RANKING_DIR, "trec-covid.analog.tsv")

SEARCH_K     = 100

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_step(desc, args):
    log(f">>> {desc}")
    t0 = time.time()
    rc = subprocess.run([sys.executable, RUN_TABLE5] + args, cwd=COLBERT_DIR).returncode
    elapsed = time.time() - t0
    if rc != 0:
        log(f"    FAILED (rc={rc}) after {elapsed:.0f}s")
        sys.exit(1)
    log(f"    Done in {elapsed/60:.1f} min")


# ── qrels 로드 (graded: 0/1/2, rel>=1 을 relevant로 처리) ────
def load_qrels(path, min_rel=1):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            qid, pid, label = parts[0], parts[2], int(parts[3])
            if label >= min_rel:
                qrels.setdefault(qid, set()).add(pid)
    return qrels


# ── ranking 로드: {qid -> [(pid, score), ...]} ───────────────
def load_ranking(path):
    rankings = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            qid, pid, rank, score = parts[0], parts[1], int(parts[2]), float(parts[3])
            rankings.setdefault(qid, []).append((rank, pid, score))
    # 각 쿼리를 rank 순 정렬
    for qid in rankings:
        rankings[qid].sort(key=lambda x: x[0])
    return rankings


# ── 분석 함수들 ───────────────────────────────────────────────
def top_k_overlap(rank_a, rank_b, k=50):
    """두 랭킹의 top-k 결과 Jaccard overlap."""
    overlaps = []
    for qid in rank_a:
        if qid not in rank_b:
            continue
        top_a = set(pid for _, pid, _ in rank_a[qid][:k])
        top_b = set(pid for _, pid, _ in rank_b[qid][:k])
        if not top_a or not top_b:
            continue
        overlap = len(top_a & top_b) / k
        overlaps.append(overlap)
    return overlaps


def mrr_at_k(rankings, qrels, ks):
    scores = {k: [] for k in ks}
    for qid, results in rankings.items():
        rel = qrels.get(qid, set())
        if not rel:
            continue
        # 첫 번째 관련 문서 rank 찾기
        first_rel_rank = None
        for rank, pid, _ in results:
            if pid in rel:
                first_rel_rank = rank
                break
        for k in ks:
            if first_rel_rank is not None and first_rel_rank <= k:
                scores[k].append(1.0 / first_rel_rank)
            else:
                scores[k].append(0.0)
    return {k: np.mean(v) for k, v in scores.items() if v}


def get_margin(results, qrel_set):
    """Score margin = score(첫 관련문서) - score(top 비관련문서)"""
    rel_score    = next((score for _, pid, score in results if pid in qrel_set),     None)
    nonrel_score = next((score for _, pid, score in results if pid not in qrel_set), None)
    if rel_score is None or nonrel_score is None:
        return None
    return rel_score - nonrel_score


def score_margins(rankings, qrels):
    margins = {}
    for qid, results in rankings.items():
        rel = qrels.get(qid, set())
        if not rel:
            continue
        m = get_margin(results, rel)
        if m is not None:
            margins[qid] = m
    return margins


# ── 결과 출력 ─────────────────────────────────────────────────
def print_analysis(rank_f16, rank_2bit, qrels):
    ks = list(range(1, 11))

    # Top-50 Overlap
    overlaps = top_k_overlap(rank_f16, rank_2bit, k=50)
    log(f"\n  Top-50 Overlap: mean={np.mean(overlaps):.4f}  "
        f"min={np.min(overlaps):.4f}  max={np.max(overlaps):.4f}  "
        f"(n={len(overlaps)} queries)")

    # MRR@k
    mrr_f16  = mrr_at_k(rank_f16,  qrels, ks)
    mrr_2bit = mrr_at_k(rank_2bit, qrels, ks)

    log("\n  MRR@k:")
    log(f"  {'k':>3}  {'float16':>10}  {'2-bit':>10}  {'delta':>10}")
    for k in ks:
        d = mrr_f16[k] - mrr_2bit[k]
        log(f"  {k:>3}  {mrr_f16[k]:>10.6f}  {mrr_2bit[k]:>10.6f}  {d:>+10.6f}")

    # Score Margin
    mg_f16  = score_margins(rank_f16,  qrels)
    mg_2bit = score_margins(rank_2bit, qrels)

    common = sorted(set(mg_f16) & set(mg_2bit))
    f16_vals  = np.array([mg_f16[q]  for q in common])
    bit2_vals = np.array([mg_2bit[q] for q in common])

    n_f16_better  = (f16_vals > bit2_vals).sum()
    n_fail_f16    = (f16_vals  < 0).sum()
    n_fail_2bit   = (bit2_vals < 0).sum()

    log(f"\n  Score Margin ({len(common)} queries):")
    log(f"    float16  mean={f16_vals.mean():+.4f}  "
        f"failures={n_fail_f16}/{len(f16_vals)} ({n_fail_f16/len(f16_vals):.1%})")
    log(f"    2-bit    mean={bit2_vals.mean():+.4f}  "
        f"failures={n_fail_2bit}/{len(bit2_vals)} ({n_fail_2bit/len(bit2_vals):.1%})")
    log(f"    float16 > 2-bit: {n_f16_better}/{len(common)} ({n_f16_better/len(common):.1%})")

    # JSON 저장
    out = {
        "top50_overlap": {"mean": float(np.mean(overlaps)), "values": [float(v) for v in overlaps]},
        "mrr_float16":   {str(k): float(v) for k, v in mrr_f16.items()},
        "mrr_2bit":      {str(k): float(v) for k, v in mrr_2bit.items()},
        "score_margin_float16": {q: float(v) for q, v in mg_f16.items()},
        "score_margin_2bit":    {q: float(v) for q, v in mg_2bit.items()},
    }
    out_path = os.path.join(RESULTS_DIR, "trec_covid_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("TREC-COVID: float16 index 빌드 + 분석")
    log("=" * 60)

    # Step 1: 재인덱싱 (raw embeddings 저장 포함)
    if not os.path.exists(os.path.join(DATA_DIR, "raw_embs")):
        run_step("Step 1: Re-indexing TREC-COVID (raw embeddings 저장)",
                 ["index", "trec-covid", COLLECTION, Q_INDEX])
    else:
        log("Step 1: raw_embs 이미 존재, skip")

    # Step 2: analog index 빌드
    a_index_dir = os.path.join(RESULTS_DIR, "indexes", A_INDEX)
    if not os.path.exists(a_index_dir):
        run_step("Step 2: Analog index 빌드 (float16 residuals)",
                 ["build_analog", "trec-covid", Q_INDEX, A_INDEX])
    else:
        log("Step 2: Analog index 이미 존재, skip")

    # Step 3: analog 검색
    if not os.path.exists(A_RANKING):
        run_step("Step 3: Analog 검색 (float16, k=100)",
                 ["search_analog", A_INDEX, QUERIES, A_RANKING])
    else:
        log("Step 3: Analog ranking 이미 존재, skip")

    # Step 3b: 2-bit 검색 (기존 ranking 없으면 재검색)
    if not os.path.exists(Q_RANKING):
        run_step("Step 3b: 2-bit 검색 (k=100)",
                 ["search", Q_INDEX, QUERIES, Q_RANKING])
    else:
        log("Step 3b: 2-bit ranking 이미 존재, skip")

    # Step 4: 분석
    log("\nStep 4: 분석 시작")
    qrels    = load_qrels(QRELS, min_rel=1)
    rank_f16  = load_ranking(A_RANKING)
    rank_2bit = load_ranking(Q_RANKING)

    log(f"  쿼리 수: float16={len(rank_f16)}, 2-bit={len(rank_2bit)}, qrels={len(qrels)}")
    print_analysis(rank_f16, rank_2bit, qrels)

    log("\n완료!")


if __name__ == "__main__":
    main()
