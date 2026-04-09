#!/usr/bin/env python3
"""
make_collection_fair.py
-----------------------
qrels의 모든 relevant passage를 포함하는 1.1M 컬렉션 생성.

출력: D:/msmarco/collection_1m_fair.tsv
"""

import os
import random
import time

FULL_COLLECTION = "D:/msmarco/full/collection.tsv"
QRELS_PATH      = r"C:/Users/nmdl-khb/ColBERT/data/msmarco/qrels.dev.small.tsv"
OUTPUT_PATH     = "D:/msmarco/collection_1m_fair.tsv"
TARGET_SIZE     = 1_100_000
SEED            = 42

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    random.seed(SEED)

    # 1. qrels에서 relevant PID 수집
    log("Loading qrels...")
    relevant_pids = set()
    with open(QRELS_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            relevant_pids.add(int(parts[2]))
    log(f"  Relevant PIDs: {len(relevant_pids)}")

    # 2. 전체 컬렉션 스캔: relevant 라인 저장 + 나머지 PID 목록 수집
    log("Scanning full collection...")
    relevant_lines = {}   # pid -> line
    other_pids = []       # relevant 아닌 PID 목록

    with open(FULL_COLLECTION, encoding="utf-8") as f:
        for i, line in enumerate(f):
            pid = int(line.split("\t", 1)[0])
            if pid in relevant_pids:
                relevant_lines[pid] = line
            else:
                other_pids.append(pid)
            if (i + 1) % 1_000_000 == 0:
                log(f"  Scanned {i+1:,} lines...")

    log(f"  Relevant lines found: {len(relevant_lines)} / {len(relevant_pids)}")
    log(f"  Non-relevant PIDs: {len(other_pids):,}")

    # 3. 나머지 랜덤 샘플링
    n_sample = TARGET_SIZE - len(relevant_lines)
    log(f"Sampling {n_sample:,} non-relevant passages...")
    sampled_pids = set(random.sample(other_pids, n_sample))

    # 4. 두 번째 패스: 선택된 라인만 출력
    log("Writing output collection (sequential PIDs 0..N-1)...")
    # ColBERT requires pid == line_idx, so we renumber all PIDs sequentially.
    # We save orig_pid -> new_pid mapping to remap qrels for evaluation.

    orig_to_new = {}   # original PID -> new sequential PID
    total_written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        # relevant passages first
        for orig_pid in sorted(relevant_lines):
            text = relevant_lines[orig_pid].split("\t", 1)[1]
            new_pid = total_written
            orig_to_new[orig_pid] = new_pid
            out.write(f"{new_pid}\t{text}")
            total_written += 1

        # sampled non-relevant passages (second pass)
        with open(FULL_COLLECTION, encoding="utf-8") as f:
            for i, line in enumerate(f):
                orig_pid = int(line.split("\t", 1)[0])
                if orig_pid in sampled_pids:
                    text = line.split("\t", 1)[1]
                    new_pid = total_written
                    orig_to_new[orig_pid] = new_pid
                    out.write(f"{new_pid}\t{text}")
                    total_written += 1
                if (i + 1) % 1_000_000 == 0:
                    log(f"  Pass2: {i+1:,} lines scanned, {total_written:,} written...")

    log(f"Done! Total written: {total_written:,}")
    log(f"Output: {OUTPUT_PATH}")

    # 4. qrels를 새 PID로 리매핑하여 저장
    QRELS_FAIR = r"C:/Users/nmdl-khb/ColBERT/data/msmarco/qrels.dev.small.fair.tsv"
    log(f"Writing remapped qrels → {QRELS_FAIR}")
    mapped = 0
    with open(QRELS_PATH) as fin, open(QRELS_FAIR, "w") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            orig_pid = int(parts[2])
            if orig_pid in orig_to_new:
                parts[2] = str(orig_to_new[orig_pid])
                fout.write("\t".join(parts) + "\n")
                mapped += 1
    log(f"  Remapped qrels: {mapped} / {len(relevant_pids)} relevant entries written")

    # 검증
    log(f"Relevant coverage: {len(orig_to_new & relevant_pids) if hasattr(orig_to_new,'__and__') else len(set(orig_to_new) & relevant_pids)}/{len(relevant_pids)}")


if __name__ == "__main__":
    main()
