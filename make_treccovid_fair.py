#!/usr/bin/env python3
"""
make_treccovid_fair.py
----------------------
BEIR TREC-COVID corpus 전체를 ColBERT용 컬렉션으로 변환.
- corpus 전체 사용 (171K, 샘플링 불필요)
- string doc ID / query ID -> int PID / QID 리매핑
- graded qrels (0/1/2) 보존

출력:
  D:/beir/trec-covid/collection.tsv          # pid\ttext
  D:/beir/trec-covid/queries.test.int.tsv    # qid\ttext
  D:/beir/trec-covid/qrels.test.int.tsv      # qid\tpid\tscore (0/1/2)
  D:/beir/trec-covid/qid_mapping.tsv         # orig_qid\tint_qid
"""

import json
import os
import time

BASE_DIR      = "D:/beir/trec-covid"
CORPUS_PATH   = f"{BASE_DIR}/corpus.jsonl"
QUERIES_PATH  = f"{BASE_DIR}/queries.jsonl"
QRELS_PATH    = f"{BASE_DIR}/qrels/test.tsv"

OUTPUT_CORPUS   = f"{BASE_DIR}/collection.tsv"
OUTPUT_QUERIES  = f"{BASE_DIR}/queries.test.tsv"
OUTPUT_QRELS    = f"{BASE_DIR}/qrels.test.int.tsv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # 1. qrels 로드 (graded: 0/1/2)
    log("Loading qrels...")
    qrel_rows = []     # (orig_qid, orig_doc_id, score)
    relevant_doc_ids = set()
    test_query_ids   = set()

    with open(QRELS_PATH) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            orig_qid, orig_doc_id, score = parts[0], parts[1], parts[2]
            qrel_rows.append((orig_qid, orig_doc_id, score))
            if int(score) >= 1:
                relevant_doc_ids.add(orig_doc_id)
            test_query_ids.add(orig_qid)

    log(f"  Qrel rows: {len(qrel_rows)}")
    log(f"  Test queries: {len(test_query_ids)}")
    log(f"  Relevant docs (score>=1): {len(relevant_doc_ids)}")

    # 2. corpus 전체 변환 (string doc_id -> int pid, pid == line_idx)
    log("Converting corpus to collection.tsv...")
    orig_to_pid = {}  # orig doc_id(str) -> new int pid
    total_written = 0

    with open(CORPUS_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_CORPUS, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            doc = json.loads(line)
            orig_id = doc["_id"]
            title = doc.get("title", "").replace("\t", " ").replace("\n", " ")
            text  = doc.get("text",  "").replace("\t", " ").replace("\n", " ")
            combined = f"{title} {text}".strip() if title else text
            new_pid = total_written
            orig_to_pid[orig_id] = new_pid
            fout.write(f"{new_pid}\t{combined}\n")
            total_written += 1
            if (i + 1) % 50_000 == 0:
                log(f"  {i+1:,} lines written...")

    log(f"  Total passages: {total_written:,}")

    # 검증: pid == line_idx
    log("Verifying pid == line_idx...")
    with open(OUTPUT_CORPUS, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            pid = int(line.split("\t", 1)[0])
            assert pid == idx, f"pid={pid} != line_idx={idx}"
    log("  Verification passed!")

    # 3. queries 저장 (qid는 이미 정수 "1"~"50")
    log("Loading queries...")
    written_q = 0
    with open(QUERIES_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_QUERIES, "w", encoding="utf-8") as fout:
        for line in fin:
            q = json.loads(line)
            if q["_id"] not in test_query_ids:
                continue
            text = q["text"].replace("\t", " ").replace("\n", " ")
            fout.write(f"{q['_id']}\t{text}\n")
            written_q += 1

    log(f"  Queries written: {written_q}")

    # 4. qrels 리매핑 (doc_id -> pid, qid는 그대로, graded score 보존)
    log("Remapping qrels (doc_id -> pid)...")
    mapped = 0
    skipped = 0
    with open(OUTPUT_QRELS, "w") as fout:
        for orig_qid, orig_doc_id, score in qrel_rows:
            if orig_doc_id not in orig_to_pid:
                skipped += 1
                continue
            new_pid = orig_to_pid[orig_doc_id]
            fout.write(f"{orig_qid}\t{new_pid}\t{score}\n")
            mapped += 1

    log(f"  Mapped: {mapped}, Skipped: {skipped}")

    log("=== Done! ===")
    log(f"  Collection : {OUTPUT_CORPUS}  ({total_written:,} passages)")
    log(f"  Queries    : {OUTPUT_QUERIES}  ({written_q} queries)")
    log(f"  Qrels      : {OUTPUT_QRELS}  ({mapped} rows)")


if __name__ == "__main__":
    main()
