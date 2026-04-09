#!/usr/bin/env python3
"""
make_hotpotqa_fair.py
---------------------
BEIR-HotpotQA corpus에서 2.6M fair 컬렉션 생성.
- qrels.test의 모든 relevant passage 포함 보장
- 나머지는 랜덤 샘플링
- ColBERT 요구사항(pid == line_idx) 맞게 순차 재번호
- qrels도 새 pid로 리매핑
"""

import json
import os
import random
import time

CORPUS_PATH     = "D:/beir/hotpotqa/corpus.jsonl"
QUERIES_PATH    = "D:/beir/hotpotqa/queries.jsonl"
QRELS_PATH      = "D:/beir/hotpotqa/qrels/test.tsv"
OUTPUT_CORPUS   = "D:/beir/hotpotqa/collection_2.6m_fair.tsv"
OUTPUT_QRELS    = "D:/beir/hotpotqa/qrels.test.fair.tsv"
OUTPUT_QUERIES  = "D:/beir/hotpotqa/queries.test.tsv"
TARGET_SIZE     = 2_600_000
SEED            = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    random.seed(SEED)

    # 1. qrels test에서 relevant doc_id 수집
    log("Loading qrels test...")
    relevant_ids = set()
    qrel_rows = []  # (query_id, doc_id, score)
    with open(QRELS_PATH) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, score = parts[0], parts[1], parts[2]
            relevant_ids.add(doc_id)
            qrel_rows.append((query_id, doc_id, score))
    log(f"  Relevant doc IDs: {len(relevant_ids)}")
    log(f"  Qrel rows: {len(qrel_rows)}")

    # 2. test query id 수집
    log("Loading test query IDs...")
    test_query_ids = set(r[0] for r in qrel_rows)
    log(f"  Test query count: {len(test_query_ids)}")

    # 3. corpus 스캔: relevant 라인 저장 + 나머지 id 목록
    log("Scanning corpus...")
    relevant_docs = {}   # doc_id(str) -> {"title": ..., "text": ...}
    other_ids = []       # non-relevant doc ids

    with open(CORPUS_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            doc_id = doc["_id"]
            if doc_id in relevant_ids:
                relevant_docs[doc_id] = doc
            else:
                other_ids.append(doc_id)
            if (i + 1) % 1_000_000 == 0:
                log(f"  Scanned {i+1:,} lines...")

    log(f"  Relevant docs found: {len(relevant_docs)} / {len(relevant_ids)}")
    log(f"  Non-relevant docs: {len(other_ids):,}")

    missing = relevant_ids - set(relevant_docs.keys())
    if missing:
        log(f"  WARNING: {len(missing)} relevant docs not found in corpus!")

    # 4. 나머지 랜덤 샘플링
    n_sample = TARGET_SIZE - len(relevant_docs)
    log(f"Sampling {n_sample:,} non-relevant passages (seed={SEED})...")
    sampled_ids = set(random.sample(other_ids, n_sample))

    # 5. 두 번째 패스: 선택된 라인 출력 (PID 순차 재번호)
    log("Writing fair corpus (sequential PIDs 0..N-1)...")
    orig_to_new = {}   # original doc_id(str) -> new int pid
    total_written = 0

    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as out:
        # relevant passages first (sorted for determinism)
        for orig_id in sorted(relevant_docs.keys()):
            doc = relevant_docs[orig_id]
            title = doc.get("title", "").replace("\t", " ").replace("\n", " ")
            text  = doc.get("text",  "").replace("\t", " ").replace("\n", " ")
            combined = f"{title} {text}".strip() if title else text
            new_pid = total_written
            orig_to_new[orig_id] = new_pid
            out.write(f"{new_pid}\t{combined}\n")
            total_written += 1

        # sampled non-relevant (second pass)
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                orig_id = doc["_id"]
                if orig_id in sampled_ids:
                    title = doc.get("title", "").replace("\t", " ").replace("\n", " ")
                    text  = doc.get("text",  "").replace("\t", " ").replace("\n", " ")
                    combined = f"{title} {text}".strip() if title else text
                    new_pid = total_written
                    orig_to_new[orig_id] = new_pid
                    out.write(f"{new_pid}\t{combined}\n")
                    total_written += 1
                if (i + 1) % 1_000_000 == 0:
                    log(f"  Pass2: {i+1:,} scanned, {total_written:,} written...")

    log(f"Done! Total written: {total_written:,}")
    assert total_written == TARGET_SIZE, f"Expected {TARGET_SIZE}, got {total_written}"

    # 검증: pid == line_idx
    log("Verifying pid == line_idx...")
    with open(OUTPUT_CORPUS, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            pid = int(line.split("\t", 1)[0])
            assert pid == idx, f"pid={pid} != line_idx={idx}"
            if idx == 999_999:
                log("  1M lines verified OK...")
    log("  Verification passed!")

    # 6. qrels 리매핑
    log(f"Writing remapped qrels → {OUTPUT_QRELS}")
    mapped = 0
    skipped = 0
    with open(OUTPUT_QRELS, "w") as fout:
        for query_id, orig_doc_id, score in qrel_rows:
            if orig_doc_id in orig_to_new:
                new_pid = orig_to_new[orig_doc_id]
                fout.write(f"{query_id}\t{new_pid}\t{score}\n")
                mapped += 1
            else:
                skipped += 1
    log(f"  Mapped: {mapped}, Skipped: {skipped}")

    # 7. test queries 저장 (ColBERT용 tsv: qid\ttext)
    log(f"Writing test queries → {OUTPUT_QUERIES}")
    written_q = 0
    with open(QUERIES_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_QUERIES, "w", encoding="utf-8") as fout:
        for line in fin:
            q = json.loads(line)
            if q["_id"] in test_query_ids:
                text = q["text"].replace("\t", " ").replace("\n", " ")
                fout.write(f"{q['_id']}\t{text}\n")
                written_q += 1
    log(f"  Test queries written: {written_q}")

    log("=== All done! ===")
    log(f"  Corpus:  {OUTPUT_CORPUS}")
    log(f"  Qrels:   {OUTPUT_QRELS}")
    log(f"  Queries: {OUTPUT_QUERIES}")


if __name__ == "__main__":
    main()
