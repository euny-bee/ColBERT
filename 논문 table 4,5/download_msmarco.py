#!/usr/bin/env python3
"""
HuggingFace에서 MS MARCO collection 다운로드 후 TSV 변환.
Tevatron/msmarco-passage-corpus 사용 (pid, text 형식)
"""
import os
import time
from datasets import load_dataset

OUT_FULL = "D:/msmarco/full/collection.tsv"
OUT_1M   = "D:/msmarco/collection_1m.tsv"
LIMIT_1M = 1_100_000

os.makedirs("D:/msmarco/full", exist_ok=True)
os.makedirs("D:/msmarco", exist_ok=True)

print("Loading MS MARCO from HuggingFace (Tevatron/msmarco-passage-corpus)...")
ds = load_dataset("Tevatron/msmarco-passage-corpus", split="train", streaming=True)

t0 = time.time()
count = 0

with open(OUT_FULL, "w", encoding="utf-8") as f_full, \
     open(OUT_1M,   "w", encoding="utf-8") as f_1m:

    for example in ds:
        pid  = example["docid"]
        text = example["text"].replace("\t", " ").replace("\n", " ")
        line = f"{pid}\t{text}\n"

        f_full.write(line)

        if count < LIMIT_1M:
            f_1m.write(line)

        count += 1

        if count % 100_000 == 0:
            elapsed = time.time() - t0
            speed   = count / elapsed
            remaining = (8_841_823 - count) / speed
            print(f"  {count:>7,}  |  {count/8841823*100:.1f}%  |  "
                  f"{speed:.0f} rows/s  |  남은시간 {remaining/60:.0f}분", flush=True)

        if count >= 8_841_823:
            break

elapsed = time.time() - t0
print(f"\n완료! {count:,}개 passages, {elapsed/60:.1f}분 소요")
print(f"  전체: {OUT_FULL}")
print(f"  1M:   {OUT_1M}")
