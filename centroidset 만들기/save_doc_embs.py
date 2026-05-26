import os
import json
import random
import numpy as np
import pandas as pd

OUT_DIR      = r"C:\Users\nmdl-khb\ColBERT\centroidset 만들기"
QRELS_PATH   = r"C:\Users\nmdl-khb\ColBERT\data\msmarco\qrels.dev.small.tsv"
COLLECTION_PATH = r"D:\msmarco\full\collection.tsv"
SEED = 42
random.seed(SEED)

# ── Step 1: 데이터 로드 ────────────────────────────────────────────────────
print("=== Step 1: 데이터 로드 ===")
doc_embs = np.load(os.path.join(OUT_DIR, "doc_embs_12919x128.npy"))   # (12919, 128)
with open(os.path.join(OUT_DIR, "doclens.json"), encoding="utf-8") as f:
    doclens = json.load(f)
print(f"  doc_embs: {doc_embs.shape}")
print(f"  doclens: {len(doclens)}개, 합계={sum(doclens)}")

# ── ordered_pids 재현 (make_centroidset.py와 동일 로직, seed=42) ───────────
qid2pids = {}
with open(QRELS_PATH, encoding="utf-8") as f:
    for line in f:
        qid, _, pid, _ = line.strip().split()
        qid2pids.setdefault(qid, []).append(pid)
selected_qids = sorted(qid2pids.keys())[:3]
relevant_pids = set()
for qid in selected_qids:
    relevant_pids.update(qid2pids[qid])

all_pids = []
with open(COLLECTION_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            all_pids.append(parts[0])

remaining = [p for p in all_pids if p not in relevant_pids]
random_pids = random.sample(remaining, 200 - len(relevant_pids))
ordered_pids = sorted(relevant_pids) + random_pids
print(f"  ordered_pids: {len(ordered_pids)}개")

# ── Step 2: 행 이름 및 is_relevant 생성 ───────────────────────────────────
print("\n=== Step 2: 행 이름 생성 ===")
row_names = []
is_relevant_col = []
for pid_idx, length in enumerate(doclens):
    pid = ordered_pids[pid_idx]
    is_rel = pid in relevant_pids
    for t in range(length):
        row_names.append(f"{pid}_t{t}")
        is_relevant_col.append(is_rel)
assert len(row_names) == doc_embs.shape[0]
print(f"  행 이름 {len(row_names)}개 생성 완료")
print(f"  예시: {row_names[:3]} ... {row_names[-2:]}")

# ── Step 3: 저장 ───────────────────────────────────────────────────────────
print("\n=== Step 3: 저장 ===")
col_names = [f"dim_{i}" for i in range(128)]
df = pd.DataFrame(doc_embs, columns=col_names)
df.insert(0, "is_relevant", is_relevant_col)   # token_id 바로 옆 열
df.index = row_names
df.index.name = "token_id"

df.to_csv(os.path.join(OUT_DIR, "doc_embs_12919x128.csv"))
print(f"  CSV 저장 완료: doc_embs_12919x128.csv")

df.to_excel(os.path.join(OUT_DIR, "doc_embs_12919x128.xlsx"))
print(f"  Excel 저장 완료: doc_embs_12919x128.xlsx")

# ── 검증 ───────────────────────────────────────────────────────────────────
print("\n=== 검증 ===")
print(f"  shape: {df.shape}")
print(f"  is_relevant=True 토큰 수: {sum(is_relevant_col)}")
print(f"  is_relevant=False 토큰 수: {len(is_relevant_col) - sum(is_relevant_col)}")

print("\n=== 완료 ===")
