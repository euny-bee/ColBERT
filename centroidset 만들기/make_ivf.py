import os
import json
import random
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

OUT_DIR      = r"C:\Users\nmdl-khb\ColBERT\centroidset 만들기"
QRELS_PATH   = r"C:\Users\nmdl-khb\ColBERT\data\msmarco\qrels.dev.small.tsv"
COLLECTION_PATH = r"D:\msmarco\full\collection.tsv"

SEED = 42
random.seed(SEED)

RELEVANT_PIDS = {"7264253", "7264266", "7264308"}

# ── Step 1: 데이터 로드 ────────────────────────────────────────────────────
print("=== Step 1: 데이터 로드 ===")
centroids = torch.from_numpy(np.load(os.path.join(OUT_DIR, "centroids_100x128.npy")))  # (100, 128)
doc_embs  = torch.from_numpy(np.load(os.path.join(OUT_DIR, "doc_embs_12919x128.npy")))  # (12919, 128)
print(f"  centroids: {centroids.shape}")
print(f"  doc_embs:  {doc_embs.shape}")

# make_centroidset.py와 동일한 ordered_pids 재현 (seed=42 고정)
qid2pids = {}
with open(QRELS_PATH, encoding="utf-8") as f:
    for line in f:
        qid, _, pid, _ = line.strip().split()
        qid2pids.setdefault(qid, []).append(pid)
selected_qids = sorted(qid2pids.keys())[:3]
relevant_pids = set()
for qid in selected_qids:
    relevant_pids.update(qid2pids[qid])

all_pids_in_collection = []
with open(COLLECTION_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            all_pids_in_collection.append(parts[0])

remaining = [p for p in all_pids_in_collection if p not in relevant_pids]
random_pids = random.sample(remaining, 200 - len(relevant_pids))
ordered_pids = sorted(relevant_pids) + random_pids
print(f"  ordered_pids: {len(ordered_pids)}개")

# doclens 복원: doc_embs 총 12919개, passage 200개
# make_centroidset.py와 동일한 방식으로 인코딩했으므로 npy에 doclens가 없음
# → doclens.json 저장 여부 확인, 없으면 재계산 필요
doclens_path = os.path.join(OUT_DIR, "doclens.json")
if os.path.exists(doclens_path):
    with open(doclens_path, encoding="utf-8") as f:
        doclens = json.load(f)
    print(f"  doclens loaded: {len(doclens)}개, 합계={sum(doclens)}")
else:
    print("  doclens.json 없음 → make_centroidset.py를 먼저 실행하세요 (doclens 저장 필요)")
    exit(1)

# ── Step 2: token → nearest centroid 배정 ─────────────────────────────────
print("\n=== Step 2: token → nearest centroid 배정 ===")
scores = centroids @ doc_embs.T              # (100, 12919)
token2centroid = scores.argmax(dim=0)        # (12919,)
print(f"  token2centroid shape: {token2centroid.shape}")

# ── Step 3: token → pid 매핑 ──────────────────────────────────────────────
print("\n=== Step 3: token → pid 매핑 ===")
token2pid = []
for pid_idx, length in enumerate(doclens):
    token2pid.extend([ordered_pids[pid_idx]] * length)
assert len(token2pid) == doc_embs.shape[0], "token 수 불일치"
print(f"  token2pid: {len(token2pid)}개")

# ── Step 4: token_id 생성 ─────────────────────────────────────────────────
print("\n=== Step 4: token_id 생성 ===")
token_ids = []
for pid_idx, length in enumerate(doclens):
    pid = ordered_pids[pid_idx]
    for t in range(length):
        token_ids.append(f"{pid}_t{t}")

# ── Step 5: centroid → token → passage 역방향 인덱스 ──────────────────────
print("\n=== Step 5: centroid → token/passage IVF 구축 ===")
centroid2tokens = defaultdict(list)   # centroid → [(token_id, pid), ...]
centroid2pids   = defaultdict(set)    # centroid → {pid, ...}

for token_idx, centroid_id in enumerate(token2centroid.tolist()):
    tid = token_ids[token_idx]
    pid = token2pid[token_idx]
    centroid2tokens[centroid_id].append((tid, pid))
    centroid2pids[centroid_id].add(pid)

total_tokens = sum(len(v) for v in centroid2tokens.values())
print(f"  배정된 centroid 수: {len(centroid2tokens)}/100")
print(f"  centroid당 평균 토큰 수: {total_tokens/100:.1f}")
print(f"  centroid당 평균 passage 수: {sum(len(v) for v in centroid2pids.values())/100:.1f}")

# ── Step 6: 저장 ───────────────────────────────────────────────────────────
print("\n=== Step 6: 저장 ===")

# Long format: centroid_id | token_id | pid | is_relevant
long_rows = []
for cid in range(100):
    for (tid, pid) in sorted(centroid2tokens.get(cid, []), key=lambda x: x[0]):
        long_rows.append({
            "centroid_id": cid,
            "token_id": tid,
            "pid": pid,
            "is_relevant": pid in RELEVANT_PIDS
        })
df_long = pd.DataFrame(long_rows, columns=["centroid_id", "token_id", "pid", "is_relevant"])
df_long.to_csv(os.path.join(OUT_DIR, "ivf_centroid2pid_long.csv"), index=False)
print(f"  long format: {len(df_long)}행 (centroid_id | token_id | pid | is_relevant)")

# Passage-level wide format: centroid_id | n_passages | pid_0 | pid_1 | ...
max_pids = max((len(v) for v in centroid2pids.values()), default=0)
wide_rows = []
for cid in range(100):
    pids = sorted(centroid2pids.get(cid, []))
    n_tokens = len(centroid2tokens.get(cid, []))
    row = {"centroid_id": cid, "n_tokens": n_tokens, "n_passages": len(pids)}
    for i, pid in enumerate(pids):
        row[f"pid_{i}"] = pid
    wide_rows.append(row)

pid_cols = [f"pid_{i}" for i in range(max_pids)]
df_wide = pd.DataFrame(wide_rows, columns=["centroid_id", "n_tokens", "n_passages"] + pid_cols)

with pd.ExcelWriter(os.path.join(OUT_DIR, "ivf_centroid2token2pid.xlsx"), engine="openpyxl") as writer:
    df_long.to_excel(writer, sheet_name="token_level", index=False)
    df_wide.to_excel(writer, sheet_name="passage_level", index=False)

print(f"  ivf_centroid2token2pid.xlsx 저장 완료")
print(f"    Sheet1 'token_level':   centroid_id | token_id | pid | is_relevant")
print(f"    Sheet2 'passage_level': centroid_id | n_tokens | n_passages | pid_0 ...")
print("\n=== 완료 ===")
