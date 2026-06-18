"""
Phase 1. 데이터 준비 (수정판)
  - collection_1m_fair.tsv에 정답 passage가 있는 query만 선정
  - true_passage_idx (collection 내 순서) 저장
"""

import os, json, random
import numpy as np
import pandas as pd
import torch
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig

random.seed(42)
np.random.seed(42)

BASE      = r'C:\Users\nmdl-khb\ColBERT\centroidset'
IDX_BASE  = r'C:\Users\nmdl-khb\ColBERT\experiments\msmarco_1m\indexes\1m.analog'
DATA_BASE = r'C:\Users\nmdl-khb\ColBERT\data\msmarco'
N_DOCS    = 50_000

# ==========================================================================
# pid2idx 로드 (check_collection.py에서 생성)
# ==========================================================================
print("pid2idx 로드 중...")
with open(f'{BASE}/collection_pid2idx.json') as f:
    pid2idx = {int(k): v for k, v in json.load(f).items()}
print(f"  collection pid 수: {len(pid2idx):,}")

# ==========================================================================
# [1-1] 유효한 Query 선정 (정답 passage가 collection에 있는 것만)
# ==========================================================================
print("\n[1-1] Query 선정 중...")

qrels = pd.read_csv(f'{DATA_BASE}/qrels.dev.small.tsv',
                    sep='\t', header=None, names=['qid','0','pid','rel'])
queries = pd.read_csv(f'{DATA_BASE}/queries.dev.tsv',
                      sep='\t', header=None, names=['qid','text'])

valid_qrels = qrels[qrels['pid'].isin(pid2idx)]
valid_qids  = valid_qrels['qid'].unique()
print(f"  유효 query 수: {len(valid_qids)}")

# 전체 사용 (255개)
selected_qids  = list(valid_qids)
selected_qrels = valid_qrels.copy()

qid2pid      = dict(zip(selected_qrels['qid'], selected_qrels['pid']))
qid2text     = dict(zip(queries['qid'], queries['text']))
qid2text     = {q: qid2text[q] for q in selected_qids if q in qid2text}
qid2pass_idx = {q: pid2idx[qid2pid[q]] for q in selected_qids}

print(f"  선정 query 수: {len(selected_qids)}")
for qid in list(selected_qids)[:3]:
    print(f"    qid={qid}: '{qid2text.get(qid, '?')}' → pid {qid2pid[qid]} → idx {qid2pass_idx[qid]}")

# ==========================================================================
# [1-2] Document Pool 50K 구성
# ==========================================================================
print("\n[1-2] Document Pool 50K 구성 중...")

required_idxs = set(qid2pass_idx.values())
print(f"  필수 포함 passage_idx 수: {len(required_idxs)}")

total_passages = len(pid2idx)
all_idxs   = list(range(total_passages))
non_req    = [i for i in all_idxs if i not in required_idxs]
extra      = random.sample(non_req, N_DOCS - len(required_idxs))
doc_pool   = sorted(list(required_idxs) + extra)

print(f"  Document pool 크기: {len(doc_pool):,}")

# ==========================================================================
# [1-3] Query Embedding 생성
# ==========================================================================
print("\n[1-3] Query Embedding 생성 중...")

config = ColBERTConfig(checkpoint='colbert-ir/colbertv2.0', query_maxlen=32, dim=128)
ckpt   = Checkpoint(config.checkpoint, colbert_config=config)
ckpt.eval()

query_texts = [qid2text.get(q, '') for q in selected_qids]
BATCH = 64
all_Q = []
for i in range(0, len(query_texts), BATCH):
    batch = query_texts[i:i+BATCH]
    with torch.no_grad():
        Q = ckpt.queryFromText(batch)
    all_Q.append(Q.cpu())
    print(f"  {min(i+BATCH, len(query_texts))}/{len(query_texts)}")

Q_all = torch.cat(all_Q, dim=0)
n_q   = len(selected_qids)
print(f"  Query embedding shape: {Q_all.shape}")

# ==========================================================================
# 저장
# ==========================================================================
print("\n저장 중...")

torch.save(Q_all, f'{BASE}/scale_query_embs_{n_q}x32x128.pt')

query_meta = pd.DataFrame({
    'query_idx':       range(n_q),
    'qid':             selected_qids,
    'text':            [qid2text.get(q,'') for q in selected_qids],
    'true_pid':        [qid2pid[q] for q in selected_qids],
    'true_passage_idx':[qid2pass_idx[q] for q in selected_qids],
})
query_meta.to_csv(f'{BASE}/scale_query_meta.csv', index=False)

pd.Series(doc_pool, name='passage_idx').to_csv(f'{BASE}/scale_doc_pool_50k.csv', index=False)

print(f"  scale_query_embs_{n_q}x32x128.pt  → {Q_all.shape}")
print(f"  scale_query_meta.csv              → {len(query_meta)} rows")
print(f"  scale_doc_pool_50k.csv            → {len(doc_pool):,} passages")
print(f"\nPhase 1 완료! (query 수: {n_q})")
