"""
Phase 1b. 50k 문서 재인코딩 (colbert-ir/colbertv2.0)
  기존 1m.analog 인덱스에서 복원한 벡터(nbits=2 잘못 디코딩)를
  colbertv2.0 모델로 직접 인코딩한 벡터로 교체
  출력:
    scale_all_vectors.npy       (n_tokens, 128)  덮어쓰기
    scale_passage_of_token.npy  (n_tokens,)      덮어쓰기
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import time

from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig

BASE       = r'C:\Users\nmdl-khb\ColBERT\centroidset'
COLLECTION = r'D:\msmarco\collection_1m.tsv'
BATCH_SIZE = 64
DIM        = 128

# ==========================================================================
# 1. 50k passage IDs 로드
# ==========================================================================
print("=" * 60)
print("50k doc pool 로드...")
pool_pids = pd.read_csv(f'{BASE}/scale_doc_pool_50k.csv')['passage_idx'].tolist()
pool_set  = set(pool_pids)
print(f"  passages: {len(pool_pids):,}")

# ==========================================================================
# 2. Collection에서 텍스트 추출
# ==========================================================================
print("collection 텍스트 로드 중...")
t0 = time.time()

pid2text = {}
with open(COLLECTION, encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t', 1)
        if len(parts) == 2:
            pid = int(parts[0])
            if pid in pool_set:
                pid2text[pid] = parts[1]

print(f"  로드 완료: {len(pid2text):,} passages ({time.time()-t0:.1f}s)")
assert len(pid2text) == len(pool_pids), \
    f"텍스트 누락: {len(pid2text)} != {len(pool_pids)}"

# ==========================================================================
# 3. colbertv2.0 모델 로드
# ==========================================================================
print()
print("=" * 60)
print("colbert-ir/colbertv2.0 모델 로드...")
config = ColBERTConfig(
    checkpoint='colbert-ir/colbertv2.0',
    doc_maxlen=220,
    dim=DIM,
)
ckpt = Checkpoint(config.checkpoint, colbert_config=config)
ckpt.eval()
print("  모델 로드 완료")

# ==========================================================================
# 4. 인코딩 (batch 단위)
# ==========================================================================
print()
print("=" * 60)
print(f"50k 문서 인코딩 중 (batch={BATCH_SIZE})...")
t0 = time.time()

doc_texts = [pid2text[pid] for pid in pool_pids]   # pool_pids 순서 유지
n_docs    = len(doc_texts)

all_vectors      = []   # 각 배치의 토큰 벡터
passage_of_token = []   # 각 토큰이 속한 passage_idx (= pid)

for i in range(0, n_docs, BATCH_SIZE):
    batch_texts = doc_texts[i:i + BATCH_SIZE]
    batch_pids  = pool_pids[i:i + BATCH_SIZE]

    with torch.no_grad():
        # keep_dims=False: 패딩 없이 (n_toks, 128) 리스트 반환
        D_list = ckpt.docFromText(batch_texts, bsize=None, keep_dims=False)

    for pid, D in zip(batch_pids, D_list):
        D_np = D.cpu().float().numpy()          # (n_toks, 128)
        all_vectors.append(D_np)
        passage_of_token.extend([pid] * len(D_np))

    if (i // BATCH_SIZE + 1) % 50 == 0:
        elapsed = time.time() - t0
        done    = i + len(batch_texts)
        eta     = elapsed / done * (n_docs - done)
        print(f"  {done:,}/{n_docs:,}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

all_vectors_np      = np.vstack(all_vectors).astype(np.float32)
passage_of_token_np = np.array(passage_of_token, dtype=np.int32)

print(f"  인코딩 완료 ({time.time()-t0:.1f}s)")
print(f"  all_vectors:      {all_vectors_np.shape}")
print(f"  passage_of_token: {passage_of_token_np.shape}")

# ==========================================================================
# 5. 저장
# ==========================================================================
print()
print("=" * 60)
print("저장 중...")
np.save(f'{BASE}/scale_all_vectors.npy',      all_vectors_np)
np.save(f'{BASE}/scale_passage_of_token.npy', passage_of_token_np)

print(f"  scale_all_vectors.npy      -> {all_vectors_np.shape}")
print(f"  scale_passage_of_token.npy -> {passage_of_token_np.shape}")

# 간단한 sanity check
norms = np.linalg.norm(all_vectors_np[:1000], axis=-1)
print(f"  L2 norm check (first 1000): mean={norms.mean():.4f} min={norms.min():.4f}")

print()
print("Phase 1b 완료!")
