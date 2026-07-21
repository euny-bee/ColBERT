"""
새 3-query 연구용 데이터셋 구축 (Step 1~7) -- margin(1등 true vs 2등 경쟁 문서) 기준 선정
  q0: qid 329114 "how much to have your house reshingled"  true_pid=471602  margin=1.62  (경쟁 치열)
  q1: qid 498398 "simple definition wave amplitude"        true_pid=822108  margin=11.35 (중간)
  q2: qid 984178 "where is hartwell ga"                    true_pid=347885  margin=20.06 (압도적 1등)

산출물 (이 폴더 안, 기존 clip99.9 파일명 관례 그대로 -- 폴더 자체가 새 실험을 구분):
  [original]query_embs_96x128.xlsx / centroids_100x128.xlsx / doc_embs_Nx128.xlsx
  [clip99.9]query_embs_96x128.xlsx / centroids_100x128.xlsx / doc_embs_Nx128.xlsx
  [clip99.9]ivf_centroid2pid.xlsx        (long_format, wide_format)
  [clip99.9]query_centroid_ranking.xlsx  (scores, ranks, rank_order)
"""

import random
import numpy as np
import pandas as pd
import torch
import faiss
from collections import defaultdict
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig

BASE       = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
META_BASE  = r'C:\Users\nmdl-khb\ColBERT\centroidset'
COLLECTION = r'D:\msmarco\collection_1m.tsv'
N_LINES    = 1_100_000
N_DOCS     = 200
N_CENT     = 100
SEED       = 42

QUERY_QIDS = [329114, 498398, 984178]     # -> q0, q1, q2
TRUE_PIDS  = [471602, 822108, 347885]     # 각 qid의 true passage (같은 순서)
RANK2_PIDS = [559729, 123013, 326168]     # 각 query의 실제 2등 경쟁 passage (500-distractor margin 계산에서 식별, 같은 순서)

# ==========================================================================
# Step 1: Query 임베딩 재사용 (기존 254-query 인코딩 결과에서 추출)
# ==========================================================================
print("[Step 1] Query 임베딩 추출 중...")
meta  = pd.read_csv(f'{META_BASE}/scale_query_meta.csv')
Q_all = torch.load(f'{META_BASE}/scale_query_embs_255x32x128.pt').float()

qid2idx = {qid: i for i, qid in enumerate(meta['qid'].tolist())}
q_texts = []
Q_list = []
for qid in QUERY_QIDS:
    i = qid2idx[qid]
    Q_list.append(Q_all[i])
    q_texts.append(meta.iloc[i]['text'])
    print(f"  qid={qid}  text='{meta.iloc[i]['text']}'  true_pid(meta)={meta.iloc[i]['true_pid']}")
Q3 = torch.stack(Q_list)   # (3, 32, 128)
assert Q3.shape == (3, 32, 128)

# ==========================================================================
# Step 2: 197개 distractor 선정 + 200-doc pool 구성
# ==========================================================================
print("\n[Step 2] Distractor 샘플링 중 (실제 2등 경쟁 passage 3개 강제 포함)...")
required = TRUE_PIDS + RANK2_PIDS
random.seed(SEED)
candidates = list(set(range(N_LINES)) - set(required))
n_random = N_DOCS - len(required)
distractors = random.sample(candidates, n_random)
pool_pids = sorted(required + distractors)
print(f"  pool 크기: {len(pool_pids)}  (true={len(TRUE_PIDS)}, rank2={len(RANK2_PIDS)}, random distractor={len(distractors)})")

# ==========================================================================
# Step 3: 200개 passage 텍스트 로드 + 인코딩
# ==========================================================================
print("\n[Step 3] Passage 텍스트 로드 중...")
pool_set = set(pool_pids)
pid2text = {}
with open(COLLECTION, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx in pool_set:
            _, text = line.rstrip('\n').split('\t', 1)
            pid2text[idx] = text
        if len(pid2text) == len(pool_set):
            break
assert len(pid2text) == N_DOCS, f"passage 텍스트 누락: {N_DOCS - len(pid2text)}개"

doc_texts = [pid2text[pid] for pid in pool_pids]

print("  ColBERT checkpoint 로드 중...")
config = ColBERTConfig(checkpoint='colbert-ir/colbertv2.0', query_maxlen=32, dim=128)
ckpt = Checkpoint(config.checkpoint, colbert_config=config)
ckpt.eval()

print("  Passage 인코딩 중 (docFromText, flatten)...")
with torch.no_grad():
    D_all, doclens = ckpt.docFromText(doc_texts, bsize=64, keep_dims='flatten', showprogress=True)
D_all = D_all.float()
N_TOK = D_all.shape[0]
print(f"  총 토큰 수: {N_TOK}  (passage당 평균 {N_TOK/N_DOCS:.1f})")

offsets = np.cumsum([0] + doclens)
token_ids = []
pid_of_token = []
for i, pid in enumerate(pool_pids):
    for t in range(doclens[i]):
        token_ids.append(f'{pid}_t{t}')
        pid_of_token.append(pid)
is_relevant = np.array([p in TRUE_PIDS for p in pid_of_token], dtype=bool)

# ==========================================================================
# Step 4: Local centroid 100개 (FAISS K-means)
# ==========================================================================
print("\n[Step 4] K-means centroid 100개 생성 중...")
D_np = D_all.numpy().astype(np.float32)
try:
    kmeans = faiss.Kmeans(128, N_CENT, niter=20, verbose=True, gpu=True, seed=SEED)
    kmeans.train(D_np)
except Exception as e:
    print(f"  GPU K-means 실패({e}), CPU로 재시도")
    kmeans = faiss.Kmeans(128, N_CENT, niter=20, verbose=True, gpu=False, seed=SEED)
    kmeans.train(D_np)
C_np = kmeans.centroids.astype(np.float32)   # (100, 128)
print(f"  centroids shape: {C_np.shape}")

# ==========================================================================
# Step 5: Raw(=[original]) Excel 저장
# ==========================================================================
print("\n[Step 5] Raw 임베딩 Excel 저장 중...")
dim_cols = [f'dim_{i}' for i in range(128)]

q_token_ids = [f'q{qi}_t{t}' for qi in range(3) for t in range(32)]
Q_np = Q3.reshape(96, 128).numpy().astype(np.float32)
query_df = pd.DataFrame(Q_np, index=q_token_ids, columns=dim_cols)
query_df.index.name = 'token_id'

cent_df = pd.DataFrame(C_np, index=[f'centroid_{i}' for i in range(N_CENT)], columns=dim_cols)
cent_df.index.name = 'centroid_id'

doc_df = pd.DataFrame(D_np, index=token_ids, columns=dim_cols)
doc_df.insert(0, 'is_relevant', is_relevant)
doc_df.index.name = 'token_id'

query_df.to_excel(f'{BASE}/[original]query_embs_96x128.xlsx')
cent_df.to_excel(f'{BASE}/[original]centroids_100x128.xlsx')
doc_df.to_excel(f'{BASE}/[original]doc_embs_{N_TOK}x128.xlsx')
print(f"  저장 완료: [original] query/centroids/doc_embs_{N_TOK}x128")

# ==========================================================================
# Step 6: 99.9 percentile clip + scale 후 Excel 저장
# ==========================================================================
print("\n[Step 6] 99.9 percentile clip+scale 중...")
all_abs = np.abs(np.concatenate([Q_np.flatten(), C_np.flatten(), D_np.flatten()]))
thr = np.percentile(all_abs, 99.9)
sf  = 1.0 / thr
print(f"  threshold(99.9%)={thr:.6f}  scale_factor={sf:.6f}")

Q_c = np.clip(Q_np, -thr, thr) * sf
C_c = np.clip(C_np, -thr, thr) * sf
D_c = np.clip(D_np, -thr, thr) * sf

query_c_df = pd.DataFrame(Q_c, index=q_token_ids, columns=dim_cols)
query_c_df.index.name = 'token_id'
cent_c_df = pd.DataFrame(C_c, index=[f'centroid_{i}' for i in range(N_CENT)], columns=dim_cols)
cent_c_df.index.name = 'centroid_id'
doc_c_df = pd.DataFrame(D_c, index=token_ids, columns=dim_cols)
doc_c_df.insert(0, 'is_relevant', is_relevant)
doc_c_df.index.name = 'token_id'

query_c_df.to_excel(f'{BASE}/[clip99.9]query_embs_96x128.xlsx')
cent_c_df.to_excel(f'{BASE}/[clip99.9]centroids_100x128.xlsx')
doc_c_df.to_excel(f'{BASE}/[clip99.9]doc_embs_{N_TOK}x128.xlsx')
print(f"  저장 완료: [clip99.9] query/centroids/doc_embs_{N_TOK}x128")

# ==========================================================================
# Step 7: IVF / query_centroid_ranking 재계산 (clip99.9 버전 기준)
# ==========================================================================
print("\n[Step 7] IVF / QCR 재계산 중...")

D_n = (D_c ** 2).sum(axis=1, keepdims=True)
C_n = (C_c ** 2).sum(axis=1)
dist = D_n + C_n[np.newaxis, :] - 2 * (D_c @ C_c.T)
asgn = np.argmin(dist, axis=1)

pids_arr = np.array(pid_of_token, dtype=np.int64)
ivf = defaultdict(dict)
for c_id, pid, is_rel in zip(asgn, pids_arr, is_relevant):
    c_id = int(c_id)
    if pid not in ivf[c_id]:
        ivf[c_id][pid] = bool(is_rel)

long_rows = []
for c_id in sorted(ivf.keys()):
    for pid, is_rel in ivf[c_id].items():
        long_rows.append({'centroid_id': c_id, 'pid': int(pid), 'is_relevant': is_rel})
long_df = pd.DataFrame(long_rows).set_index('centroid_id')
long_df.index.name = 'centroid_id'

wide_rows = {}
for c_id in range(N_CENT):
    pid_list = list(ivf[c_id].keys()) if c_id in ivf else []
    row = {'n_passages': len(pid_list)}
    for i, p in enumerate(pid_list):
        row[f'pid_{i}'] = int(p)
    wide_rows[c_id] = row
wide_df = pd.DataFrame(wide_rows).T
wide_df.index.name = 'centroid_id'
wide_df['n_passages'] = wide_df['n_passages'].astype(int)

query_ids_arr = np.array([qi for qi in range(3) for _ in range(32)])
scores = Q_c @ C_c.T   # (96, 100)
centroid_cols = [f'centroid_{i}' for i in range(N_CENT)]

scores_df = pd.DataFrame(scores, index=q_token_ids, columns=centroid_cols)
scores_df.insert(0, 'query_id', query_ids_arr)
scores_df.index.name = 'token_id'

ranks = scores.shape[1] - np.argsort(np.argsort(scores, axis=1), axis=1)
ranks_df = pd.DataFrame(ranks, index=q_token_ids, columns=centroid_cols)
ranks_df.insert(0, 'query_id', query_ids_arr)
ranks_df.index.name = 'token_id'

order = np.argsort(-scores, axis=1)
rank_cols = [f'rank_{i+1}' for i in range(N_CENT)]
order_df = pd.DataFrame(order, index=q_token_ids, columns=rank_cols)
order_df.insert(0, 'query_id', query_ids_arr)
order_df.index.name = 'token_id'

ivf_path = f'{BASE}/[clip99.9]ivf_centroid2pid.xlsx'
with pd.ExcelWriter(ivf_path, engine='openpyxl') as writer:
    long_df.to_excel(writer, sheet_name='long_format')
    wide_df.to_excel(writer, sheet_name='wide_format')
print(f"  저장: [clip99.9]ivf_centroid2pid.xlsx")

qcr_path = f'{BASE}/[clip99.9]query_centroid_ranking.xlsx'
with pd.ExcelWriter(qcr_path, engine='openpyxl') as writer:
    scores_df.to_excel(writer, sheet_name='scores')
    ranks_df.to_excel(writer, sheet_name='ranks')
    order_df.to_excel(writer, sheet_name='rank_order')
print(f"  저장: [clip99.9]query_centroid_ranking.xlsx")

print(f"\n완료! N_TOK={N_TOK}")
with open(f'{BASE}/meta.txt', 'w', encoding='utf-8') as f:
    f.write(f"N_TOK={N_TOK}\n")
    for qi, (qid, text, tpid) in enumerate(zip(QUERY_QIDS, q_texts, TRUE_PIDS)):
        f.write(f"q{qi}: qid={qid} true_pid={tpid} text='{text}'\n")
