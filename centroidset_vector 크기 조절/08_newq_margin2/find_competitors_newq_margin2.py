"""
newq_margin2 (새 3-query, margin 타이트한 케이스 3개 추가 선정) 용
  각 query의 "경쟁 passage" top-10을 식별한다.

  nq0: qid 581521 "what can cause the right side of your back to hurt"  true_pid=493988  margin=1.76
  nq1: qid 579133 "what blood stream messenger"                        true_pid=303786  margin=1.94
  nq2: qid 690508 "what is a medical mc provider"                      true_pid=178693  margin=2.01

방법론 (기존 q0/q1/q2용 top10_competitors.json을 만든 원본 스크립트는 유실되어 정확한 재현이
불가능함이 확인되어, 아래와 같이 새로 명확하게 정의한 방법을 채택):
  1. 20000-doc pool(centroidset/scale_doc_pool_20k.csv, scale_centroids_20k.npy 등 -- 기존
     phase_reduce_pool.py 산출물, 255개 true passage 항상 포함)에서 query별 IVF top-2 centroid
     후보 passage를 뽑는다 (NPROBE=2, digital 방식과 동일).
  2. 후보 중 true_pid보다 maxsim 점수가 낮은 것만 남기고 점수 내림차순 정렬한다
     (= true passage 바로 아래에서 경쟁하는 "진짜 존재하는" passage들, 점수가 true보다 높은
     범용 고득점 passage는 "경쟁자"가 아니라 노이즈로 간주해 제외).
  3. 상위 10개를 top-10 경쟁자로 저장한다.
"""

import json
import numpy as np
import pandas as pd
import torch

BASE   = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\08_newq_margin2'
META_BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'
DEVICE = torch.device('cuda:0')
NPROBE = 2
N_TOP  = 10

NEW_QIDS  = [581521, 579133, 690508]
TRUE_PIDS = {581521: 493988, 579133: 303786, 690508: 178693}

print("[Step 1] 20k-pool IVF 인덱스 로드 중...")
Q_all    = torch.load(f'{META_BASE}/scale_query_embs_255x32x128.pt').float().to(DEVICE)
C        = torch.tensor(np.load(f'{META_BASE}/scale_centroids_20k.npy'), dtype=torch.float32, device=DEVICE)
vecs_np  = np.load(f'{META_BASE}/scale_all_vectors_20k.npy').astype(np.float32)
pass_t   = np.load(f'{META_BASE}/scale_passage_of_token_20k.npy')
ivf_pids = np.load(f'{META_BASE}/scale_ivf_pids_20k.npy', allow_pickle=True)
meta     = pd.read_csv(f'{META_BASE}/scale_query_meta.csv')
vecs     = torch.tensor(vecs_np, dtype=torch.float32, device=DEVICE)
print(f"  centroids: {C.shape}  vectors: {vecs.shape}")

order = np.argsort(pass_t, kind='stable'); sp = pass_t[order]
up, starts = np.unique(sp, return_index=True)
ends = np.concatenate([starts[1:], [len(pass_t)]])
pid2tok = {int(p): order[s:e] for p, s, e in zip(up, starts, ends)}

qid2idx = {qid: i for i, qid in enumerate(meta['qid'].tolist())}

def dig_step2(Q_q):
    return torch.argsort(Q_q @ C.T, dim=1, descending=True)[:, :NPROBE].cpu().numpy()

def step3(top_c):
    pids = set()
    for c_ids in top_c:
        for c in c_ids:
            pids.update(ivf_pids[int(c)].tolist())
    return list(pids)

def score_cands(Q_q, cands):
    scores = {}
    for pid in cands:
        idxs = pid2tok.get(pid)
        if idxs is None:
            continue
        scores[pid] = float((Q_q @ vecs[idxs].T).max(dim=1).values.sum())
    return scores

print("\n[Step 2] query별 IVF 후보 생성 + true 이하 경쟁자 추출...")
competitors = {}
for qid in NEW_QIDS:
    true_pid = TRUE_PIDS[qid]
    Q_q = Q_all[qid2idx[qid]]
    top = dig_step2(Q_q)
    cd  = step3(top)
    sdf = score_cands(Q_q, cd)
    assert true_pid in sdf, f"qid={qid}: true_pid이 20k-pool IVF 후보에 없음"
    score_true = sdf[true_pid]

    filtered = {p: s for p, s in sdf.items() if p != true_pid and s < score_true}
    ranked = sorted(filtered.items(), key=lambda x: -x[1])
    top10 = [p for p, _ in ranked[:N_TOP]]
    competitors[str(qid)] = top10

    print(f"  qid={qid}  n_cands={len(sdf)}  score_true={score_true:.4f}")
    print(f"    top10 경쟁자: {top10}")
    print(f"    top10 점수:   {[round(s, 4) for _, s in ranked[:N_TOP]]}")

out_path = f'{BASE}/newq_margin2_top10_competitors.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(competitors, f, indent=2)
print(f"\n저장: {out_path}")
print("완료!")
