# -*- coding: utf-8 -*-
"""Phase 3-Digital: 255 queries digital pipeline (GPU0)"""

import os, time, sys
import numpy as np
import pandas as pd
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DEVICE = torch.device('cuda:0')
BASE   = r'C:\Users\nmdl-khb\ColBERT\centroidset'
NPROBE = 2

print("=== Digital Pipeline (GPU0) ===")
t0 = time.time()

Q_all    = torch.load(f'{BASE}/scale_query_embs_255x32x128.pt').float().to(DEVICE)
C        = torch.tensor(np.load(f'{BASE}/scale_centroids_32k.npy'), dtype=torch.float32, device=DEVICE)
vecs_np  = np.load(f'{BASE}/scale_all_vectors.npy').astype(np.float32)
assign   = np.load(f'{BASE}/scale_assignments.npy').astype(np.int64)
pass_t   = np.load(f'{BASE}/scale_passage_of_token.npy')
ivf_pids = np.load(f'{BASE}/scale_ivf_pids.npy', allow_pickle=True)
meta     = pd.read_csv(f'{BASE}/scale_query_meta.csv')

vecs_f16 = torch.tensor(vecs_np, dtype=torch.float16, device=DEVICE)
print(f"  vecs loaded  {time.time()-t0:.1f}s")

flat = vecs_np[:500_000].flatten()
p    = np.percentile(flat, [12.5, 37.5, 62.5, 87.5])
bc   = np.array([(flat[flat<=p[0]]).mean(),
                 (flat[(flat>p[0])&(flat<=p[1])]).mean(),
                 (flat[(flat>p[1])&(flat<=p[2])]).mean(),
                 (flat[flat>p[2]]).mean()], dtype=np.float32)
bc_t = torch.tensor(bc, device=DEVICE)
tvs  = (bc_t[:-1]+bc_t[1:])/2
C_np = np.load(f'{BASE}/scale_centroids_32k.npy')
BATCH = 1_000_000
vecs_2bit = torch.empty_like(vecs_f16)
for s in range(0, len(vecs_np), BATCH):
    e   = min(s+BATCH, len(vecs_np))
    v   = torch.tensor(vecs_np[s:e], dtype=torch.float32, device=DEVICE)
    c   = torch.tensor(C_np[assign[s:e]], dtype=torch.float32, device=DEVICE)
    res = v - c
    cd  = (res>tvs[0]).long()+(res>tvs[1]).long()+(res>tvs[2]).long()
    vecs_2bit[s:e] = (c+bc_t[cd]).half()
    del v, c, res, cd
print(f"  2bit done  {time.time()-t0:.1f}s")

order=np.argsort(pass_t,kind='stable'); sp=pass_t[order]
up,starts=np.unique(sp,return_index=True)
ends=np.concatenate([starts[1:],[len(pass_t)]])
pid2tok={int(p):order[s:e] for p,s,e in zip(up,starts,ends)}
true_pids=meta['true_passage_idx'].tolist()
del vecs_np
print(f"  ready  {time.time()-t0:.1f}s")

def dig_step2(Q_q):
    return torch.argsort(Q_q @ C.T, dim=1, descending=True)[:, :NPROBE].cpu().numpy()

def step3(top_c):
    pids=set()
    for c_ids in top_c:
        for c in c_ids: pids.update(ivf_pids[int(c)].tolist())
    return list(pids)

def score_dig(Q_q, vecs, cands):
    scores={}
    for pid in cands:
        idxs=pid2tok.get(pid)
        if idxs is None: continue
        scores[pid]=float((Q_q @ vecs[idxs].float().T).max(dim=1).values.sum())
    return scores

def get_rank(sc, pid, asc):
    if pid not in sc: return None
    return sorted(sc.values(), reverse=not asc).index(sc[pid])+1

n_q = len(meta)
print(f"\nProcessing {n_q} queries...")
results=[]
for q_idx in range(n_q):
    tp  = true_pids[q_idx]
    Q_q = Q_all[q_idx]
    top = dig_step2(Q_q)
    cd  = step3(top)
    sdf = score_dig(Q_q, vecs_f16,  cd)
    sd2 = score_dig(Q_q, vecs_2bit, cd)
    margin=None
    if tp in sdf and len(sdf)>=2:
        sv=sorted(sdf.values(),reverse=True); margin=sv[0]-sv[1]
    results.append({
        'query_idx':    q_idx,
        'qid':          meta.loc[q_idx,'qid'],
        'true_pid':     tp,
        'n_cands_dig':  len(sdf),
        'in_dig':       tp in sdf,
        'rank_dig_f32': get_rank(sdf,tp,False),
        'rank_dig_2bt': get_rank(sd2,tp,False),
        'score_dig_f32':sdf.get(tp),
        'score_dig_2bt':sd2.get(tp),
        'margin':       margin,
    })
    if (q_idx+1)%50==0:
        print(f"  [{q_idx+1}/{n_q}]  {time.time()-t0:.1f}s"); sys.stdout.flush()

pd.DataFrame(results).to_csv(f'{BASE}/phase3_digital_results.csv', index=False)
print(f"\nDone: {time.time()-t0:.1f}s  ->  phase3_digital_results.csv")
