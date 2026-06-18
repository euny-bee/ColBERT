# -*- coding: utf-8 -*-
"""Phase 3-Analog: analog pipeline (query range)
usage: python phase3_analog.py <gpu_id> <q_start> <q_end>
"""

import os, time, sys
import numpy as np
import pandas as pd
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

gpu_id  = int(sys.argv[1]) if len(sys.argv) > 1 else 1
q_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
q_end   = int(sys.argv[3]) if len(sys.argv) > 3 else 255

DEVICE = torch.device(f'cuda:{gpu_id}')
BASE   = r'C:\Users\nmdl-khb\ColBERT\centroidset'
NPROBE = 2

Vth=0.151045; L_f=6.112020; K_f=2.731949; B_f=-10.713911

def calc_ids(diff):
    vgs  = diff + Vth
    ids  = 10**(B_f + L_f/(1+torch.exp(-K_f*(vgs-Vth))))
    VoD  = vgs - Vth
    fact = torch.ones_like(VoD)
    on   = VoD > 0
    Vo   = VoD[on]
    Im   = torch.where(Vo>1.0, Vo*1.0-0.5, Vo**2/2)
    It   = torch.where(Vo>1.7, Vo*1.7-1.7**2/2, Vo**2/2)
    fact[on] = torch.where(Im>0, It/Im, torch.ones_like(Im))
    return ids * fact

print(f"=== Analog Pipeline (GPU{gpu_id}, q{q_start}~{q_end-1}) ===")
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

def ana_step2(Q_q, batch=4096):
    I=torch.zeros(32, len(C), device=DEVICE)
    for s in range(0, len(C), batch):
        e=min(s+batch, len(C))
        diff=torch.abs(Q_q.unsqueeze(1)-C[s:e].unsqueeze(0))
        I[:,s:e]=calc_ids(diff).sum(dim=2)*1e6
    return torch.argsort(I, dim=1)[:, :NPROBE].cpu().numpy()

def step3(top_c):
    pids=set()
    for c_ids in top_c:
        for c in c_ids: pids.update(ivf_pids[int(c)].tolist())
    return list(pids)

def score_ana(Q_q, vecs, cands):
    scores={}
    for pid in cands:
        idxs=pid2tok.get(pid)
        if idxs is None: continue
        D=vecs[idxs].float()
        diff=torch.abs(Q_q.unsqueeze(1)-D.unsqueeze(0))
        I=calc_ids(diff).sum(dim=2)*1e6
        scores[pid]=float(I.min(dim=1).values.sum())
    return scores

def get_rank(sc, pid, asc):
    if pid not in sc: return None
    return sorted(sc.values(), reverse=not asc).index(sc[pid])+1

total = q_end - q_start
print(f"\nProcessing {total} queries ({q_start}~{q_end-1})...")
results=[]
for q_idx in range(q_start, q_end):
    tp  = true_pids[q_idx]
    Q_q = Q_all[q_idx]
    top = ana_step2(Q_q)
    ca  = step3(top)
    saf = score_ana(Q_q, vecs_f16,  ca)
    sa2 = score_ana(Q_q, vecs_2bit, ca)
    results.append({
        'query_idx':   q_idx,
        'qid':         meta.loc[q_idx,'qid'],
        'true_pid':    tp,
        'n_cands_ana': len(saf),
        'in_ana':      tp in saf,
        'rank_ana_f32':get_rank(saf,tp,True),
        'rank_ana_2bt':get_rank(sa2,tp,True),
        'curr_ana_f32':saf.get(tp),
        'curr_ana_2bt':sa2.get(tp),
    })
    done = q_idx - q_start + 1
    if done % 50 == 0:
        el=time.time()-t0; eta=el/done*(total-done)
        print(f"  [{done}/{total}]  {el:.0f}s  ETA:{eta:.0f}s"); sys.stdout.flush()

out = f'{BASE}/phase3_analog_results_gpu{gpu_id}.csv'
pd.DataFrame(results).to_csv(out, index=False)
print(f"\nDone: {time.time()-t0:.1f}s  ->  {out}")
