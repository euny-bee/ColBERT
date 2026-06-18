"""
Phase 3. Digital + Analog Pipeline (500 queries x 4가지) - 2 GPU 버전
  GPU0: digital 연산 (vecs_f32 + vecs_2bit 모두 보유)
  GPU1: analog  연산 (vecs_f32 + vecs_2bit 모두 보유)
  → 메인 루프에서 GPU 간 전송 없음
"""

import os, time, sys
import numpy as np
import pandas as pd
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
GPU0 = torch.device('cuda:0')
GPU1 = torch.device('cuda:1')
BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'
NPROBE = 2

# -- 아날로그 회로 ----------------------------------------------------------
Vth=0.151045; L_f=6.112020; K_f=2.731949; B_f=-10.713911

def calc_ids(diff):
    vgs  = diff + Vth
    ids  = 10 ** (B_f + L_f / (1 + torch.exp(-K_f * (vgs - Vth))))
    VoD  = vgs - Vth
    fact = torch.ones_like(VoD)
    on   = VoD > 0;  Vo = VoD[on]
    Im   = torch.where(Vo > 1.0, Vo*1.0-0.5, Vo**2/2)
    It   = torch.where(Vo > 1.7, Vo*1.7-1.7**2/2, Vo**2/2)
    fact[on] = torch.where(Im > 0, It/Im, torch.ones_like(Im))
    return ids * fact

def mem(dev): return torch.cuda.memory_allocated(dev)/1e9

# ==========================================================================
print("="*60); print("데이터 로드 중...")

Q_all    = torch.load(f'{BASE}/scale_query_embs_500x32x128.pt').float()
C_np     = np.load(f'{BASE}/scale_centroids_32k.npy').astype(np.float32)
vecs_np  = np.load(f'{BASE}/scale_all_vectors.npy').astype(np.float32)
assign   = np.load(f'{BASE}/scale_assignments.npy').astype(np.int64)
pass_t   = np.load(f'{BASE}/scale_passage_of_token.npy')
ivf_pids = np.load(f'{BASE}/scale_ivf_pids.npy', allow_pickle=True)
meta     = pd.read_csv(f'{BASE}/scale_query_meta.csv')

# vecs_f32 → GPU0 float16
print("  vecs_f32 → GPU0...")
vecs_f16_0 = torch.tensor(vecs_np, dtype=torch.float16, device=GPU0)
print(f"  GPU0: {mem(GPU0):.2f}GB")

# 2bit 양자화 → GPU1 배치
print("  2bit 양자화 중 (GPU1)...")
flat = vecs_np[:500_000].flatten()
p   = np.percentile(flat, [12.5, 37.5, 62.5, 87.5])
bc  = np.array([(flat[flat<=p[0]]).mean(),
                (flat[(flat>p[0])&(flat<=p[1])]).mean(),
                (flat[(flat>p[1])&(flat<=p[2])]).mean(),
                (flat[flat>p[2]]).mean()], dtype=np.float32)
print(f"  bucket centers: {bc.round(4)}")
bc1   = torch.tensor(bc, device=GPU1)
tvs   = (bc1[:-1]+bc1[1:])/2
BATCH = 1_000_000
vecs_2bit_f16_1 = torch.empty(len(vecs_np), 128, dtype=torch.float16, device=GPU1)
for s in range(0, len(vecs_np), BATCH):
    e   = min(s+BATCH, len(vecs_np))
    v   = torch.tensor(vecs_np[s:e], dtype=torch.float32, device=GPU1)
    c   = torch.tensor(C_np[assign[s:e]], dtype=torch.float32, device=GPU1)
    res = v - c
    cd  = (res>tvs[0]).long()+(res>tvs[1]).long()+(res>tvs[2]).long()
    vecs_2bit_f16_1[s:e] = (c + bc1[cd]).half()
    del v, c, res, cd
    print(f"    {e:,}/{len(vecs_np):,}  GPU1:{mem(GPU1):.2f}GB")

# 각 GPU에 두 vec 세트 모두 배치 (GPU 간 전송 제거)
print("  vecs 복사 중...")
vecs_f16_1    = vecs_f16_0.to(GPU1)           # GPU0→GPU1 (f32 copy)
vecs_2bit_f16_0 = vecs_2bit_f16_1.to(GPU0)   # GPU1→GPU0 (2bit copy)
print(f"  GPU0: {mem(GPU0):.2f}GB  GPU1: {mem(GPU1):.2f}GB")

del vecs_np
# GPU: C, Q 배치
C0=torch.tensor(C_np,device=GPU0); C1=torch.tensor(C_np,device=GPU1)
Q0=Q_all.to(GPU0);                  Q1=Q_all.to(GPU1)

# pid2tokidxs
print("  pid2tokidxs 구성...")
order=np.argsort(pass_t,kind='stable'); sp=pass_t[order]
up,starts=np.unique(sp,return_index=True)
ends=np.concatenate([starts[1:],[len(pass_t)]])
pid2tok={int(p):order[s:e] for p,s,e in zip(up,starts,ends)}
true_pids=meta['true_pid'].tolist()
print(f"  완료: {len(pid2tok):,}개  |  GPU0:{mem(GPU0):.2f}GB  GPU1:{mem(GPU1):.2f}GB")

# ==========================================================================
def dig_step2(Q_q, C, nprobe):
    return torch.argsort(Q_q@C.T, dim=1, descending=True)[:, :nprobe]

def ana_step2(Q_q, C, nprobe, batch=4096):
    I=torch.zeros(32, len(C), device=Q_q.device)
    for s in range(0,len(C),batch):
        e=min(s+batch,len(C))
        diff=torch.abs(Q_q.unsqueeze(1)-C[s:e].unsqueeze(0))
        I[:,s:e]=calc_ids(diff).sum(dim=2)*1e6
    return torch.argsort(I, dim=1)[:, :nprobe]

def step3(top_c):
    pids=set()
    for c_ids in top_c:
        for c in c_ids: pids.update(ivf_pids[int(c)].tolist())
    return list(pids)

def score_dig(Q_q, vecs_f16, cands, pid2tok):
    scores={}
    for pid in cands:
        idxs=pid2tok.get(pid)
        if idxs is None: continue
        D=vecs_f16[idxs].float()
        scores[pid]=float((Q_q@D.T).max(dim=1).values.sum())
    return scores

def score_ana(Q_q, vecs_f16, cands, pid2tok):
    scores={}
    for pid in cands:
        idxs=pid2tok.get(pid)
        if idxs is None: continue
        D=vecs_f16[idxs].float()
        diff=torch.abs(Q_q.unsqueeze(1)-D.unsqueeze(0))
        I=calc_ids(diff).sum(dim=2)*1e6
        scores[pid]=float(I.min(dim=1).values.sum())
    return scores

def get_rank(sc, pid, asc):
    if pid not in sc: return None
    return sorted(sc.values(), reverse=not asc).index(sc[pid])+1

# ==========================================================================
print(); print("="*60)
print("Phase 3 실행 (GPU0=digital / GPU1=analog)...")
sys.stdout.flush()
t0=time.time(); results=[]

for q_idx in range(500):
    tp = true_pids[q_idx]
    q0 = Q0[q_idx];  q1 = Q1[q_idx]

    # Step 2 (순차 — GPU 각각)
    top_dig = dig_step2(q0, C0, NPROBE).cpu().numpy()
    top_ana = ana_step2(q1, C1, NPROBE).cpu().numpy()

    # Step 3
    cd = step3(top_dig);  ca = step3(top_ana)

    # Step 6 (GPU0: digital, GPU1: analog)
    sdf = score_dig(q0, vecs_f16_0,      cd, pid2tok)
    sd2 = score_dig(q0, vecs_2bit_f16_0, cd, pid2tok)
    saf = score_ana(q1, vecs_f16_1,      ca, pid2tok)
    sa2 = score_ana(q1, vecs_2bit_f16_1, ca, pid2tok)

    margin=None
    if tp in sdf and len(sdf)>=2:
        sv=sorted(sdf.values(),reverse=True); margin=sv[0]-sv[1]

    results.append({
        'query_idx':q_idx, 'qid':meta.loc[q_idx,'qid'], 'true_pid':tp,
        'n_cands_dig':len(sdf), 'n_cands_ana':len(saf),
        'in_dig':tp in sdf,  'in_ana':tp in saf,
        'rank_dig_f32':get_rank(sdf,tp,False), 'rank_dig_2bt':get_rank(sd2,tp,False),
        'rank_ana_f32':get_rank(saf,tp,True),  'rank_ana_2bt':get_rank(sa2,tp,True),
        'score_dig_f32':sdf.get(tp), 'score_dig_2bt':sd2.get(tp),
        'curr_ana_f32': saf.get(tp), 'curr_ana_2bt': sa2.get(tp),
        'margin':margin,
    })

    if (q_idx+1) % 50 == 0:
        el=time.time()-t0; eta=el/(q_idx+1)*(500-q_idx-1)
        print(f"  [{q_idx+1:3d}/500]  {el:.0f}s  ETA:{eta:.0f}s")
        sys.stdout.flush()

print(f"\n  완료: {time.time()-t0:.1f}s")
df=pd.DataFrame(results)
df.to_csv(f'{BASE}/phase3_results.csv', index=False)
print(f"저장: phase3_results.csv ({len(df)} rows)")
print("Phase 3 완료!")
