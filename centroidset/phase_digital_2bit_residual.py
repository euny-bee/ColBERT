"""
Digital baseline + 2bit residual 압축 (ColBERTv2/PLAID 방식)
  - Step 2/3 (centroid 선택, 후보 검색): 기존 Digital과 동일 (압축 영향 없음, 실제 PLAID도 동일)
  - Step 6 (MaxSim 랭킹): 문서 토큰 벡터를 "centroid + 2bit 양자화 residual"로 복원한 값으로 계산
  비교 대상: phase3_vth_results.csv 의 rank_dig(float32), rank_optA(Vth-comp 아날로그)
  출력: phase_digital_2bit_results.csv, 콘솔에 비교 테이블
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import time

BASE   = r'C:\Users\nmdl-khb\ColBERT\centroidset'
NPROBE = 2
NBITS  = 2
N_BUCKETS = 2 ** NBITS          # 4
SEED   = 42
DEV    = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEV}")

# ==========================================================================
# 데이터 로드 (phase3_vth_pipeline.py와 동일)
# ==========================================================================
print("=" * 60)
print("Load data...")
t_start = time.time()

Q_np     = torch.load(f'{BASE}/scale_query_embs_255x32x128.pt').float().numpy()
C_np     = np.load(f'{BASE}/scale_centroids_2k.npy').astype(np.float32)
vecs     = np.load(f'{BASE}/scale_all_vectors.npy').astype(np.float32)
ivf_pids = np.load(f'{BASE}/scale_ivf_pids_2k.npy', allow_pickle=True)
pass_t   = np.load(f'{BASE}/scale_passage_of_token.npy')
meta     = pd.read_csv(f'{BASE}/scale_query_meta.csv')

order = np.argsort(pass_t, kind='stable')
sp = pass_t[order]
up, starts = np.unique(sp, return_index=True)
ends = np.concatenate([starts[1:], [len(pass_t)]])
pid2tok = {int(p): order[s:e] for p, s, e in zip(up, starts, ends)}

true_pids = meta['true_pid'].tolist()
NQ = len(Q_np)
NC = len(C_np)

Q_gpu    = torch.tensor(Q_np, dtype=torch.float32, device=DEV)
C_gpu    = torch.tensor(C_np, dtype=torch.float32, device=DEV)
vecs_gpu = torch.tensor(vecs, dtype=torch.float32, device=DEV)

print(f"  done ({time.time()-t_start:.1f}s)")
print(f"  Q:{Q_np.shape}  C:{C_np.shape}  vecs:{vecs.shape}  passages:{len(pid2tok):,}  centroids:{NC}")
print(f"  GPU mem: {torch.cuda.memory_allocated(DEV)/1e9:.2f} GB", flush=True)

# ==========================================================================
# Step 2 + 3 -- Digital (full precision query-centroid dot product)
# ==========================================================================
print()
print("=" * 60)
print("Step 2+3 -- Digital candidate set...", flush=True)
t1 = time.time()

scores = Q_gpu @ C_gpu.T
top_dig = torch.topk(scores, NPROBE, dim=2, largest=True).indices.cpu().numpy().astype(np.int32)

def step3(top_q):
    pids = set()
    for c_ids in top_q:
        for c in c_ids:
            arr = ivf_pids[int(c)]
            if len(arr) > 0:
                pids.update(arr.tolist())
    return list(pids)

cands_dig = [step3(top_dig[q]) for q in range(NQ)]
n_cands = [len(c) for c in cands_dig]
print(f"  done ({time.time()-t1:.1f}s)")
print(f"  avg candidates/query: {np.mean(n_cands):.1f}  (min={np.min(n_cands)}, max={np.max(n_cands)})")

# ==========================================================================
# 2bit residual 코덱 구축 (centroid는 동일한 2048개 재사용)
#   1) 모든 토큰을 가장 가까운 centroid에 할당
#   2) residual = vec - centroid
#   3) residual 표본으로 bucket cutoffs/weights 학습 (전 차원 공통, 공식 ColBERT 방식)
#   4) 전체 residual 양자화 -> 복원
# ==========================================================================
print()
print("=" * 60)
print("2bit residual codec 구축...", flush=True)
t1 = time.time()

N_TOK = vecs_gpu.shape[0]
ASSIGN_BATCH = 200_000
assignment = torch.empty(N_TOK, dtype=torch.long, device=DEV)
for s in range(0, N_TOK, ASSIGN_BATCH):
    e = min(s + ASSIGN_BATCH, N_TOK)
    d2 = torch.cdist(vecs_gpu[s:e], C_gpu)          # (batch, 2048)
    assignment[s:e] = d2.argmin(dim=1)
print(f"  centroid 할당 완료 ({time.time()-t1:.1f}s)")

# 표본으로 cutoffs/weights 학습 (공식 ColBERT: 전 차원 공통 1D 분포)
# 전체 residual(N_TOK x 128)을 한번에 GPU에 올리지 않고, 무작위 token 표본만 추출
SAMPLE_TOK = min(20_000, N_TOK)
sample_idx = torch.randint(0, N_TOK, (SAMPLE_TOK,), device=DEV)
sample_resid = vecs_gpu[sample_idx] - C_gpu[assignment[sample_idx]]   # (SAMPLE_TOK, 128)
sample = sample_resid.reshape(-1)                                      # (SAMPLE_TOK*128,)

quantiles = torch.arange(1, N_BUCKETS, device=DEV).float() / N_BUCKETS   # [0.25,0.5,0.75]
bucket_cutoffs = torch.quantile(sample.float(), quantiles)               # (3,)

bucket_idx_sample = torch.bucketize(sample, bucket_cutoffs)              # 0..3
bucket_weights = torch.zeros(N_BUCKETS, device=DEV)
for b in range(N_BUCKETS):
    mask = bucket_idx_sample == b
    if mask.any():
        bucket_weights[b] = sample[mask].mean()
    else:
        bucket_weights[b] = bucket_cutoffs[min(b, len(bucket_cutoffs)-1)]

print(f"  bucket_cutoffs : {bucket_cutoffs.cpu().numpy()}")
print(f"  bucket_weights : {bucket_weights.cpu().numpy()}")

# 전체 residual 양자화 + 복원 (배치, residual을 매번 즉석 계산해 메모리 절약)
recon_vecs_gpu = torch.empty_like(vecs_gpu)
QUANT_BATCH = 200_000
for s in range(0, N_TOK, QUANT_BATCH):
    e = min(s + QUANT_BATCH, N_TOK)
    centroid_b = C_gpu[assignment[s:e]]                       # (batch, 128)
    residual_b = vecs_gpu[s:e] - centroid_b                   # (batch, 128)
    bidx = torch.bucketize(residual_b, bucket_cutoffs)        # (batch, 128)
    resid_hat = bucket_weights[bidx]                          # (batch, 128)
    recon_vecs_gpu[s:e] = centroid_b + resid_hat

err_sum, norm_sum, n_seen = 0.0, 0.0, 0
for s in range(0, N_TOK, QUANT_BATCH):
    e = min(s + QUANT_BATCH, N_TOK)
    err_sum  += (recon_vecs_gpu[s:e] - vecs_gpu[s:e]).norm(dim=-1).sum().item()
    norm_sum += vecs_gpu[s:e].norm(dim=-1).sum().item()
    n_seen   += (e - s)
print(f"  복원 완료 ({time.time()-t1:.1f}s total)")
print(f"  평균 L2 복원 오차: {err_sum/n_seen:.4f}  "
      f"(원본 평균 norm: {norm_sum/n_seen:.4f}, "
      f"상대오차: {err_sum/norm_sum*100:.2f}%)")
print(f"  압축률: {NBITS}bit/dim vs 원본 32bit/dim float -> 이론상 {32/NBITS:.0f}x (centroid id 오버헤드 별도)")

# ==========================================================================
# Step 6 -- MaxSim 랭킹 (복원된 2bit 벡터 사용, query는 full precision)
# ==========================================================================
PBATCH = 50

def _prep_cands(cands):
    return [(int(p), pid2tok[int(p)]) for p in cands
            if pid2tok.get(int(p)) is not None and len(pid2tok[int(p)]) > 0]

def _rank_score(scores_list, pid_list, true_pid, ascending):
    if true_pid not in pid_list:
        return len(scores_list), None
    vals  = torch.tensor(scores_list, device=DEV)
    t_val = vals[pid_list.index(true_pid)]
    if ascending:
        rank = int((vals < t_val).sum()) + 1
    else:
        rank = int((vals > t_val).sum()) + 1
    return len(scores_list), rank

def s6_digital_2bit(q_idx, cands, doc_vecs):
    Qq    = Q_gpu[q_idx]
    tp    = int(true_pids[q_idx])
    valid = _prep_cands(cands)
    if not valid: return 0, None
    scores, pids = [], []
    for bi in range(0, len(valid), PBATCH):
        batch   = valid[bi:bi+PBATCH]
        pids_b  = [p for p, _ in batch]
        counts  = [len(t) for _, t in batch]
        all_tok = np.concatenate([t for _, t in batch])
        D   = doc_vecs[all_tok]          # (n_t, 128)
        sim = Qq @ D.T                   # (32, n_t)
        start = 0
        for pid, n in zip(pids_b, counts):
            scores.append(float(sim[:, start:start+n].max(dim=1).values.sum()))
            pids.append(pid)
            start += n
    return _rank_score(scores, pids, tp, ascending=False)

print()
print("=" * 60)
print("Step 6 -- Digital (2bit residual 복원 벡터)...", flush=True)
t1 = time.time()
results = []
for q in range(NQ):
    n_c, r = s6_digital_2bit(q, cands_dig[q], recon_vecs_gpu)
    results.append({
        'query_idx': q,
        'true_pid':  int(true_pids[q]),
        'n_cands':   n_c,
        'rank_dig_2bit': r,
    })
    if (q + 1) % 64 == 0:
        print(f"    {q+1}/{NQ} ({time.time()-t1:.1f}s)", flush=True)
print(f"  done ({time.time()-t1:.1f}s)")

df = pd.DataFrame(results)
out = f'{BASE}/phase_digital_2bit_results.csv'
df.to_csv(out, index=False)
print(f"  저장: {out}")

# ==========================================================================
# 지표 계산 + 비교 (Digital float32 / Option A 는 기존 phase3_vth_results.csv 사용)
# ==========================================================================
def mrr_at_k(ranks, k=10):
    return float(np.mean([1/r if (r is not None and not np.isnan(float(r)) and r <= k) else 0.0
                          for r in ranks]))
def r_at_k(ranks, k):
    return float(np.mean([1 if (r is not None and not np.isnan(float(r)) and r <= k) else 0.0
                          for r in ranks]))
def ndcg_at_k(ranks, k=10):
    return float(np.mean([1/np.log2(r+1) if (r is not None and not np.isnan(float(r)) and r <= k) else 0.0
                          for r in ranks]))
def metrics(ranks):
    return {
        'MRR@10':  round(mrr_at_k(ranks, 10), 4),
        'nDCG@10': round(ndcg_at_k(ranks, 10), 4),
        'R@50':    round(r_at_k(ranks, 50),   4),
        'R@1k':    round(r_at_k(ranks, 1000), 4),
    }

ref = pd.read_csv(f'{BASE}/phase3_vth_results.csv')
ref_v1 = ref[ref['vth_cond'] == 'v1']

m_dig_f32 = metrics(ref_v1['rank_dig'].tolist())
m_optA    = metrics(ref_v1['rank_optA'].tolist())
m_dig_2b  = metrics(df['rank_dig_2bit'].tolist())

print()
print("=" * 78)
print(f"{'Method':<30} {'MRR@10':>7} {'nDCG@10':>8} {'R@50':>6} {'R@1k':>6}")
print("-" * 78)
for name, m in [('Digital (float32)', m_dig_f32),
                ('Digital (2bit residual)', m_dig_2b),
                ('Option A (VthComp, analog)', m_optA)]:
    print(f"{name:<30} {m['MRR@10']:>7.4f} {m['nDCG@10']:>8.4f} {m['R@50']:>6.4f} {m['R@1k']:>6.4f}")
print("=" * 78)

print()
print(f"후보 검색률(candidate recall): Digital(2bit) = {(df['rank_dig_2bit'].notna()).mean():.3f}  "
      f"(Step2/3는 압축 영향 없음, float32 Digital과 동일)")

print()
print(f"Total: {time.time()-t_start:.1f}s")
print("Digital 2bit residual 압축 비교 완료!")
