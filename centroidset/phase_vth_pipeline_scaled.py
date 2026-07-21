"""
Phase 3 Vth 변이 평가 파이프라인 (pool 크기 가변판)
  Dataset : MS MARCO 255 queries x {5k, 20k} doc pool (phase_reduce_pool.py 산출물)
  Methods : Digital | Option A (VthComp) | Option C (NoComp)
  Vth 조건 : v0a[0,0.2] / v0b[0,0.3] / v1[0,0.5] / v2[0,1] / v3[0,2] / v4[0,3] V

  phase3_vth_pipeline.py와의 차이 (Option C에만 적용, Digital/Option A는 무관):
    - 기존: Step2/Step6 모두 "매 쿼리·매 배치마다 새로 샘플링" + "128차원이 한 Vth 값 공유"
    - 변경: PBS 조건당 1회만 샘플링한 (centroid/token, dim)별 고정 Vth 테이블을 만들어
            전 쿼리(255개)가 동일한 값을 재사용. 차원(128개)도 서로 독립적으로 샘플링.
            -> "물리적으로 그 소자의 Vth가 PBS로 인해 고정 변형됨"을 반영하는 모델.
      Step2(centroid 후보검색): vm0_centroid/vm3_centroid (NC, DIM)
      Step6(문서 토큰 랭킹)   : vm0_table/vm3_table       (NT, DIM)

  사용법: python phase_vth_pipeline_scaled.py <suffix>
    예)   python phase_vth_pipeline_scaled.py 5k
          python phase_vth_pipeline_scaled.py 20k
  출력: phase3_vth_results_{suffix}.csv
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import torch
from scipy.special import erf
import time

BASE   = r'C:\Users\nmdl-khb\ColBERT\centroidset'
SUFFIX = sys.argv[1]
NPROBE = 2
SEED   = 42
DEV    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)

print(f"Device: {DEV}")
print(f"Pool suffix: {SUFFIX}")

VTH_CONDS = [
    ('v0a', 0.0, 0.2, 0.06),
    ('v0b', 0.0, 0.3, 0.09),
    ('v1',  0.0, 0.5, 0.15),
    ('v2',  0.0, 1.0, 0.30),
    ('v3',  0.0, 2.0, 0.60),
    ('v4',  0.0, 3.0, 0.90),
]

# ==========================================================================
# 셀 모델 파라미터 (die4, VDD=1.7V)
# ==========================================================================
Vth_orig = 0.151045
L_f, K_f, B_f = 6.112020, 2.731949, -10.713911
VSAT, SLOPE, VDD, CLIP = 2.88, 4.0332e-5, 1.7, 1e-14
VSAT_OD = VSAT - Vth_orig

def I_single_t(vgs, vth_val):
    if isinstance(vth_val, float):
        vsat_abs = vth_val + VSAT_OD
        v = torch.clamp(vgs, max=vsat_abs)
        return (10 ** (B_f + L_f / (1 + torch.exp(-K_f * (v - vth_val))))
                + torch.clamp(vgs - vsat_abs, min=0.0) * SLOPE)
    else:
        vsat_abs = vth_val + VSAT_OD
        v = torch.minimum(vgs, vsat_abs)
        return (10 ** (B_f + L_f / (1 + torch.exp(-K_f * (v - vth_val))))
                + torch.clamp(vgs - vsat_abs, min=0.0) * SLOPE)

def cell_A_t(d):
    """Option A: vth_stored = Vth_orig -> Vth 완전 상쇄"""
    vgd0 = d + Vth_orig
    I0 = torch.where(d >= 0,
                     torch.clamp(I_single_t(vgd0, Vth_orig) - I_single_t(vgd0 - VDD, Vth_orig), min=0.0),
                     torch.zeros_like(d))
    vgd3 = -d + Vth_orig
    I3 = torch.where(d < 0,
                     torch.clamp(I_single_t(vgd3, Vth_orig) - I_single_t(vgd3 - VDD, Vth_orig), min=0.0),
                     torch.zeros_like(d))
    return torch.clamp(I0 + I3, min=CLIP)

def cell_C_t(d, vm0, vm3):
    """Option C: vth_stored = 0 -> dead zone [0, vth]"""
    I0 = torch.where(d >= 0,
                     torch.clamp(I_single_t(d, vm0) - I_single_t(d - VDD, vm0), min=0.0),
                     torch.zeros_like(d))
    I3 = torch.where(d < 0,
                     torch.clamp(I_single_t(-d, vm3) - I_single_t(-d - VDD, vm3), min=0.0),
                     torch.zeros_like(d))
    return torch.clamp(I0 + I3, min=CLIP)

# ==========================================================================
# GPU 샘플러: torch.erfinv 역 CDF
# ==========================================================================
def make_sampler_gpu(lo, hi, std):
    _sqrt2 = float(np.sqrt(2))
    Phi_a = float(0.5 * (1 + erf(lo / (std * _sqrt2))))
    Phi_b = float(0.5 * (1 + erf(hi / (std * _sqrt2))))

    def _sample(shape):
        u = torch.rand(shape, device=DEV) * (Phi_b - Phi_a) + Phi_a
        u = u.clamp(1e-6, 1 - 1e-6)
        shift = std * _sqrt2 * torch.erfinv(2 * u - 1)
        return Vth_orig + shift

    return _sample

# ==========================================================================
# 데이터 로드
# ==========================================================================
print("=" * 60)
print("Load data...")
t_start = time.time()

Q_np     = torch.load(f'{BASE}/scale_query_embs_255x32x128.pt').float().numpy()
C_np     = np.load(f'{BASE}/scale_centroids_{SUFFIX}.npy').astype(np.float32)
vecs     = np.load(f'{BASE}/scale_all_vectors_{SUFFIX}.npy').astype(np.float32)
ivf_pids = np.load(f'{BASE}/scale_ivf_pids_{SUFFIX}.npy', allow_pickle=True)
pass_t   = np.load(f'{BASE}/scale_passage_of_token_{SUFFIX}.npy')
meta     = pd.read_csv(f'{BASE}/scale_query_meta.csv')

order = np.argsort(pass_t, kind='stable')
sp = pass_t[order]
up, starts = np.unique(sp, return_index=True)
ends = np.concatenate([starts[1:], [len(pass_t)]])
pid2tok = {int(p): order[s:e] for p, s, e in zip(up, starts, ends)}

true_pids = meta['true_pid'].tolist()
NQ = len(Q_np)
NC = len(C_np)
DIM = C_np.shape[1]

Q_gpu    = torch.tensor(Q_np, dtype=torch.float32, device=DEV)
C_gpu    = torch.tensor(C_np, dtype=torch.float32, device=DEV)
vecs_gpu = torch.tensor(vecs, dtype=torch.float32, device=DEV)
NT       = vecs_gpu.shape[0]

print(f"  done ({time.time()-t_start:.1f}s)")
print(f"  Q:{Q_np.shape}  C:{C_np.shape}  vecs:{vecs.shape}  passages:{len(pid2tok):,}")
print(f"  NC(centroids)={NC}  NT(doc tokens)={NT:,}")
print(f"  GPU mem: {torch.cuda.memory_allocated(DEV)/1e9:.2f} GB", flush=True)

# ==========================================================================
# Step 2
# ==========================================================================
CBATCH_S2 = 2048

def s2_digital():
    scores = Q_gpu @ C_gpu.T
    return torch.topk(scores, NPROBE, dim=2, largest=True).indices.cpu().numpy().astype(np.int32)

def s2_analog(cell_fn):
    top = np.empty((NQ, 32, NPROBE), dtype=np.int32)
    for q in range(NQ):
        I_q = torch.empty(32, NC, device=DEV)
        Qq = Q_gpu[q]
        for cb in range(0, NC, CBATCH_S2):
            ce = min(cb + CBATCH_S2, NC)
            d = Qq[:, None, :] - C_gpu[cb:ce][None, :, :]
            I_q[:, cb:ce] = cell_fn(d).sum(dim=-1)
        top[q] = torch.topk(I_q, NPROBE, dim=1, largest=False).indices.cpu().numpy()
        if (q + 1) % 64 == 0:
            print(f"    {q+1}/{NQ}", flush=True)
    return top

def s2_optC(vm0_centroid, vm3_centroid):
    """vm0_centroid/vm3_centroid: (NC, DIM) -- centroid-dim 고정 Vth, 전 쿼리 공통 재사용"""
    top = np.empty((NQ, 32, NPROBE), dtype=np.int32)
    for q in range(NQ):
        I_q = torch.empty(32, NC, device=DEV)
        Qq = Q_gpu[q]
        for cb in range(0, NC, CBATCH_S2):
            ce = min(cb + CBATCH_S2, NC)
            d   = Qq[:, None, :] - C_gpu[cb:ce][None, :, :]   # (32, size, 128)
            vm0 = vm0_centroid[cb:ce].unsqueeze(0)             # (1, size, 128)
            vm3 = vm3_centroid[cb:ce].unsqueeze(0)
            I_q[:, cb:ce] = cell_C_t(d, vm0, vm3).sum(dim=-1)
        top[q] = torch.topk(I_q, NPROBE, dim=1, largest=False).indices.cpu().numpy()
        if (q + 1) % 64 == 0:
            print(f"    {q+1}/{NQ}", flush=True)
    return top

# ==========================================================================
# Step 3
# ==========================================================================
def step3(top_q):
    pids = set()
    for c_ids in top_q:
        for c in c_ids:
            arr = ivf_pids[int(c)]
            if len(arr) > 0:
                pids.update(arr.tolist())
    return list(pids)

# ==========================================================================
# Step 6
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

def s6_digital(q_idx, cands):
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
        D   = vecs_gpu[all_tok]
        sim = Qq @ D.T
        start = 0
        for pid, n in zip(pids_b, counts):
            scores.append(float(sim[:, start:start+n].max(dim=1).values.sum()))
            pids.append(pid)
            start += n
    return _rank_score(scores, pids, tp, ascending=False)

def s6_optA(q_idx, cands):
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
        D = vecs_gpu[all_tok]
        d = Qq[:, None, :] - D[None, :, :]
        I = cell_A_t(d).sum(dim=-1)
        start = 0
        for pid, n in zip(pids_b, counts):
            scores.append(float(I[:, start:start+n].min(dim=1).values.sum()))
            pids.append(pid)
            start += n
    return _rank_score(scores, pids, tp, ascending=True)

def s6_optC(q_idx, cands, vm0_table, vm3_table):
    """vm0_table/vm3_table: (NT, DIM) -- 토큰-dim 고정 Vth, 전 쿼리 공통 재사용"""
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
        D   = vecs_gpu[all_tok]
        d   = Qq[:, None, :] - D[None, :, :]                # (32, n_t, 128)
        tok_idx = torch.as_tensor(all_tok, device=DEV, dtype=torch.long)
        vm0 = vm0_table[tok_idx].unsqueeze(0)                # (1, n_t, 128)
        vm3 = vm3_table[tok_idx].unsqueeze(0)
        I   = cell_C_t(d, vm0, vm3).sum(dim=-1)
        start = 0
        for pid, n in zip(pids_b, counts):
            scores.append(float(I[:, start:start+n].min(dim=1).values.sum()))
            pids.append(pid)
            start += n
    return _rank_score(scores, pids, tp, ascending=True)

# ==========================================================================
# 실행
# ==========================================================================
print()
print("=" * 60)

print("Step 2 -- Digital...", flush=True)
t1 = time.time()
top_dig = s2_digital()
print(f"  done ({time.time()-t1:.1f}s)")

print("Step 2 -- Option A (Vth-independent, 1 run)...", flush=True)
t1 = time.time()
top_optA = s2_analog(cell_A_t)
print(f"  done ({time.time()-t1:.1f}s)")

print("Step 3 -- Digital + Option A...", flush=True)
cands_dig  = [step3(top_dig[q])  for q in range(NQ)]
cands_optA = [step3(top_optA[q]) for q in range(NQ)]

print("Step 6 -- Digital + Option A...", flush=True)
t1 = time.time()
dig_res, optA_res = [], []
for q in range(NQ):
    dig_res.append(s6_digital(q, cands_dig[q]))
    optA_res.append(s6_optA(q, cands_optA[q]))
    if (q + 1) % 64 == 0:
        print(f"  {q+1}/{NQ} ({time.time()-t1:.1f}s)", flush=True)
print(f"  done ({time.time()-t1:.1f}s)")

# -- Option C: 6 Vth conditions, 셀-고정/차원독립 모델 --
results = []

for cname, lo, hi, std in VTH_CONDS:
    print()
    print("=" * 60)
    print(f"Vth {cname}: shift [{lo:.2f},{hi:.2f}]V  "
          f"actual [{Vth_orig+lo:.3f},{Vth_orig+hi:.3f}]V  std={std}", flush=True)
    sampler = make_sampler_gpu(lo, hi, std)

    # Step2용 centroid-dim 고정 Vth 테이블 (조건당 1회, 전 쿼리 공유)
    vm0_centroid = sampler((NC, DIM))
    vm3_centroid = sampler((NC, DIM))

    print("  Step 2 -- Option C...", flush=True)
    t1 = time.time()
    top_optC   = s2_optC(vm0_centroid, vm3_centroid)
    cands_optC = [step3(top_optC[q]) for q in range(NQ)]
    print(f"  done ({time.time()-t1:.1f}s)")

    del vm0_centroid, vm3_centroid
    torch.cuda.empty_cache()

    # Step6용 token-dim 고정 Vth 테이블 (조건당 1회, 전 쿼리 공유)
    vm0_table = sampler((NT, DIM))
    vm3_table = sampler((NT, DIM))

    print("  Step 6 -- Option C...", flush=True)
    t1 = time.time()
    optC_res = []
    for q in range(NQ):
        optC_res.append(s6_optC(q, cands_optC[q], vm0_table, vm3_table))
        if (q + 1) % 64 == 0:
            print(f"    {q+1}/{NQ} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  done ({time.time()-t1:.1f}s)")

    del vm0_table, vm3_table
    torch.cuda.empty_cache()

    for q in range(NQ):
        n_d, r_d = dig_res[q]
        n_a, r_a = optA_res[q]
        n_c, r_c = optC_res[q]
        results.append({
            'query_idx':    q,
            'vth_cond':     cname,
            'vth_shift_hi': hi,
            'true_pid':     int(true_pids[q]),
            'n_cands_dig':  n_d,
            'n_cands_optA': n_a,
            'n_cands_optC': n_c,
            'rank_dig':     r_d,
            'rank_optA':    r_a,
            'rank_optC':    r_c,
        })

df = pd.DataFrame(results)
out = f'{BASE}/phase3_vth_results_{SUFFIX}.csv'
df.to_csv(out, index=False)

print()
print("=" * 60)
print(f"Saved: {out}  ({len(df)} rows)")
print(f"Total: {time.time()-t_start:.1f}s")
print("Phase 3 Vth (scaled) done!")
