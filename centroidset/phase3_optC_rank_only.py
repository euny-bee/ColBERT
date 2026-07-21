"""
Phase 3 (rank-only). Option C 랭킹 단계만의 PBS 영향 분리 측정
  - Option C 자체 Step2/Step3(centroid 선택, 후보 검색)는 사용하지 않음
  - 대신 Option A(VthComp)가 찾은 깨끗한 후보 집합(cands_optA) 위에서
    Option C의 Step6(MaxSim, dead-zone 전류 기반 랭킹)만 실행
  - 후보 검색 문제와 랭킹 문제의 기여도를 분리하기 위함
  출력: phase3_optC_rank_only_results.csv  (255 x 4 = 1020 rows)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
from scipy.special import erf
import time

BASE   = r'C:\Users\nmdl-khb\ColBERT\centroidset'
NPROBE = 2
SEED   = 42
DEV    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)

print(f"Device: {DEV}")

VTH_CONDS = [
    ('v2', 0.0, 1.0,  0.30),
    ('v3', 0.0, 2.0,  0.60),
    ('v4', 0.0, 3.0,  0.90),
]

# ==========================================================================
# 셀 모델 파라미터 (die4, VDD=1.7V) -- phase3_vth_pipeline.py와 동일
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
# GPU 샘플러
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
print(f"  Q:{Q_np.shape}  C:{C_np.shape}  vecs:{vecs.shape}  passages:{len(pid2tok):,}")
print(f"  GPU mem: {torch.cuda.memory_allocated(DEV)/1e9:.2f} GB", flush=True)

# ==========================================================================
# Step 2 -- Option A (후보 집합 확보용, Vth-independent라 1회만)
# ==========================================================================
CBATCH_S2 = 2048

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

def step3(top_q):
    pids = set()
    for c_ids in top_q:
        for c in c_ids:
            arr = ivf_pids[int(c)]
            if len(arr) > 0:
                pids.update(arr.tolist())
    return list(pids)

# ==========================================================================
# Step 6 -- Option C (cands_optA 위에서 실행)
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

def s6_optC(q_idx, cands, vm0_table, vm3_table):
    """vm0_table/vm3_table: (NT, 128) -- 셀(토큰 x 차원)에 고정된 Vth, 전 쿼리 공통 재사용"""
    Qq    = Q_gpu[q_idx]
    tp    = int(true_pids[q_idx])
    valid = _prep_cands(cands)
    if not valid: return 0, None

    # 쿼리당 1회만 host->GPU 변환 + gather (배치마다 반복하면 10x 느려짐)
    pids_all   = [p for p, _ in valid]
    counts_all = [len(t) for _, t in valid]
    all_tok    = np.concatenate([t for _, t in valid])
    tok_idx    = torch.as_tensor(all_tok, device=DEV, dtype=torch.long)
    D_all      = vecs_gpu[tok_idx]
    vm0_all    = vm0_table[tok_idx]
    vm3_all    = vm3_table[tok_idx]

    scores, pids = [], []
    start = 0
    for bi in range(0, len(valid), PBATCH):
        counts = counts_all[bi:bi+PBATCH]
        pids_b = pids_all[bi:bi+PBATCH]
        n_t    = sum(counts)
        D   = D_all[start:start+n_t]
        vm0 = vm0_all[start:start+n_t].unsqueeze(0)          # (1, n_t, 128): 차원별 독립, 쿼리 무관 고정
        vm3 = vm3_all[start:start+n_t].unsqueeze(0)
        d   = Qq[:, None, :] - D[None, :, :]                  # (32, n_t, 128)
        I   = cell_C_t(d, vm0, vm3).sum(dim=-1)
        s2 = 0
        for pid, n in zip(pids_b, counts):
            scores.append(float(I[:, s2:s2+n].min(dim=1).values.sum()))
            pids.append(pid)
            s2 += n
        start += n_t
    return _rank_score(scores, pids, tp, ascending=True)

# ==========================================================================
# 실행
# ==========================================================================
print()
print("=" * 60)

print("Step 2+3 -- Option A (candidate set for rank-only experiment)...", flush=True)
t1 = time.time()
top_optA   = s2_analog(cell_A_t)
cands_optA = [step3(top_optA[q]) for q in range(NQ)]
print(f"  done ({time.time()-t1:.1f}s)")

n_cands_optA = [len(c) for c in cands_optA]
print(f"  avg candidates/query: {np.mean(n_cands_optA):.1f}  "
      f"(min={np.min(n_cands_optA)}, max={np.max(n_cands_optA)})")

NT = vecs_gpu.shape[0]
DIM = vecs_gpu.shape[1]
print(f"  token cells: {NT:,} x {DIM} dims")

results = []

for cname, lo, hi, std in VTH_CONDS:
    print()
    print("=" * 60)
    print(f"Vth {cname}: shift [{lo:.1f},{hi:.1f}]V  "
          f"actual [{Vth_orig+lo:.3f},{Vth_orig+hi:.3f}]V  std={std}", flush=True)
    sampler = make_sampler_gpu(lo, hi, std)

    # 셀(토큰 x 차원) 고정 Vth 테이블 -- 이 PBS 조건 안에서는 모든 쿼리가 동일한 값을 공유
    vm0_table = sampler((NT, DIM))
    vm3_table = sampler((NT, DIM))

    print("  Step 6 -- Option C (rank-only, using Option A candidates)...", flush=True)
    t1 = time.time()
    optC_res = []
    for q in range(NQ):
        optC_res.append(s6_optC(q, cands_optA[q], vm0_table, vm3_table))
        if (q + 1) % 64 == 0:
            print(f"    {q+1}/{NQ} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  done ({time.time()-t1:.1f}s)")

    del vm0_table, vm3_table
    torch.cuda.empty_cache()

    for q in range(NQ):
        n_c, r_c = optC_res[q]
        results.append({
            'query_idx':           q,
            'vth_cond':            cname,
            'vth_shift_hi':        hi,
            'true_pid':            int(true_pids[q]),
            'n_cands':             n_c,
            'rank_optC_rankonly':  r_c,
        })

df = pd.DataFrame(results)
out = f'{BASE}/phase3_optC_rank_only_results.csv'
if os.path.exists(out):
    df_old = pd.read_csv(out)
    df_old = df_old[~df_old['vth_cond'].isin(df['vth_cond'].unique())]
    df = pd.concat([df_old, df], ignore_index=True)
df.to_csv(out, index=False)

print()
print("=" * 60)
print(f"Saved: {out}  ({len(df)} rows)")
print(f"Total: {time.time()-t_start:.1f}s")
print("Phase 3 (rank-only) done!")
