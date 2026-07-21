"""
newq_margin(200 docs, query당 rank2 경쟁문서 1개 강제포함) pool 고정, Vth SEED만 0~39로
바꿔가며 Option C(No Comp)의 true passage rank를 40회 반복 측정.

Digital과 Option A는 Vth 랜덤 샘플링에 의존하지 않으므로(Option A는 고정 Vth_orig만 사용) 한 번만 계산.

v2: ivf lookup / doc token lookup을 dict 사전계산으로 벡터화 (기존 pandas 루프 제거),
    seed마다 CSV에 즉시 append + flush하여 진행상황을 실시간으로 확인 가능.
"""

import sys
import csv
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
NPROBE         = 2
QUERY_RELEVANT = {0: 471602, 1: 822108, 2: 347885}
N_SEEDS        = 40
OUT_CSV        = f'{BASE}/seed_sweep_optC_results.csv'

# ==========================================================================
# 셀 모델 파라미터 (die4, VDD=1.7V)
# ==========================================================================
L_f    =  6.112020
K_f    =  2.731949
B_f    = -10.713911
VSAT   =  2.88
SLOPE  =  4.0332e-5
VDD    =  1.7
CLIP   =  1e-14
Vth_orig = 0.151045
VSAT_OD  = VSAT - Vth_orig

def I_single(vgs, vth):
    vsat_abs = vth + VSAT_OD
    v = np.minimum(vgs, vsat_abs)
    return (10 ** (B_f + L_f / (1 + np.exp(-K_f * (v - vth))))
            + np.maximum(0.0, vgs - vsat_abs) * SLOPE)

def _cell_current(d, vth_m0, vth_m3, vth_s_m0, vth_s_m3):
    vgd_m0 = d + vth_s_m0
    I_m0   = np.where(d >= 0,
                      np.maximum(I_single(vgd_m0, vth_m0)
                                 - I_single(vgd_m0 - VDD, vth_m0), 0.0),
                      0.0)
    vgd_m3 = -d + vth_s_m3
    I_m3   = np.where(d < 0,
                      np.maximum(I_single(vgd_m3, vth_m3)
                                 - I_single(vgd_m3 - VDD, vth_m3), 0.0),
                      0.0)
    return np.clip(I_m0 + I_m3, CLIP, None) * 1e9   # nA

def cell_optA(d, vth_m0, vth_m3):
    return _cell_current(d, vth_m0, vth_m3, vth_s_m0=vth_m0, vth_s_m3=vth_m3)

def cell_optC(d, vth_m0, vth_m3):
    return _cell_current(d, vth_m0, vth_m3, vth_s_m0=0.0, vth_s_m3=0.0)

VTH_STD = 0.90
VTH_LO, VTH_HI = 0.0, 3.0
_a = (VTH_LO - 0.0) / VTH_STD
_b = (VTH_HI - 0.0) / VTH_STD

def sample_vth(shape, rng):
    shift = truncnorm.rvs(_a, _b, loc=0.0, scale=VTH_STD, size=shape, random_state=rng)
    return Vth_orig + shift

def log(msg):
    print(msg, flush=True)

# ==========================================================================
# 데이터 로드 (pool 고정 -- 한 번만)
# ==========================================================================
log("데이터 로드 중...")
Q_df = pd.read_excel(f'{BASE}/[clip99.9]query_embs_96x128.xlsx',  index_col=0)
C_df = pd.read_excel(f'{BASE}/[clip99.9]centroids_100x128.xlsx',  index_col=0)
doc  = pd.read_excel(f'{BASE}/[clip99.9]doc_embs_13007x128.xlsx', index_col=0)
ivf  = pd.read_excel(f'{BASE}/[clip99.9]ivf_centroid2pid.xlsx',
                     sheet_name='long_format', index_col=0)
qcr  = pd.read_excel(f'{BASE}/[clip99.9]query_centroid_ranking.xlsx',
                     sheet_name='rank_order', index_col=0)

dim_cols = [c for c in doc.columns if c.startswith('dim_')]
Q = Q_df.values.astype(float)
C = C_df.values.astype(float)
D = doc[dim_cols].values.astype(float)

rank_cols      = [f'rank_{i+1}' for i in range(100)]
rank_order_arr = qcr[rank_cols].values

NC = C.shape[0]
NT = D.shape[0]
DIM = D.shape[1]
log(f"  NC={NC}  NT={NT}  DIM={DIM}")

# --- 사전계산 (seed 무관, 한 번만): centroid -> pid 목록, pid -> 토큰 인덱스 배열 ---
log("사전계산 중 (centroid->pid, pid->token idx)...")
centroid_to_pids = {c_id: grp['pid'].to_numpy() for c_id, grp in ivf.groupby(ivf.index)}

pid_to_idxs = {}
for i, tok in enumerate(doc.index):
    pid = int(str(tok).rsplit('_t', 1)[0])
    pid_to_idxs.setdefault(pid, []).append(i)
pid_to_idxs = {pid: np.array(idxs) for pid, idxs in pid_to_idxs.items()}

def candidate_pids(top_arr, q_id):
    """top_arr: (96, NPROBE) centroid indices. 해당 query의 32개 토큰 top-2 centroid에 걸린 고유 pid 집합."""
    c_ids = np.unique(top_arr[q_id * 32:(q_id + 1) * 32].flatten())
    pids = set()
    for c_id in c_ids:
        pids.update(centroid_to_pids.get(int(c_id), []).tolist())
    return list(pids)

def digital_maxsim(Q_q, D_pid):
    return float((Q_q @ D_pid.T).max(axis=1).sum())

def mincurrent_optA(Q_q, D_pid):
    d    = Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :]
    I_uA = cell_optA(d, Vth_orig, Vth_orig).sum(axis=2) * 1e-3
    return float(I_uA.min(axis=1).sum())

def mincurrent_optC(Q_q, D_pid, vth_m0_tok, vth_m3_tok):
    d    = Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :]
    I_uA = cell_optC(d, vth_m0_tok[np.newaxis, :, :], vth_m3_tok[np.newaxis, :, :]).sum(axis=2) * 1e-3
    return float(I_uA.min(axis=1).sum())

def rank_of_true(pids, scores, true_pid, ascending):
    """scores: list aligned with pids. ascending=True면 작을수록 1등(전류), False면 클수록 1등(maxsim)."""
    order = np.argsort(scores) if ascending else np.argsort(scores)[::-1]
    sorted_pids = [pids[i] for i in order]
    if true_pid not in sorted_pids:
        return None, len(pids)
    return sorted_pids.index(true_pid) + 1, len(pids)

# ==========================================================================
# Digital + Option A: 한 번만 계산 (Vth 랜덤과 무관)
# ==========================================================================
log("\nDigital + Option A 계산 중 (seed 무관, 1회)...")
digital_top = rank_order_arr[:, :NPROBE]

d_step2   = Q[:, np.newaxis, :] - C[np.newaxis, :, :]
I_optA_uA = cell_optA(d_step2, Vth_orig, Vth_orig).sum(axis=2) * 1e-3
optA_top  = np.argsort(I_optA_uA, axis=1)[:, :NPROBE]

fixed_results = {}
for q_id in [0, 1, 2]:
    Q_q      = Q[q_id * 32:(q_id + 1) * 32]
    true_pid = QUERY_RELEVANT[q_id]

    dig_pids   = candidate_pids(digital_top, q_id)
    dig_scores = [digital_maxsim(Q_q, D[pid_to_idxs[p]]) for p in dig_pids]
    rank_dig, n_dig = rank_of_true(dig_pids, dig_scores, true_pid, ascending=False)

    A_pids   = candidate_pids(optA_top, q_id)
    A_scores = [mincurrent_optA(Q_q, D[pid_to_idxs[p]]) for p in A_pids]
    rank_A, n_A = rank_of_true(A_pids, A_scores, true_pid, ascending=True)

    fixed_results[q_id] = {'rank_dig': rank_dig, 'n_dig': n_dig, 'rank_A': rank_A, 'n_A': n_A}
    log(f"  q{q_id}: Digital rank={rank_dig}/{n_dig}  Option A rank={rank_A}/{n_A}")

# ==========================================================================
# Option C: 40 seed 반복, seed마다 CSV에 즉시 append
# ==========================================================================
log(f"\nOption C를 {N_SEEDS}개 seed로 반복 계산 중...")

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['seed', 'q_id', 'rank_C', 'n_C', 'found', 'rank_dig', 'n_dig', 'rank_A', 'n_A'])
    f.flush()

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed=seed)
        vth_m0_cent = sample_vth((NC, DIM), rng)
        vth_m3_cent = sample_vth((NC, DIM), rng)
        vth_m0_doc  = sample_vth((NT, DIM), rng)
        vth_m3_doc  = sample_vth((NT, DIM), rng)

        vm0_b = vth_m0_cent[np.newaxis, :, :]
        vm3_b = vth_m3_cent[np.newaxis, :, :]
        I_optC_uA = cell_optC(d_step2, vm0_b, vm3_b).sum(axis=2) * 1e-3
        optC_top  = np.argsort(I_optC_uA, axis=1)[:, :NPROBE]

        line = f"  seed={seed:2d}"
        for q_id in [0, 1, 2]:
            Q_q      = Q[q_id * 32:(q_id + 1) * 32]
            true_pid = QUERY_RELEVANT[q_id]

            C_pids = candidate_pids(optC_top, q_id)
            C_scores = []
            for p in C_pids:
                idxs = pid_to_idxs[p]
                C_scores.append(mincurrent_optC(Q_q, D[idxs], vth_m0_doc[idxs], vth_m3_doc[idxs]))
            rank_C, n_C = rank_of_true(C_pids, C_scores, true_pid, ascending=True)
            found = rank_C is not None

            writer.writerow([seed, q_id, rank_C if rank_C is not None else '', n_C, found,
                              fixed_results[q_id]['rank_dig'], fixed_results[q_id]['n_dig'],
                              fixed_results[q_id]['rank_A'], fixed_results[q_id]['n_A']])
            line += f"  q{q_id}=C:{rank_C}/{n_C}"
        f.flush()
        log(line)

log(f"\n저장: {OUT_CSV}")

# ==========================================================================
# 요약
# ==========================================================================
df_sweep = pd.read_csv(OUT_CSV)
log("\n=== 요약 ===")
for q_id in [0, 1, 2]:
    sub = df_sweep[df_sweep['q_id'] == q_id]
    ranks = sub['rank_C'].dropna()
    n_dropped = sub['found'].eq(False).sum()
    log(f"\nq{q_id}: Digital={fixed_results[q_id]['rank_dig']}  Option A={fixed_results[q_id]['rank_A']}")
    if len(ranks):
        log(f"  Option C rank 분포 (40 seed): min={ranks.min()}  max={ranks.max()}  mean={ranks.mean():.2f}  median={ranks.median()}")
        worst_seed = sub.loc[sub['rank_C'].idxmax()]
        log(f"  worst seed={int(worst_seed['seed'])}: rank_C={int(worst_seed['rank_C'])}/{int(worst_seed['n_C'])}")
    log(f"  candidate 탈락 seed 수: {n_dropped}/{N_SEEDS}")

log("\n완료!")
