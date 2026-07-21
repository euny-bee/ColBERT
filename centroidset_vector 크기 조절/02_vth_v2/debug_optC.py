"""
Option C rank 1 원인 분석
Test A: 구 모델(|d|, 단일 Vth) + clip99.9 데이터  → clip99.9 효과 확인
Test B: 신 모델(signed d, M0/M3 Vth) + 원본 데이터 → 셀 모델 효과 확인
+ Step3 n_tokens 비교
"""

import pandas as pd
import numpy as np
from scipy.stats import truncnorm

BASE_NEW = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\02_vth_v2'
DATA_BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline'
BASE_OLD = r'C:\Users\nmdl-khb\ColBERT\centroidset'

QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}
NPROBE = 2
SEED   = 42

# ── 파라미터 ──────────────────────────────────────────────────────────────────
L_f = 6.112020; K_f = 2.731949; B_f = -10.713911
Vth_orig = 0.151045
VDS_MEAS = 1.0; VDS_TARG = 1.7
VSAT = 2.88; SLOPE = 4.0332e-5; VDD = 1.7; CLIP = 1e-14

VTH_STD = 0.15; VTH_LO = 0.0; VTH_HI = 0.5
_a = (VTH_LO - 0.0) / VTH_STD
_b = (VTH_HI - 0.0) / VTH_STD

def make_rng(): return np.random.default_rng(seed=SEED)

def sample_vth(shape, rng):
    return Vth_orig + truncnorm.rvs(_a, _b, loc=0.0, scale=VTH_STD,
                                    size=shape, random_state=rng)

# ── 구 모델 (|d|, 단일 Vth) ───────────────────────────────────────────────────
def calc_ids_old(diff, vth):
    vgs = diff
    ids = 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - vth))))
    VoD = vgs - vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > VDS_MEAS, Vo*VDS_MEAS - VDS_MEAS**2/2, Vo**2/2)
    It = np.where(Vo > VDS_TARG, Vo*VDS_TARG - VDS_TARG**2/2, Vo**2/2)
    factor[on] = np.where(Im > 0, It/Im, 1.0)
    return ids * factor

# ── 신 모델 (signed d, M0/M3) ─────────────────────────────────────────────────
def I_single(vgs, vth):
    v = np.minimum(vgs, VSAT)
    return (10 ** (B_f + L_f / (1 + np.exp(-K_f * (v - vth))))
            + np.maximum(0.0, vgs - VSAT) * SLOPE)

def calc_ids_new(d, vth_m0, vth_m3):
    vgd_m0 = d
    I_m0   = np.where(d >= 0,
                      np.maximum(I_single(vgd_m0, vth_m0)
                                 - I_single(vgd_m0 - VDD, vth_m0), 0.0), 0.0)
    vgd_m3 = -d
    I_m3   = np.where(d < 0,
                      np.maximum(I_single(vgd_m3, vth_m3)
                                 - I_single(vgd_m3 - VDD, vth_m3), 0.0), 0.0)
    return np.clip(I_m0 + I_m3, CLIP, None) * 1e9   # nA

# ── Step 2 + 3 공통 로직 ──────────────────────────────────────────────────────
def ivf_lookup(top_arr, ivf, Q_df):
    results = {}
    for q_id in [0, 1, 2]:
        true_pid = QUERY_RELEVANT[q_id]
        pid_info = {}
        for i in range(q_id*32, (q_id+1)*32):
            for c_id in top_arr[i]:
                c_id = int(c_id)
                if c_id not in ivf.index: continue
                for _, row in ivf.loc[[c_id]].iterrows():
                    pid = int(row['pid'])
                    if pid not in pid_info:
                        pid_info[pid] = {'token_ids': set()}
                    pid_info[pid]['token_ids'].add(Q_df.index[i])
        results[q_id] = {pid: len(info['token_ids']) for pid, info in pid_info.items()}
    return results

def step6_score(Q_q, D_pid, model, rng):
    """MinCurrent 계산. model: 'old' or 'new'"""
    M = D_pid.shape[0]
    if model == 'old':
        diff = np.abs(Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :])   # (32,M,128)
        vth  = sample_vth((32, M), rng)[:, :, np.newaxis]
        I    = calc_ids_old(diff, vth).sum(axis=2) * 1e6   # uA
    else:
        d       = Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :]
        vth_m0  = sample_vth((32, M), rng)[:, :, np.newaxis]
        vth_m3  = sample_vth((32, M), rng)[:, :, np.newaxis]
        I       = calc_ids_new(d, vth_m0, vth_m3).sum(axis=2) * 1e-3   # uA
    return float(I.min(axis=1).sum())

# =============================================================================
print("=" * 60)
print("Step3 n_tokens 비교 (true document)")
print("=" * 60)
for q_id in [0, 1, 2]:
    true_pid = QUERY_RELEVANT[q_id]

    # 구 centroidset optC
    old_df = pd.read_excel(f'{BASE_OLD}/step3_optionC_candidate_pids.xlsx',
                           sheet_name=f'q{q_id}')
    old_row = old_df[old_df['pid'] == true_pid]
    old_ntok = int(old_row['n_tokens'].values[0]) if len(old_row) else 'N/A'

    # 신 vth_v2 optC
    new_df = pd.read_excel(f'{BASE_NEW}/[vth_v2]step3_optC_candidate_pids.xlsx',
                           sheet_name=f'q{q_id}')
    new_row = new_df[new_df['pid'] == true_pid]
    new_ntok = int(new_row['n_tokens'].values[0]) if len(new_row) else 'N/A'

    # 신 vth_v2 digital
    dig_df = pd.read_excel(f'{BASE_NEW}/[vth_v2]step3_digital_candidate_pids.xlsx',
                           sheet_name=f'q{q_id}')
    dig_row = dig_df[dig_df['pid'] == true_pid]
    dig_ntok = int(dig_row['n_tokens'].values[0]) if len(dig_row) else 'N/A'

    print(f"  q{q_id} true_pid={true_pid}:")
    print(f"    Digital                n_tokens = {dig_ntok}")
    print(f"    구 optC (centroidset)  n_tokens = {old_ntok}")
    print(f"    신 optC (vth_v2)       n_tokens = {new_ntok}")

# =============================================================================
print("\n" + "=" * 60)
print("Test A: 구 모델 + clip99.9 데이터")
print("=" * 60)
Q_new = pd.read_excel(f'{DATA_BASE}/[clip99.9]query_embs_96x128.xlsx',  index_col=0)
C_new = pd.read_excel(f'{DATA_BASE}/[clip99.9]centroids_100x128.xlsx',  index_col=0)
doc_new = pd.read_excel(f'{DATA_BASE}/[clip99.9]doc_embs_12919x128.xlsx', index_col=0)
ivf_new = pd.read_excel(f'{DATA_BASE}/[clip99.9]ivf_centroid2pid.xlsx',
                        sheet_name='long_format', index_col=0)

Q_n = Q_new.values.astype(float)
C_n = C_new.values.astype(float)
dim_cols = [c for c in doc_new.columns if c.startswith('dim_')]
D_n = doc_new[dim_cols].values.astype(float)
doc_tok_n = {tok: i for i, tok in enumerate(doc_new.index)}

rng_a = make_rng()
diff_s2 = np.abs(Q_n[:, np.newaxis, :] - C_n[np.newaxis, :, :])
vth_s2  = sample_vth((96, 100), rng_a)[:, :, np.newaxis]
I_s2    = calc_ids_old(diff_s2, vth_s2).sum(axis=2) * 1e6
top_a   = np.argsort(I_s2, axis=1)[:, :NPROBE]

cands_a = ivf_lookup(top_a, ivf_new, Q_new)

for q_id in [0, 1, 2]:
    Q_q      = Q_n[q_id*32:(q_id+1)*32]
    true_pid = QUERY_RELEVANT[q_id]
    pids     = list(cands_a[q_id].keys())
    rows = []
    for pid in pids:
        toks = [k for k in doc_new.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_tok_n[k] for k in toks]
        rows.append({'pid': pid,
                     'score': step6_score(Q_q, D_n[idxs], 'old', make_rng())})
    df = pd.DataFrame(rows)
    df['rank'] = df['score'].rank(ascending=True).astype(int)
    true_rank = int(df[df['pid']==true_pid]['rank'].values[0]) if true_pid in df['pid'].values else 'N/A'
    true_ntok = cands_a[q_id].get(true_pid, 'N/A')
    print(f"  q{q_id}: true_pid rank = {true_rank}/{len(df)}, n_tokens={true_ntok}")

# =============================================================================
print("\n" + "=" * 60)
print("Test B: 신 모델 + 원본 데이터 (centroidset)")
print("=" * 60)
Q_old = pd.read_excel(f'{BASE_OLD}/query_embs_96x128.xlsx',     index_col=0)
C_old = pd.read_excel(f'{BASE_OLD}/centroids_100x128.xlsx',     index_col=0)
doc_old = pd.read_excel(f'{BASE_OLD}/doc_embs_12919x128.xlsx',  index_col=0)
ivf_old = pd.read_excel(f'{BASE_OLD}/ivf_centroid2token2pid.xlsx', index_col=0)

Q_o = Q_old.values.astype(float)
C_o = C_old.values.astype(float)
dim_cols_o = [c for c in doc_old.columns if c.startswith('dim_')]
D_o = doc_old[dim_cols_o].values.astype(float)
doc_tok_o = {tok: i for i, tok in enumerate(doc_old.index)}

rng_b = make_rng()
d_s2_b    = Q_o[:, np.newaxis, :] - C_o[np.newaxis, :, :]
vth_m0_s2 = sample_vth((96, 100), rng_b)[:, :, np.newaxis]
vth_m3_s2 = sample_vth((96, 100), rng_b)[:, :, np.newaxis]
I_s2_b    = calc_ids_new(d_s2_b, vth_m0_s2, vth_m3_s2).sum(axis=2) * 1e-3
top_b     = np.argsort(I_s2_b, axis=1)[:, :NPROBE]

# ivf_old 구조 확인 후 lookup
if 'pid' in ivf_old.columns:
    ivf_b = ivf_old
else:
    ivf_b = ivf_old.reset_index()

cands_b = {}
for q_id in [0, 1, 2]:
    true_pid = QUERY_RELEVANT[q_id]
    pid_info = {}
    for i in range(q_id*32, (q_id+1)*32):
        for c_id in top_b[i]:
            c_id = int(c_id)
            rows_ivf = ivf_b[ivf_b.index == c_id] if c_id in ivf_b.index else pd.DataFrame()
            for _, row in rows_ivf.iterrows():
                pid = int(row['pid'])
                if pid not in pid_info:
                    pid_info[pid] = set()
                pid_info[pid].add(Q_old.index[i])
    cands_b[q_id] = {pid: len(toks) for pid, toks in pid_info.items()}

for q_id in [0, 1, 2]:
    Q_q      = Q_o[q_id*32:(q_id+1)*32]
    true_pid = QUERY_RELEVANT[q_id]
    pids     = list(cands_b[q_id].keys())
    rows = []
    for pid in pids:
        toks = [k for k in doc_old.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_tok_o[k] for k in toks]
        rows.append({'pid': pid,
                     'score': step6_score(Q_q, D_o[idxs], 'new', make_rng())})
    df = pd.DataFrame(rows)
    df['rank'] = df['score'].rank(ascending=True).astype(int)
    true_rank = int(df[df['pid']==true_pid]['rank'].values[0]) if true_pid in df['pid'].values else 'N/A'
    true_ntok = cands_b[q_id].get(true_pid, 'N/A')
    print(f"  q{q_id}: true_pid rank = {true_rank}/{len(df)}, n_tokens={true_ntok}")

print("\n완료!")
