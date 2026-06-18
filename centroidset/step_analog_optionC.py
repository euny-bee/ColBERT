"""
Option C Analog ColBERTv2 Pipeline (Step 2, 3, 6)
회로 모델: Option C  vgs = |Q_k - D_k|  (Vth 보상 없음)
Vth: Truncated Gaussian (mean=0, std=0.15, range=[-0.5,+0.5]) per (Qi,Dj) pair
seed=42  (rng 객체 하나로 전체 순차 사용 -> 재현 가능)
"""

import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import os

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset'
NPROBE         = 2
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}

# -- 회로 파라미터 (die4) -------------------------------------------------------
L_f      =  6.112020
K_f      =  2.731949
B_f      = -10.713911
Vth_orig =  0.151045
VDS_MEAS =  1.0
VDS_TARG =  1.7

# -- Vth 분포 설정 -------------------------------------------------------------
VTH_MEAN, VTH_STD = 0.0, 0.15
VTH_LO,   VTH_HI  = 0.0, 0.5   # PBS: positive shift only
_a = (VTH_LO - VTH_MEAN) / VTH_STD
_b = (VTH_HI - VTH_MEAN) / VTH_STD
rng = np.random.default_rng(seed=42)

def sample_vth(shape):
    """Truncated Gaussian에서 Vth shift 샘플링 -> Vth_orig + shift"""
    shifts = truncnorm.rvs(_a, _b, loc=VTH_MEAN, scale=VTH_STD,
                           size=shape, random_state=rng)
    return Vth_orig + shifts


# -- Option C IDS 계산 (벡터화) ------------------------------------------------
def calc_ids_optC(diff, vth):
    """
    Option C: vgs = diff = |Q - D|  (Vth 보상 없음)
    diff, vth: 임의 shape (브로드캐스트 가능해야 함)
    반환: IDS [A], 같은 shape
    """
    vgs    = diff                           # Option C 핵심
    ids    = 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - vth))))
    VoD    = vgs - vth
    factor = np.ones_like(VoD, dtype=float)
    on     = VoD > 0
    Vo     = VoD[on]
    Im     = np.where(Vo > VDS_MEAS, Vo * VDS_MEAS - VDS_MEAS**2 / 2, Vo**2 / 2)
    It     = np.where(Vo > VDS_TARG, Vo * VDS_TARG - VDS_TARG**2 / 2, Vo**2 / 2)
    factor[on] = np.where(Im > 0, It / Im, 1.0)
    return ids * factor


# -- Score 계산 함수 -----------------------------------------------------------
def digital_maxsim(Q_q, D_pid):
    """Digital MaxSim: Σᵢ maxⱼ (Qᵢ · Dⱼᵀ)"""
    return (Q_q @ D_pid.T).max(axis=1).sum()


def optC_mincurrent(Q_q, D_pid, Vth_mat):
    """
    Option C MinCurrent: Σᵢ minⱼ I_total(Qᵢ, Dⱼ)
    Q_q:     (32, 128)
    D_pid:   (M,  128)
    Vth_mat: (32, M)    <- 이미 샘플링된 Vth
    반환: scalar score (낮을수록 유사)
    """
    # diff: (32, M, 128)
    diff = np.abs(Q_q[:, None, :] - D_pid[None, :, :])
    # Vth broadcast: (32, M) -> (32, M, 1)
    vth  = Vth_mat[:, :, None]
    # IDS: (32, M, 128) -> sum dim -> (32, M) [uA]
    currents = calc_ids_optC(diff, vth).sum(axis=2) * 1e6
    # 각 Qᵢ에 대해 min over Dⱼ -> (32,) -> sum
    return float(currents.min(axis=1).sum())


# -- 데이터 로드 ---------------------------------------------------------------
print("데이터 로드 중...")
Q_raw = pd.read_excel(f'{BASE}/query_embs_96x128.xlsx',      header=0, index_col=0)
C_df  = pd.read_excel(f'{BASE}/centroids_100x128.xlsx',       header=0, index_col=0)
ivf   = pd.read_excel(f'{BASE}/ivf_centroid2token2pid.xlsx',  header=0, index_col=0)
r2b   = pd.read_csv(  f'{BASE}/residuals_2bit.csv',           header=0, index_col=0)
doc   = pd.read_excel(f'{BASE}/doc_embs_12919x128.xlsx',      header=0, index_col=0)

Q = Q_raw.values.astype(float)   # (96,  128)
C = C_df.values.astype(float)    # (100, 128)

dim_cols        = [c for c in doc.columns if c.startswith('dim_')]
D_f32           = doc[dim_cols].values.astype(float)        # (12919, 128)
token2centroid  = dict(zip(ivf['token_id'], ivf.index))
centroid_ids    = [token2centroid[t] for t in doc.index]
Ct              = C[centroid_ids]
D_2bt           = Ct + r2b.values.astype(float)             # (12919, 128)
doc_token_index = {tok: i for i, tok in enumerate(doc.index)}

# =============================================================================
# STEP 2: Option C centroid 선택
# =============================================================================
print("\n[Step 2] Option C centroid 전류 계산 중...")

# Vth 샘플링: (96, 100) pairs
Vth_step2 = sample_vth((96, 100))   # (96, 100)

# diff: (96, 100, 128)
diff_step2 = np.abs(Q[:, None, :] - C[None, :, :])
vth_step2  = Vth_step2[:, :, None]                          # (96, 100, 1)
I_matrix   = calc_ids_optC(diff_step2, vth_step2).sum(axis=2) * 1e6  # (96, 100)

optC_top = np.argsort(I_matrix, axis=1)[:, :NPROBE]         # (96, nprobe)

rows_step2 = []
for i in range(96):
    q_id       = i // 32
    token_name = Q_raw.index[i]
    for rank, c_id in enumerate(optC_top[i]):
        rows_step2.append({
            'token_id':    token_name,
            'query_id':    q_id,
            'rank':        rank + 1,
            'centroid_id': int(c_id),
            'current_uA':  round(float(I_matrix[i, c_id]), 4),
        })

pd.DataFrame(rows_step2).to_excel(f'{BASE}/step2_optionC_centroid.xlsx', index=False)
print("  저장: step2_optionC_centroid.xlsx")

# =============================================================================
# STEP 3: Inverted list 조회
# =============================================================================
print("\n[Step 3] Option C inverted list 조회 중...")

results_step3 = {}
for q_id in [0, 1, 2]:
    token_range = range(q_id * 32, (q_id + 1) * 32)
    true_pid    = QUERY_RELEVANT[q_id]
    pid_info    = {}

    for i in token_range:
        token_name = Q_raw.index[i]
        for c_id in optC_top[i]:
            for _, row in ivf.loc[[c_id]].iterrows():
                pid = row['pid']
                if pid not in pid_info:
                    pid_info[pid] = {'centroid_ids': set(), 'token_ids': set()}
                pid_info[pid]['centroid_ids'].add(int(c_id))
                pid_info[pid]['token_ids'].add(token_name)

    rows_list = [{'pid': pid,
                  'is_relevant': pid == true_pid,
                  'n_centroids': len(info['centroid_ids']),
                  'n_tokens':    len(info['token_ids'])}
                 for pid, info in pid_info.items()]

    df_q = (pd.DataFrame(rows_list)
              .sort_values('n_tokens', ascending=False)
              .reset_index(drop=True))
    results_step3[f'q{q_id}'] = df_q
    print(f"  [q{q_id}] 후보 {len(df_q)}개, 정답 pid {true_pid} 포함: {true_pid in df_q['pid'].values}")

with pd.ExcelWriter(f'{BASE}/step3_optionC_candidate_pids.xlsx', engine='openpyxl') as w:
    for q_name, df_r in results_step3.items():
        df_r.to_excel(w, sheet_name=q_name, index=False)
print("  저장: step3_optionC_candidate_pids.xlsx")

# =============================================================================
# STEP 6: MinCurrent 랭킹 (f32 / 2bit)
# =============================================================================
print("\n[Step 6] MinCurrent 계산 중...")

all_results = {}

for q_id in [0, 1, 2]:
    Q_q      = Q[q_id * 32:(q_id + 1) * 32]   # (32, 128)
    true_pid = QUERY_RELEVANT[q_id]

    # --- digital (기존 digital 후보 사용) ---
    digital_pids = pd.read_excel(f'{BASE}/step3_candidate_pids.xlsx',
                                 sheet_name=f'q{q_id}')['pid'].tolist()
    dig_rows = []
    for pid in digital_pids:
        token_keys = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not token_keys:
            continue
        idxs = [doc_token_index[k] for k in token_keys]
        dig_rows.append({
            'pid':           pid,
            'is_relevant':   pid == true_pid,
            'score_dig_f32': round(digital_maxsim(Q_q, D_f32[idxs]), 4),
            'score_dig_2bt': round(digital_maxsim(Q_q, D_2bt[idxs]), 4),
        })
    df_dig = pd.DataFrame(dig_rows)
    df_dig['rank_dig_f32'] = df_dig['score_dig_f32'].rank(ascending=False).astype(int)
    df_dig['rank_dig_2bt'] = df_dig['score_dig_2bt'].rank(ascending=False).astype(int)

    # --- Option C (optC 후보 사용, Vth는 문서당 1회 샘플링) ---
    optC_pids = results_step3[f'q{q_id}']['pid'].tolist()
    optC_rows = []
    for pid in optC_pids:
        token_keys = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not token_keys:
            continue
        idxs  = [doc_token_index[k] for k in token_keys]
        M     = len(idxs)

        # (32, M) Vth 행렬: f32 / 2bit 동일한 Vth 사용 (같은 하드웨어)
        Vth_mat = sample_vth((32, M))

        optC_rows.append({
            'pid':            pid,
            'is_relevant':    pid == true_pid,
            'curr_optC_f32':  round(optC_mincurrent(Q_q, D_f32[idxs], Vth_mat), 4),
            'curr_optC_2bt':  round(optC_mincurrent(Q_q, D_2bt[idxs], Vth_mat), 4),
        })

    df_optC = pd.DataFrame(optC_rows)
    df_optC['rank_optC_f32'] = df_optC['curr_optC_f32'].rank(ascending=True).astype(int)
    df_optC['rank_optC_2bt'] = df_optC['curr_optC_2bt'].rank(ascending=True).astype(int)

    all_results[f'q{q_id}'] = {'digital': df_dig, 'optC': df_optC}

    # 중간 출력
    print(f"\n  [q{q_id}] 정답 pid: {true_pid}")
    for label, df_, rcol, scol in [
        ('digital  f32', df_dig,  'rank_dig_f32',  'score_dig_f32'),
        ('digital  2bt', df_dig,  'rank_dig_2bt',  'score_dig_2bt'),
        ('optionC f32',  df_optC, 'rank_optC_f32', 'curr_optC_f32'),
        ('optionC 2bt',  df_optC, 'rank_optC_2bt', 'curr_optC_2bt'),
    ]:
        rel = df_[df_['is_relevant']]
        if len(rel):
            r   = rel.iloc[0]
            n   = len(df_)
            print(f"    {label}: score/curr={r[scol]:9.4f}  rank={int(r[rcol])}/{n}")

# =============================================================================
# 저장
# =============================================================================
def in_topk(rank, k):
    return '✓' if pd.notna(rank) and int(rank) <= k else '✗'

summary_rows = []
for q_id in [0, 1, 2]:
    df_dig  = all_results[f'q{q_id}']['digital']
    df_optC = all_results[f'q{q_id}']['optC']
    rel_dig  = df_dig[df_dig['is_relevant']]
    rel_optC = df_optC[df_optC['is_relevant']]

    row = {
        'query':        f'q{q_id}',
        'true_pid':     QUERY_RELEVANT[q_id],
        'n_cands_dig':  len(df_dig),
        'n_cands_optC': len(df_optC),
    }
    if len(rel_dig):
        r = rel_dig.iloc[0]
        for col in ['rank_dig_f32', 'rank_dig_2bt']:
            row[col] = int(r[col])
        for k in [10, 20, 50]:
            row[f'top{k}_dig_f32'] = in_topk(r['rank_dig_f32'], k)
            row[f'top{k}_dig_2bt'] = in_topk(r['rank_dig_2bt'], k)
    if len(rel_optC):
        r = rel_optC.iloc[0]
        for col in ['rank_optC_f32', 'rank_optC_2bt']:
            row[col] = int(r[col])
        for k in [10, 20, 50]:
            row[f'top{k}_optC_f32'] = in_topk(r['rank_optC_f32'], k)
            row[f'top{k}_optC_2bt'] = in_topk(r['rank_optC_2bt'], k)
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)

# Success@k 집계 행 추가
for k in [10, 20, 50]:
    srow = {'query': f'Success@{k}', 'true_pid': '', 'n_cands_dig': '', 'n_cands_optC': ''}
    for tag in ['dig_f32', 'dig_2bt', 'optC_f32', 'optC_2bt']:
        col = f'top{k}_{tag}'
        if col in df_summary.columns:
            n = (df_summary[col] == '✓').sum()
            srow[col] = f'{n}/3 ({n/3*100:.0f}%)'
    summary_rows.append(srow)

out_path = f'{BASE}/step6_optionC_results.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='summary', index=False)
    for q_id in [0, 1, 2]:
        (all_results[f'q{q_id}']['digital']
         .sort_values('rank_dig_f32')
         .to_excel(writer, sheet_name=f'q{q_id}_digital', index=False))
        (all_results[f'q{q_id}']['optC']
         .sort_values('rank_optC_f32')
         .to_excel(writer, sheet_name=f'q{q_id}_optionC', index=False))

print(f"\nSaved: {out_path}")
