"""
[clip99.9] 임베딩 기반 ColBERT 아날로그 파이프라인
Step 2: Centroid 선택 (Digital / Option A / Option C)
Step 3: IVF lookup → 후보 pid 수집
Step 6: MaxSim / MinCurrent 점수 계산 → Ranking
"""

import pandas as pd
import numpy as np
from scipy.stats import truncnorm

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline'
NPROBE         = 2
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}

# ==========================================================================
# 회로 파라미터 (die4)
# ==========================================================================
L_f      =  6.112020
K_f      =  2.731949
B_f      = -10.713911
Vth_orig =  0.151045
VDS_MEAS =  1.0
VDS_TARG =  1.7

def logistic_ids(vgs, vth):
    return 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - vth))))

def vds_correction(vgs, vth):
    VoD    = vgs - vth
    factor = np.ones_like(VoD, dtype=float)
    on     = VoD > 0
    Vo     = VoD[on]
    Im     = np.where(Vo > VDS_MEAS, Vo*VDS_MEAS - VDS_MEAS**2/2, Vo**2/2)
    It     = np.where(Vo > VDS_TARG, Vo*VDS_TARG - VDS_TARG**2/2, Vo**2/2)
    factor[on] = np.where(Im > 0, It/Im, 1.0)
    return factor

def calc_ids_optA(diff):
    """Option A: VGS = |V2-V1| + Vth (고정)"""
    vgs = diff + Vth_orig
    return logistic_ids(vgs, Vth_orig) * vds_correction(vgs, Vth_orig)

def calc_ids_optC(diff, vth):
    """Option C: VGS = |V2-V1|, Vth ~ TruncGaussian"""
    vgs = diff
    return logistic_ids(vgs, vth) * vds_correction(vgs, vth)

# Option C Vth 분포
VTH_MEAN, VTH_STD = 0.0, 0.15
VTH_LO,   VTH_HI  = 0.0, 0.5
_a = (VTH_LO - VTH_MEAN) / VTH_STD
_b = (VTH_HI - VTH_MEAN) / VTH_STD
rng = np.random.default_rng(seed=42)

def sample_vth(shape):
    return Vth_orig + truncnorm.rvs(_a, _b, loc=VTH_MEAN, scale=VTH_STD,
                                    size=shape, random_state=rng)

# ==========================================================================
# 데이터 로드
# ==========================================================================
print("데이터 로드 중...")
Q_df  = pd.read_excel(f'{BASE}/[clip99.9]query_embs_96x128.xlsx',  index_col=0)
C_df  = pd.read_excel(f'{BASE}/[clip99.9]centroids_100x128.xlsx',  index_col=0)
doc   = pd.read_excel(f'{BASE}/[clip99.9]doc_embs_12919x128.xlsx', index_col=0)
ivf   = pd.read_excel(f'{BASE}/[clip99.9]ivf_centroid2pid.xlsx',
                      sheet_name='long_format', index_col=0)
qcr   = pd.read_excel(f'{BASE}/[clip99.9]query_centroid_ranking.xlsx',
                      sheet_name='rank_order', index_col=0)

dim_cols = [c for c in doc.columns if c.startswith('dim_')]
Q = Q_df.values.astype(float)                  # (96, 128)
C = C_df.values.astype(float)                  # (100, 128)
D = doc[dim_cols].values.astype(float)         # (12919, 128)
is_rel_doc = doc['is_relevant'].values.astype(bool)

doc_token_index = {tok: i for i, tok in enumerate(doc.index)}
rank_cols       = [f'rank_{i+1}' for i in range(100)]
rank_order_arr  = qcr[rank_cols].values        # (96, 100)

# ==========================================================================
# STEP 2: Centroid 선택
# ==========================================================================
print("\n[Step 2] Centroid 선택 중...")

# --- Digital: rank_order에서 top-2
digital_top = rank_order_arr[:, :NPROBE]   # (96, 2)
print("  Digital: rank_order top-2 완료")

# --- Option A: |Q-C| → IDS → MinCurrent top-2
print("  Option A: 전류 계산 중... (96×100)")
I_optA = np.zeros((96, 100))
for i in range(96):
    diff = np.abs(Q[i][np.newaxis, :] - C)    # (100, 128)
    I_optA[i] = calc_ids_optA(diff).sum(axis=1) * 1e6
optA_top = np.argsort(I_optA, axis=1)[:, :NPROBE]
print("  Option A 완료")

# --- Option C: Vth mismatch
print("  Option C: 전류 계산 중... (96×100×128)")
I_optC = np.zeros((96, 100))
for i in range(96):
    diff = np.abs(Q[i][np.newaxis, :] - C)     # (100, 128)
    vth  = sample_vth((100, 128))
    I_optC[i] = calc_ids_optC(diff, vth).sum(axis=1) * 1e6
optC_top = np.argsort(I_optC, axis=1)[:, :NPROBE]
print("  Option C 완료")

# Step 2 저장
def save_step2(top_arr, I_arr, name, score_col):
    rows = []
    for i in range(96):
        q_id       = i // 32
        token_name = Q_df.index[i]
        for rank, c_id in enumerate(top_arr[i]):
            rows.append({
                'token_id':    token_name,
                'query_id':    q_id,
                'rank':        rank + 1,
                'centroid_id': int(c_id),
                score_col:     round(float(I_arr[i, c_id]), 4) if I_arr is not None else None,
            })
    df = pd.DataFrame(rows)
    df.to_excel(f'{BASE}/[clip99.9]step2_{name}_centroid.xlsx', index=False)
    print(f"  저장: [clip99.9]step2_{name}_centroid.xlsx")
    return df

save_step2(digital_top, None,   'digital', 'score_rank')
save_step2(optA_top,    I_optA, 'optA',    'current_uA')
save_step2(optC_top,    I_optC, 'optC',    'current_uA')

# ==========================================================================
# STEP 3: IVF lookup → 후보 pid 수집
# ==========================================================================
print("\n[Step 3] IVF lookup 중...")

def ivf_lookup(top_arr, method_name):
    results = {}
    for q_id in [0, 1, 2]:
        true_pid    = QUERY_RELEVANT[q_id]
        token_range = range(q_id * 32, (q_id + 1) * 32)
        pid_info    = {}
        for i in token_range:
            token_name = Q_df.index[i]
            for c_id in top_arr[i]:
                c_id = int(c_id)
                if c_id not in ivf.index:
                    continue
                rows = ivf.loc[[c_id]]
                for _, row in rows.iterrows():
                    pid = int(row['pid'])
                    if pid not in pid_info:
                        pid_info[pid] = {'centroid_ids': set(), 'token_ids': set()}
                    pid_info[pid]['centroid_ids'].add(c_id)
                    pid_info[pid]['token_ids'].add(token_name)

        rows_list = []
        for pid, info in pid_info.items():
            rows_list.append({
                'pid':         pid,
                'is_relevant': pid == true_pid,
                'n_centroids': len(info['centroid_ids']),
                'n_tokens':    len(info['token_ids']),
            })
        df_q = (pd.DataFrame(rows_list)
                  .sort_values('n_tokens', ascending=False)
                  .reset_index(drop=True))
        results[f'q{q_id}'] = df_q
        found = true_pid in df_q['pid'].values
        print(f"  [{method_name} q{q_id}] 후보 {len(df_q)}개  정답 포함: {found}")

    out = f'{BASE}/[clip99.9]step3_{method_name}_candidate_pids.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        for q_name, df_r in results.items():
            df_r.to_excel(writer, sheet_name=q_name, index=False)
    print(f"  저장: [clip99.9]step3_{method_name}_candidate_pids.xlsx")
    return results

cands_digital = ivf_lookup(digital_top, 'digital')
cands_optA    = ivf_lookup(optA_top,    'optA')
cands_optC    = ivf_lookup(optC_top,    'optC')

# ==========================================================================
# STEP 6: MaxSim / MinCurrent 점수 → Ranking
# ==========================================================================
print("\n[Step 6] 점수 계산 중...")

def digital_maxsim(Q_q, D_pid):
    return float((Q_q @ D_pid.T).max(axis=1).sum())

def analog_mincurrent_optA(Q_q, D_pid):
    diff = np.abs(Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :])  # (32, n, 128)
    I    = calc_ids_optA(diff).sum(axis=2) * 1e6                     # (32, n)
    return float(I.min(axis=1).sum())

def analog_mincurrent_optC(Q_q, D_pid, vth):
    diff = np.abs(Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :])  # (32, n, 128)
    I    = calc_ids_optC(diff, vth).sum(axis=2) * 1e6                # (32, n)
    return float(I.min(axis=1).sum())

all_results = {}

for q_id in [0, 1, 2]:
    Q_q      = Q[q_id*32:(q_id+1)*32]
    true_pid = QUERY_RELEVANT[q_id]

    dig_pids  = cands_digital[f'q{q_id}']['pid'].tolist()
    optA_pids = cands_optA[f'q{q_id}']['pid'].tolist()
    optC_pids = cands_optC[f'q{q_id}']['pid'].tolist()

    # --- Digital MaxSim
    dig_rows = []
    for pid in dig_pids:
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_token_index[k] for k in toks]
        dig_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                         'score_dig_f32': round(digital_maxsim(Q_q, D[idxs]), 4)})
    df_dig = pd.DataFrame(dig_rows)
    df_dig['rank_dig_f32'] = df_dig['score_dig_f32'].rank(ascending=False).astype(int)

    # --- Option A MinCurrent
    optA_rows = []
    for pid in optA_pids:
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_token_index[k] for k in toks]
        optA_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                          'current_optA_f32': round(analog_mincurrent_optA(Q_q, D[idxs]), 4)})
    df_optA = pd.DataFrame(optA_rows)
    df_optA['rank_optA_f32'] = df_optA['current_optA_f32'].rank(ascending=True).astype(int)

    # --- Option C MinCurrent (Vth mismatch per (query_token, doc_token, dim))
    optC_rows = []
    for pid in optC_pids:
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs  = [doc_token_index[k] for k in toks]
        D_pid = D[idxs]
        n_doc = len(D_pid)
        vth   = sample_vth((32, n_doc, 128))
        optC_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                          'current_optC_f32': round(analog_mincurrent_optC(Q_q, D_pid, vth), 4)})
    df_optC = pd.DataFrame(optC_rows)
    df_optC['rank_optC_f32'] = df_optC['current_optC_f32'].rank(ascending=True).astype(int)

    all_results[f'q{q_id}'] = {'digital': df_dig, 'optA': df_optA, 'optC': df_optC}

    print(f"\n  [q{q_id}] 정답 pid: {true_pid}")
    for label, df_r, rank_col in [
        ('Digital f32', df_dig,  'rank_dig_f32'),
        ('Option A',    df_optA, 'rank_optA_f32'),
        ('Option C',    df_optC, 'rank_optC_f32'),
    ]:
        rel = df_r[df_r['is_relevant']]
        if len(rel):
            r = rel.iloc[0]
            print(f"    {label:<12} rank: {int(r[rank_col])}/{len(df_r)}")
        else:
            print(f"    {label:<12} 정답 후보 없음")

# --- 저장
out_path = f'{BASE}/[clip99.9]step6_all_results.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    summary = []
    for q_id in [0, 1, 2]:
        r = all_results[f'q{q_id}']
        r['digital'].sort_values('rank_dig_f32').to_excel(writer, sheet_name=f'q{q_id}_digital', index=False)
        r['optA'].sort_values('rank_optA_f32').to_excel(writer,   sheet_name=f'q{q_id}_optA',    index=False)
        r['optC'].sort_values('rank_optC_f32').to_excel(writer,   sheet_name=f'q{q_id}_optC',    index=False)

        row = {'query': f'q{q_id}', 'true_pid': QUERY_RELEVANT[q_id],
               'n_digital': len(r['digital']), 'n_optA': len(r['optA']), 'n_optC': len(r['optC'])}
        for label, df_r, rank_col in [
            ('digital', r['digital'], 'rank_dig_f32'),
            ('optA',    r['optA'],    'rank_optA_f32'),
            ('optC',    r['optC'],    'rank_optC_f32'),
        ]:
            rel = df_r[df_r['is_relevant']]
            if len(rel):
                row[f'rank_{label}'] = int(rel.iloc[0][rank_col])
        summary.append(row)

    pd.DataFrame(summary).to_excel(writer, sheet_name='summary', index=False)

print(f"\n저장: [clip99.9]step6_all_results.xlsx")
print("\nPipeline 완료!")
