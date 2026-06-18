"""
Analog ColBERTv2 Pipeline (Step 2, 3, 6)
회로 모델: Option A  IDS = f(|V2-V1|)  (Vth 상쇄)

4가지 비교:
  digital  × float32 doc  (기존 완료 - 참조용 재계산)
  digital  × 2bit    doc
  analog   × float32 doc
  analog   × 2bit    doc
"""

import pandas as pd
import numpy as np
import os

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'
NPROBE = 2
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}

# -- 회로 파라미터 (die4 Option A) ------------------------------------------
L_f =  6.112020
K_f =  2.731949
B_f = -10.713911
Vth =  0.151045

def logistic_ids(vgs):
    return 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - Vth))))

def vds_correction(vgs):
    VoD = vgs - Vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > 1.0, Vo*1.0 - 1.0**2/2, Vo**2/2)
    It = np.where(Vo > 1.7, Vo*1.7 - 1.7**2/2, Vo**2/2)
    factor[on] = np.where(Im > 0, It/Im, 1.0)
    return factor

def calc_ids(diff):
    """Option A: IDS = f(|V2-V1|), Vth 상쇄. diff = |V2-V1|"""
    vgs = diff + Vth
    return logistic_ids(vgs) * vds_correction(vgs)   # [A]

def analog_total_current(A, B):
    """
    A: (..., 128)  B: (..., 128)
    반환: (...) — 각 쌍의 Match Line 전류 합 [uA]
    """
    diff = np.abs(A - B)
    return calc_ids(diff).sum(axis=-1) * 1e6

# -- 데이터 로드 -------------------------------------------------------------
print("데이터 로드 중...")
Q_raw = pd.read_excel(f'{BASE}/query_embs_96x128.xlsx',     header=0, index_col=0)
C_df  = pd.read_excel(f'{BASE}/centroids_100x128.xlsx',      header=0, index_col=0)
ivf   = pd.read_excel(f'{BASE}/ivf_centroid2token2pid.xlsx', header=0, index_col=0)
r2b   = pd.read_csv(  f'{BASE}/residuals_2bit.csv',          header=0, index_col=0)
doc   = pd.read_excel(f'{BASE}/doc_embs_12919x128.xlsx',     header=0, index_col=0)

Q = Q_raw.values.astype(float)                   # (96, 128)
C = C_df.values.astype(float)                    # (100, 128)

dim_cols = [c for c in doc.columns if c.startswith('dim_')]
D_f32 = doc[dim_cols].values.astype(float)       # (12919, 128) float32

token2centroid = dict(zip(ivf['token_id'], ivf.index))
centroid_ids   = [token2centroid[t] for t in doc.index]
Ct             = C[centroid_ids]
D_2bt          = Ct + r2b.values.astype(float)   # (12919, 128) 2bit

doc_token_index = {tok: i for i, tok in enumerate(doc.index)}

# ==========================================================================
# STEP 2: Analog centroid selection
# ==========================================================================
print("\n[Step 2] Analog centroid 전류 계산 중...")

# (96, 100, 128) → (96, 100)
I_matrix = np.zeros((96, 100))
for i in range(96):
    for j in range(100):
        diff = np.abs(Q[i] - C[j])
        I_matrix[i, j] = calc_ids(diff).sum() * 1e6

# 토큰당 전류 최소 nprobe개 centroid 선택
analog_top = np.argsort(I_matrix, axis=1)[:, :NPROBE]   # (96, nprobe)

# 저장용 DataFrame
rows_step2 = []
for i in range(96):
    q_id = i // 32
    token_name = Q_raw.index[i]
    for rank, c_id in enumerate(analog_top[i]):
        rows_step2.append({
            'token_id':   token_name,
            'query_id':   q_id,
            'rank':       rank+1,
            'centroid_id': int(c_id),
            'current_uA': round(I_matrix[i, c_id], 4),
        })

df_step2 = pd.DataFrame(rows_step2)
df_step2.to_excel(f'{BASE}/step2_analog_centroid.xlsx', index=False)
print(f"  저장: step2_analog_centroid.xlsx")

# ==========================================================================
# STEP 3: Inverted list lookup (analog 선택 centroid 기준)
# ==========================================================================
print("\n[Step 3] Analog inverted list 조회 중...")

results_step3 = {}

for q_id in [0, 1, 2]:
    token_range = range(q_id*32, (q_id+1)*32)
    true_pid = QUERY_RELEVANT[q_id]

    pid_info = {}
    for i in token_range:
        token_name = Q_raw.index[i]
        for c_id in analog_top[i]:
            rows = ivf.loc[[c_id]]
            for _, row in rows.iterrows():
                pid = row['pid']
                if pid not in pid_info:
                    pid_info[pid] = {'centroid_ids': set(), 'token_ids': set()}
                pid_info[pid]['centroid_ids'].add(int(c_id))
                pid_info[pid]['token_ids'].add(token_name)

    rows_list = []
    for pid, info in pid_info.items():
        rows_list.append({
            'pid':         pid,
            'is_relevant': pid == true_pid,
            'n_centroids': len(info['centroid_ids']),
            'n_tokens':    len(info['token_ids']),
        })

    df_q = pd.DataFrame(rows_list).sort_values('n_tokens', ascending=False).reset_index(drop=True)
    results_step3[f'q{q_id}'] = df_q

    found = true_pid in df_q['pid'].values
    print(f"  [q{q_id}] 후보 {len(df_q)}개, 정답 pid {true_pid} 포함: {found}")

with pd.ExcelWriter(f'{BASE}/step3_analog_candidate_pids.xlsx', engine='openpyxl') as writer:
    for q_name, df_r in results_step3.items():
        df_r.to_excel(writer, sheet_name=q_name, index=False)
print(f"  저장: step3_analog_candidate_pids.xlsx")

# ==========================================================================
# STEP 6: MaxSim (4가지)
# ==========================================================================
print("\n[Step 6] MaxSim / Analog MinCurrent 계산 중...")

def digital_maxsim(Q_q, D_pid):
    sim = Q_q @ D_pid.T
    return sim.max(axis=1).sum()

def analog_mincurrent(Q_q, D_pid):
    I = analog_total_current(Q_q[:, np.newaxis, :], D_pid[np.newaxis, :, :])  # (32, n_tok)
    return I.min(axis=1).sum()

all_results = {}

for q_id in [0, 1, 2]:
    Q_q      = Q[q_id*32:(q_id+1)*32]          # (32, 128)
    true_pid = QUERY_RELEVANT[q_id]

    # 디지털/아날로그 각각 자기 후보 집합 사용
    digital_pids = pd.read_excel(f'{BASE}/step3_candidate_pids.xlsx',
                                 sheet_name=f'q{q_id}')['pid'].tolist()
    analog_pids  = results_step3[f'q{q_id}']['pid'].tolist()

    # digital 후보 기준 랭킹
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

    # analog 후보 기준 랭킹
    ana_rows = []
    for pid in analog_pids:
        token_keys = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not token_keys:
            continue
        idxs = [doc_token_index[k] for k in token_keys]
        ana_rows.append({
            'pid':              pid,
            'is_relevant':      pid == true_pid,
            'current_ana_f32':  round(analog_mincurrent(Q_q, D_f32[idxs]), 4),
            'current_ana_2bt':  round(analog_mincurrent(Q_q, D_2bt[idxs]), 4),
        })
    df_ana = pd.DataFrame(ana_rows)
    df_ana['rank_ana_f32'] = df_ana['current_ana_f32'].rank(ascending=True).astype(int)
    df_ana['rank_ana_2bt'] = df_ana['current_ana_2bt'].rank(ascending=True).astype(int)

    all_results[f'q{q_id}'] = {'digital': df_dig, 'analog': df_ana}

    print(f"\n  [q{q_id}] 정답 pid: {true_pid}")
    rel_dig = df_dig[df_dig['is_relevant']]
    rel_ana = df_ana[df_ana['is_relevant']]
    if len(rel_dig):
        r = rel_dig.iloc[0]
        print(f"    digital  float32 → score: {r['score_dig_f32']:7.4f},  rank: {r['rank_dig_f32']}/{len(df_dig)}")
        print(f"    digital  2bit    → score: {r['score_dig_2bt']:7.4f},  rank: {r['rank_dig_2bt']}/{len(df_dig)}")
    if len(rel_ana):
        r = rel_ana.iloc[0]
        print(f"    analog   float32 → current: {r['current_ana_f32']:7.4f} uA, rank: {r['rank_ana_f32']}/{len(df_ana)}")
        print(f"    analog   2bit    → current: {r['current_ana_2bt']:7.4f} uA, rank: {r['rank_ana_2bt']}/{len(df_ana)}")

# -- 저장 -------------------------------------------------------------------
out_path = f'{BASE}/step6_all_results.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    for q_id in [0, 1, 2]:
        all_results[f'q{q_id}']['digital'].sort_values('rank_dig_f32').to_excel(
            writer, sheet_name=f'q{q_id}_digital', index=False)
        all_results[f'q{q_id}']['analog'].sort_values('rank_ana_f32').to_excel(
            writer, sheet_name=f'q{q_id}_analog', index=False)

    summary = []
    for q_id in [0, 1, 2]:
        df_dig = all_results[f'q{q_id}']['digital']
        df_ana = all_results[f'q{q_id}']['analog']
        rel_dig = df_dig[df_dig['is_relevant']]
        rel_ana = df_ana[df_ana['is_relevant']]
        row = {
            'query':           f'q{q_id}',
            'true_pid':        QUERY_RELEVANT[q_id],
            'n_cands_digital': len(df_dig),
            'n_cands_analog':  len(df_ana),
        }
        if len(rel_dig):
            r = rel_dig.iloc[0]
            row.update({'rank_dig_f32': int(r['rank_dig_f32']),
                        'rank_dig_2bt': int(r['rank_dig_2bt']),
                        'score_dig_f32': r['score_dig_f32'],
                        'score_dig_2bt': r['score_dig_2bt']})
        if len(rel_ana):
            r = rel_ana.iloc[0]
            row.update({'rank_ana_f32': int(r['rank_ana_f32']),
                        'rank_ana_2bt': int(r['rank_ana_2bt']),
                        'current_ana_f32': r['current_ana_f32'],
                        'current_ana_2bt': r['current_ana_2bt']})
        summary.append(row)

    pd.DataFrame(summary).to_excel(writer, sheet_name='summary', index=False)

print(f"\nSaved: {out_path}")
