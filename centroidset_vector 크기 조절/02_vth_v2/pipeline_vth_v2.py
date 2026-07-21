"""
[clip99.9] ColBERT 아날로그 파이프라인 — Vth mismatch v2 모델
  Option A: M0/M3 각자 vth_stored 보유 → 개별 calibration → Vth 완전 상쇄
  Option C: 보상 없음, M0/M3 Vth 독립 샘플링 → dead zone 비대칭

셀 모델 (plot_vth_mismatch_v2.py 기준):
  I_single(vgs, vth) = 10^(B + L/(1+exp(-K*(min(vgs,VSAT)-vth)))) + max(0, vgs-VSAT)*SLOPE
  Option A: vgd_m0 = d + vth_stored_m0  (= d + vth_m0 → Vth 상쇄)
            vgd_m3 = -d + vth_stored_m3 (= -d + vth_m3 → Vth 상쇄)
  Option C: vgd_m0 = d,  vgd_m3 = -d   (보상 없음)
  VDD = 1.7V  (실제 동작 전압)

Step 2: centroid 선택 (96×100)
Step 3: IVF lookup → 후보 pid 수집
Step 6: MinCurrent 점수 → Ranking
"""

import pandas as pd
import numpy as np
from scipy.stats import truncnorm

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\02_vth_v2'
DATA_BASE      = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline'
NPROBE         = 2
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}
SEED           = 42

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

def I_single(vgs, vth):
    """단방향 전류 [A]. vgs, vth: ndarray (브로드캐스트 가능)"""
    v = np.minimum(vgs, VSAT)
    return (10 ** (B_f + L_f / (1 + np.exp(-K_f * (v - vth))))
            + np.maximum(0.0, vgs - VSAT) * SLOPE)

def _cell_current(d, vth_m0, vth_m3, vth_s_m0, vth_s_m3):
    """
    Dual 3T1C 셀 전류 [nA].
    d        : Q - D  (부호 있음, ndarray)
    vth_m0/3 : 실제 트랜지스터 Vth
    vth_s_m0/3: 저장된 보상값 (Option A: = vth_m0/3, Option C: 0)
    """
    # M0: d >= 0 구간
    vgd_m0 = d + vth_s_m0
    I_m0   = np.where(d >= 0,
                      np.maximum(I_single(vgd_m0, vth_m0)
                                 - I_single(vgd_m0 - VDD, vth_m0), 0.0),
                      0.0)
    # M3: d < 0 구간
    vgd_m3 = -d + vth_s_m3
    I_m3   = np.where(d < 0,
                      np.maximum(I_single(vgd_m3, vth_m3)
                                 - I_single(vgd_m3 - VDD, vth_m3), 0.0),
                      0.0)
    return np.clip(I_m0 + I_m3, CLIP, None) * 1e9   # nA

def cell_optA(d, vth_m0, vth_m3):
    """
    Option A — 개별 보상: vth_stored = 실제 vth → Vth 완전 상쇄.
    Vth가 얼마든 결과는 Vth_orig 고정과 동일.
    """
    return _cell_current(d, vth_m0, vth_m3, vth_s_m0=vth_m0, vth_s_m3=vth_m3)

def cell_optC(d, vth_m0, vth_m3):
    """Option C — 보상 없음: vth_stored = 0."""
    return _cell_current(d, vth_m0, vth_m3, vth_s_m0=0.0, vth_s_m3=0.0)

# ==========================================================================
# Vth 분포 (Option C용, M0/M3 독립 샘플링)
# ==========================================================================
VTH_STD = 0.15
VTH_LO, VTH_HI = 0.0, 0.5
_a = (VTH_LO - 0.0) / VTH_STD
_b = (VTH_HI - 0.0) / VTH_STD
rng = np.random.default_rng(seed=SEED)

def sample_vth(shape):
    """Vth_orig + TruncGauss[0, 0.5]V shift"""
    shift = truncnorm.rvs(_a, _b, loc=0.0, scale=VTH_STD,
                          size=shape, random_state=rng)
    return Vth_orig + shift

# ==========================================================================
# 데이터 로드
# ==========================================================================
print("데이터 로드 중...")
Q_df = pd.read_excel(f'{DATA_BASE}/[clip99.9]query_embs_96x128.xlsx',  index_col=0)
C_df = pd.read_excel(f'{DATA_BASE}/[clip99.9]centroids_100x128.xlsx',  index_col=0)
doc  = pd.read_excel(f'{DATA_BASE}/[clip99.9]doc_embs_12919x128.xlsx', index_col=0)
ivf  = pd.read_excel(f'{DATA_BASE}/[clip99.9]ivf_centroid2pid.xlsx',
                     sheet_name='long_format', index_col=0)
qcr  = pd.read_excel(f'{DATA_BASE}/[clip99.9]query_centroid_ranking.xlsx',
                     sheet_name='rank_order', index_col=0)

dim_cols = [c for c in doc.columns if c.startswith('dim_')]
Q = Q_df.values.astype(float)          # (96, 128)
C = C_df.values.astype(float)          # (100, 128)
D = doc[dim_cols].values.astype(float) # (12919, 128)

doc_token_index = {tok: i for i, tok in enumerate(doc.index)}
rank_cols       = [f'rank_{i+1}' for i in range(100)]
rank_order_arr  = qcr[rank_cols].values   # (96, 100)

# ==========================================================================
# STEP 2: Centroid 선택
# ==========================================================================
print("\n[Step 2] Centroid 선택 중...")

# --- Digital: 기존 rank_order top-2 재사용
digital_top = rank_order_arr[:, :NPROBE]   # (96, 2)
print("  Digital: rank_order top-2 완료")

# --- Option A: d = Q - C (부호 있음), Vth 개별 보상
print("  Option A: 전류 계산 중... (96×100)")
d_step2 = Q[:, np.newaxis, :] - C[np.newaxis, :, :]   # (96, 100, 128)
# Option A: vth_stored = vth → Vth 상쇄 → 실제 vth 값 무관
# Vth_orig 고정 사용 (어떤 값이어도 결과 동일)
I_optA = cell_optA(d_step2,
                   vth_m0=Vth_orig,
                   vth_m3=Vth_orig).sum(axis=2) * 1e-9   # (96, 100) [A→uA via nA]
I_optA_uA = I_optA * 1e3   # nA → uA... wait, cell returns nA already
# cell_optA returns nA, sum over 128 dims → nA, ×1e-3 → uA
I_optA_uA = cell_optA(d_step2, Vth_orig, Vth_orig).sum(axis=2) * 1e-3  # uA
optA_top  = np.argsort(I_optA_uA, axis=1)[:, :NPROBE]
print("  Option A 완료")

# --- Option C: M0/M3 Vth 독립 샘플링
print("  Option C: 전류 계산 중... (96×100)")
Vth_M0_s2 = sample_vth((96, 100))[:, :, np.newaxis]   # (96, 100, 1)
Vth_M3_s2 = sample_vth((96, 100))[:, :, np.newaxis]   # (96, 100, 1)
I_optC_uA = cell_optC(d_step2, Vth_M0_s2, Vth_M3_s2).sum(axis=2) * 1e-3  # uA
optC_top  = np.argsort(I_optC_uA, axis=1)[:, :NPROBE]
print("  Option C 완료")

# --- Step 2 저장
def save_step2(top_arr, I_uA, name):
    rows = []
    for i in range(96):
        for rank, c_id in enumerate(top_arr[i]):
            rows.append({
                'token_id':    Q_df.index[i],
                'query_id':    i // 32,
                'rank':        rank + 1,
                'centroid_id': int(c_id),
                'current_uA':  round(float(I_uA[i, c_id]), 6) if I_uA is not None else None,
            })
    out = f'{BASE}/[vth_v2]step2_{name}_centroid.xlsx'
    pd.DataFrame(rows).to_excel(out, index=False)
    print(f"  저장: [vth_v2]step2_{name}_centroid.xlsx")

save_step2(digital_top, None,       'digital')
save_step2(optA_top,    I_optA_uA,  'optA')
save_step2(optC_top,    I_optC_uA,  'optC')

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
            for c_id in top_arr[i]:
                c_id = int(c_id)
                if c_id not in ivf.index:
                    continue
                for _, row in ivf.loc[[c_id]].iterrows():
                    pid = int(row['pid'])
                    if pid not in pid_info:
                        pid_info[pid] = {'centroid_ids': set(), 'token_ids': set()}
                    pid_info[pid]['centroid_ids'].add(c_id)
                    pid_info[pid]['token_ids'].add(Q_df.index[i])

        rows_list = [{'pid':         pid,
                      'is_relevant': pid == true_pid,
                      'n_centroids': len(info['centroid_ids']),
                      'n_tokens':    len(info['token_ids'])}
                     for pid, info in pid_info.items()]
        df_q = (pd.DataFrame(rows_list)
                  .sort_values('n_tokens', ascending=False)
                  .reset_index(drop=True))
        results[f'q{q_id}'] = df_q
        print(f"  [{method_name} q{q_id}] 후보 {len(df_q)}개  정답 포함: {true_pid in df_q['pid'].values}")

    out = f'{BASE}/[vth_v2]step3_{method_name}_candidate_pids.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        for q_name, df_r in results.items():
            df_r.to_excel(w, sheet_name=q_name, index=False)
    print(f"  저장: [vth_v2]step3_{method_name}_candidate_pids.xlsx")
    return results

cands_digital = ivf_lookup(digital_top, 'digital')
cands_optA    = ivf_lookup(optA_top,    'optA')
cands_optC    = ivf_lookup(optC_top,    'optC')

# ==========================================================================
# STEP 6: MinCurrent → Ranking
# ==========================================================================
print("\n[Step 6] 점수 계산 중...")

def digital_maxsim(Q_q, D_pid):
    return float((Q_q @ D_pid.T).max(axis=1).sum())

def mincurrent_optA(Q_q, D_pid):
    """Option A: Vth 개별 보상 → Vth 무관, Vth_orig 사용"""
    d    = Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :]   # (32, M, 128)
    I_nA = cell_optA(d, Vth_orig, Vth_orig)                   # (32, M, 128)
    I_uA = I_nA.sum(axis=2) * 1e-3                            # (32, M) [uA]
    return float(I_uA.min(axis=1).sum())

def mincurrent_optC(Q_q, D_pid, vth_m0, vth_m3):
    """
    Option C: M0/M3 Vth 독립 샘플링
    vth_m0, vth_m3: (32, M) → (32, M, 1) broadcast
    """
    d    = Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :]   # (32, M, 128)
    I_nA = cell_optC(d,
                     vth_m0[:, :, np.newaxis],
                     vth_m3[:, :, np.newaxis])                 # (32, M, 128)
    I_uA = I_nA.sum(axis=2) * 1e-3                            # (32, M) [uA]
    return float(I_uA.min(axis=1).sum())

all_results = {}

for q_id in [0, 1, 2]:
    Q_q      = Q[q_id * 32:(q_id + 1) * 32]   # (32, 128)
    true_pid = QUERY_RELEVANT[q_id]

    # --- Digital MaxSim (digital 후보)
    dig_rows = []
    for pid in cands_digital[f'q{q_id}']['pid'].tolist():
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_token_index[k] for k in toks]
        dig_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                         'score_dig_f32': round(digital_maxsim(Q_q, D[idxs]), 4)})
    df_dig = pd.DataFrame(dig_rows)
    df_dig['rank_dig_f32'] = df_dig['score_dig_f32'].rank(ascending=False).astype(int)

    # --- Option A MinCurrent (optA 후보)
    optA_rows = []
    for pid in cands_optA[f'q{q_id}']['pid'].tolist():
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_token_index[k] for k in toks]
        optA_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                          'current_optA_uA': round(mincurrent_optA(Q_q, D[idxs]), 4)})
    df_optA = pd.DataFrame(optA_rows)
    df_optA['rank_optA'] = df_optA['current_optA_uA'].rank(ascending=True).astype(int)

    # --- Option C MinCurrent (optC 후보, M0/M3 Vth 독립 샘플링)
    optC_rows = []
    for pid in cands_optC[f'q{q_id}']['pid'].tolist():
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs  = [doc_token_index[k] for k in toks]
        M_tok = len(idxs)
        vth_m0 = sample_vth((32, M_tok))   # (32, M_tok)
        vth_m3 = sample_vth((32, M_tok))   # (32, M_tok) — M3 독립 샘플링
        optC_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                          'current_optC_uA': round(mincurrent_optC(Q_q, D[idxs], vth_m0, vth_m3), 4)})
    df_optC = pd.DataFrame(optC_rows)
    df_optC['rank_optC'] = df_optC['current_optC_uA'].rank(ascending=True).astype(int)

    all_results[f'q{q_id}'] = {'digital': df_dig, 'optA': df_optA, 'optC': df_optC}

    print(f"\n  [q{q_id}] 정답 pid: {true_pid}")
    for label, df_r, rank_col in [
        ('Digital f32', df_dig,  'rank_dig_f32'),
        ('Option A',    df_optA, 'rank_optA'),
        ('Option C',    df_optC, 'rank_optC'),
    ]:
        rel = df_r[df_r['is_relevant']]
        if len(rel):
            print(f"    {label:<12} rank: {int(rel.iloc[0][rank_col])}/{len(df_r)}")
        else:
            print(f"    {label:<12} 정답 후보 없음")

# ==========================================================================
# 저장
# ==========================================================================
out_path = f'{BASE}/[vth_v2]step6_all_results.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    summary = []
    for q_id in [0, 1, 2]:
        r = all_results[f'q{q_id}']
        r['digital'].sort_values('rank_dig_f32').to_excel(writer, sheet_name=f'q{q_id}_digital', index=False)
        r['optA'].sort_values('rank_optA').to_excel(writer,        sheet_name=f'q{q_id}_optA',    index=False)
        r['optC'].sort_values('rank_optC').to_excel(writer,        sheet_name=f'q{q_id}_optC',    index=False)

        row = {'query': f'q{q_id}', 'true_pid': QUERY_RELEVANT[q_id],
               'n_digital': len(r['digital']),
               'n_optA':    len(r['optA']),
               'n_optC':    len(r['optC'])}
        for label, df_r, rank_col in [
            ('digital', r['digital'], 'rank_dig_f32'),
            ('optA',    r['optA'],    'rank_optA'),
            ('optC',    r['optC'],    'rank_optC'),
        ]:
            rel = df_r[df_r['is_relevant']]
            if len(rel):
                row[f'rank_{label}'] = int(rel.iloc[0][rank_col])
        summary.append(row)

    pd.DataFrame(summary).to_excel(writer, sheet_name='summary', index=False)

print(f"\n저장: [vth_v2]step6_all_results.xlsx")
print("\nPipeline 완료!")
