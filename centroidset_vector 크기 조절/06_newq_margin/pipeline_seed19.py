"""
newq_margin pool(200 docs, rank2 강제포함) 그대로, Vth SEED=19(40-seed 스윕에서 q0 No Comp rank=8로
가장 크게 밀렸던 worst case)로 전체 Step2/3/6을 다시 계산해 step3/step6 결과를 저장.
plot_seed19.py에서 이 결과로 scatter plot을 그림.
"""

import pandas as pd
import numpy as np
from scipy.stats import truncnorm

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
NPROBE         = 2
QUERY_RELEVANT = {0: 471602, 1: 822108, 2: 347885}
SEED           = 19

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
rng = np.random.default_rng(seed=SEED)

def sample_vth(shape):
    shift = truncnorm.rvs(_a, _b, loc=0.0, scale=VTH_STD, size=shape, random_state=rng)
    return Vth_orig + shift

print("데이터 로드 중...")
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

doc_token_index = {tok: i for i, tok in enumerate(doc.index)}
rank_cols       = [f'rank_{i+1}' for i in range(100)]
rank_order_arr  = qcr[rank_cols].values

NC, NT, DIM = C.shape[0], D.shape[0], D.shape[1]
print(f"  NC={NC}  NT={NT}  DIM={DIM}  SEED={SEED}")

vth_m0_cent = sample_vth((NC, DIM))
vth_m3_cent = sample_vth((NC, DIM))
vth_m0_doc  = sample_vth((NT, DIM))
vth_m3_doc  = sample_vth((NT, DIM))

print("\n[Step 2] Centroid 선택 중...")
digital_top = rank_order_arr[:, :NPROBE]

d_step2   = Q[:, np.newaxis, :] - C[np.newaxis, :, :]
I_optA_uA = cell_optA(d_step2, Vth_orig, Vth_orig).sum(axis=2) * 1e-3
optA_top  = np.argsort(I_optA_uA, axis=1)[:, :NPROBE]

vm0_b = vth_m0_cent[np.newaxis, :, :]
vm3_b = vth_m3_cent[np.newaxis, :, :]
I_optC_uA = cell_optC(d_step2, vm0_b, vm3_b).sum(axis=2) * 1e-3
optC_top  = np.argsort(I_optC_uA, axis=1)[:, :NPROBE]

print("\n[Step 3] IVF lookup 중...")
def ivf_lookup(top_arr, method_name):
    results = {}
    for q_id in [0, 1, 2]:
        true_pid = QUERY_RELEVANT[q_id]
        pid_info = {}
        for i in range(q_id * 32, (q_id + 1) * 32):
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
        rows_list = [{'pid': pid, 'is_relevant': pid == true_pid,
                      'n_centroids': len(info['centroid_ids']), 'n_tokens': len(info['token_ids'])}
                     for pid, info in pid_info.items()]
        df_q = pd.DataFrame(rows_list).sort_values('n_tokens', ascending=False).reset_index(drop=True)
        results[f'q{q_id}'] = df_q
        print(f"  [{method_name} q{q_id}] 후보 {len(df_q)}개  정답 포함: {true_pid in df_q['pid'].values}")
    out = f'{BASE}/[seed19]step3_{method_name}_candidate_pids.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        for q_name, df_r in results.items():
            df_r.to_excel(w, sheet_name=q_name, index=False)
    return results

cands_digital = ivf_lookup(digital_top, 'digital')
cands_optA    = ivf_lookup(optA_top,    'optA')
cands_optC    = ivf_lookup(optC_top,    'optC')

print("\n[Step 6] 점수 계산 중...")
def digital_maxsim(Q_q, D_pid):
    return float((Q_q @ D_pid.T).max(axis=1).sum())

def mincurrent_optA(Q_q, D_pid):
    d = Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :]
    return float((cell_optA(d, Vth_orig, Vth_orig).sum(axis=2) * 1e-3).min(axis=1).sum())

def mincurrent_optC(Q_q, D_pid, vth_m0_tok, vth_m3_tok):
    d = Q_q[:, np.newaxis, :] - D_pid[np.newaxis, :, :]
    I_uA = cell_optC(d, vth_m0_tok[np.newaxis, :, :], vth_m3_tok[np.newaxis, :, :]).sum(axis=2) * 1e-3
    return float(I_uA.min(axis=1).sum())

all_results = {}
for q_id in [0, 1, 2]:
    Q_q = Q[q_id * 32:(q_id + 1) * 32]
    true_pid = QUERY_RELEVANT[q_id]

    dig_rows = []
    for pid in cands_digital[f'q{q_id}']['pid'].tolist():
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_token_index[k] for k in toks]
        dig_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                         'score_dig_f32': round(digital_maxsim(Q_q, D[idxs]), 4)})
    df_dig = pd.DataFrame(dig_rows)
    df_dig['rank_dig_f32'] = df_dig['score_dig_f32'].rank(ascending=False).astype(int)

    optA_rows = []
    for pid in cands_optA[f'q{q_id}']['pid'].tolist():
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_token_index[k] for k in toks]
        optA_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                          'current_optA_uA': round(mincurrent_optA(Q_q, D[idxs]), 4)})
    df_optA = pd.DataFrame(optA_rows)
    df_optA['rank_optA'] = df_optA['current_optA_uA'].rank(ascending=True).astype(int)

    optC_rows = []
    for pid in cands_optC[f'q{q_id}']['pid'].tolist():
        toks = [k for k in doc.index if k.startswith(f'{pid}_t')]
        if not toks: continue
        idxs = [doc_token_index[k] for k in toks]
        vth_m0_tok, vth_m3_tok = vth_m0_doc[idxs], vth_m3_doc[idxs]
        optC_rows.append({'pid': pid, 'is_relevant': pid == true_pid,
                          'current_optC_uA': round(mincurrent_optC(Q_q, D[idxs], vth_m0_tok, vth_m3_tok), 4)})
    df_optC = pd.DataFrame(optC_rows)
    df_optC['rank_optC'] = df_optC['current_optC_uA'].rank(ascending=True).astype(int)

    all_results[f'q{q_id}'] = {'digital': df_dig, 'optA': df_optA, 'optC': df_optC}
    print(f"\n  [q{q_id}] 정답 pid: {true_pid}")
    for label, df_r, rank_col in [('Digital f32', df_dig, 'rank_dig_f32'),
                                   ('Option A', df_optA, 'rank_optA'),
                                   ('Option C', df_optC, 'rank_optC')]:
        rel = df_r[df_r['is_relevant']]
        if len(rel):
            print(f"    {label:<12} rank: {int(rel.iloc[0][rank_col])}/{len(df_r)}")
        else:
            print(f"    {label:<12} 정답 후보 없음")

out_path = f'{BASE}/[seed19]step6_all_results.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    for q_id in [0, 1, 2]:
        r = all_results[f'q{q_id}']
        r['digital'].sort_values('rank_dig_f32').to_excel(writer, sheet_name=f'q{q_id}_digital', index=False)
        r['optA'].sort_values('rank_optA').to_excel(writer, sheet_name=f'q{q_id}_optA', index=False)
        r['optC'].sort_values('rank_optC').to_excel(writer, sheet_name=f'q{q_id}_optC', index=False)

print(f"\n저장: [seed19]step3/step6 완료")
