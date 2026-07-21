import pandas as pd
import numpy as np

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\00_early_exploration'

ivf_orig   = pd.read_excel(f'{BASE}/ivf_centroid2pid.xlsx',        index_col=0)
ivf_global = pd.read_excel(f'{BASE}/ivf_centroid2pid_global.xlsx', index_col=0)
ivf_clip   = pd.read_excel(f'{BASE}/ivf_centroid2pid_clip.xlsx',   index_col=0)

def to_set(df):
    return set(zip(df.index.tolist(), df['pid'].tolist()))

orig_set   = to_set(ivf_orig)
global_set = to_set(ivf_global)
clip_set   = to_set(ivf_clip)

# global vs 원본
diff_go = orig_set.symmetric_difference(global_set)
print("ivf_global vs 원본: 차이", len(diff_go), "개")
if len(diff_go) == 0:
    print("  -> 완전 일치 (uniform scaling L2 ranking 불변 확인)")

# clip vs 원본
in_orig_not_clip = orig_set - clip_set
in_clip_not_orig = clip_set - orig_set
print("ivf_clip vs 원본:")
print("  원본에만 있는 pairs:", len(in_orig_not_clip))
print("  clip에만 있는 pairs:", len(in_clip_not_orig))
if in_orig_not_clip:
    print("  예시(원본->clip 달라진):", list(in_orig_not_clip)[:3])

# clip된 값 통계
doc_df = pd.read_excel(f'{BASE}/doc_embs_12919x128.xlsx', index_col=0)
dim_cols = [c for c in doc_df.columns if c.startswith('dim_')]
D = doc_df[dim_cols].values.astype(float)
cent_df = pd.read_excel(f'{BASE}/centroids_100x128.xlsx', index_col=0)
C = cent_df.values.astype(float)

clip_thr  = float(np.abs(C).max())
n_clipped = int((np.abs(D) > clip_thr).sum())
print("clip된 doc dim 값 수:", n_clipped, "/", D.size, "=", round(n_clipped/D.size*100, 4), "%")

# centroid 재배정 토큰 수
def get_assignments(D_emb, C_emb):
    D_n = (D_emb**2).sum(axis=1, keepdims=True)
    C_n = (C_emb**2).sum(axis=1)
    dist = D_n + C_n[np.newaxis, :] - 2*(D_emb @ C_emb.T)
    return np.argmin(dist, axis=1)

max_doc = float(np.abs(D).max())
assign_global = get_assignments(D * (1/max_doc),    C * (1/max_doc))
assign_clip   = get_assignments(np.clip(D,-clip_thr,clip_thr) * (1/clip_thr), C * (1/clip_thr))
reassigned    = int((assign_global != assign_clip).sum())
print("centroid 재배정 달라진 토큰:", reassigned, "/", len(assign_global))
