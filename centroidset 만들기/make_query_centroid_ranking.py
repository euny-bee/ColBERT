import os
import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\nmdl-khb\ColBERT\centroidset 만들기"

# ── Step 1: 데이터 로드 ────────────────────────────────────────────────────
print("=== Step 1: 데이터 로드 ===")
Q = np.load(os.path.join(OUT_DIR, "query_embs_96x128.npy"))    # (96, 128)
C = np.load(os.path.join(OUT_DIR, "centroids_100x128.npy"))    # (100, 128)
print(f"  Q (query embeddings): {Q.shape}")
print(f"  C (centroids):        {C.shape}")

# ── Step 2: Dot product ────────────────────────────────────────────────────
print("\n=== Step 2: Dot product (Q @ C.T) ===")
scores = Q @ C.T    # (96, 100)
print(f"  scores shape: {scores.shape}")
print(f"  scores range: [{scores.min():.4f}, {scores.max():.4f}]")

# ── Step 3: Rank 계산 (내림차순, rank 1 = 가장 가까운 centroid) ────────────
print("\n=== Step 3: Rank 계산 ===")
ranks = scores.shape[1] - scores.argsort(axis=1).argsort(axis=1)
print(f"  ranks shape: {ranks.shape}")
print(f"  rank range: [{ranks.min()}, {ranks.max()}]")
assert (np.sort(ranks, axis=1) == np.arange(1, 101)).all(), "rank 값 오류"
print(f"  rank 검증 통과: 각 행에 1~100이 정확히 1번씩 등장")

# ── Step 4: 행/열 이름 ────────────────────────────────────────────────────
row_names = [f"q{q}_t{t}" for q in range(3) for t in range(32)]
query_ids = [q for q in range(3) for _ in range(32)]
col_names = [f"centroid_{i}" for i in range(100)]

# ── Step 5: 저장 ───────────────────────────────────────────────────────────
print("\n=== Step 5: 저장 ===")

def make_df(data, row_names, query_ids, col_names):
    df = pd.DataFrame(data, columns=col_names)
    df.insert(0, "query_id", query_ids)
    df.index = row_names
    df.index.name = "token_id"
    return df

df_scores = make_df(scores, row_names, query_ids, col_names)
df_ranks  = make_df(ranks,  row_names, query_ids, col_names)

# CSV
df_scores.to_csv(os.path.join(OUT_DIR, "query_centroid_ranking_scores.csv"))
df_ranks.to_csv(os.path.join(OUT_DIR, "query_centroid_ranking_ranks.csv"))
print(f"  CSV 저장 완료")

# rank_order: 각 query token에 대해 rank1=몇번 centroid, rank2=몇번 centroid ...
# scores 내림차순 argsort → centroid ID 순서
order = np.argsort(-scores, axis=1)   # (96, 100), 각 행: score 높은순 centroid 인덱스
rank_col_names = [f"rank_{i+1}" for i in range(100)]
df_order = pd.DataFrame(order, columns=rank_col_names)
df_order.insert(0, "query_id", query_ids)
df_order.index = row_names
df_order.index.name = "token_id"

# Excel (3 sheets)
with pd.ExcelWriter(os.path.join(OUT_DIR, "query_centroid_ranking.xlsx"), engine="openpyxl") as writer:
    df_scores.to_excel(writer, sheet_name="scores")
    df_ranks.to_excel(writer, sheet_name="ranks")
    df_order.to_excel(writer, sheet_name="rank_order")
print(f"  query_centroid_ranking.xlsx 저장 완료")
print(f"    Sheet1 'scores':      dot product 값 (96 x 100)")
print(f"    Sheet2 'ranks':       rank 1~100 (centroid 기준)")
print(f"    Sheet3 'rank_order':  rank1=몇번, rank2=몇번 ... (token 기준)")

print("\n=== 완료 ===")
