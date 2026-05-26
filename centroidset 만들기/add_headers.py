import numpy as np
import pandas as pd
import os

OUT_DIR = r"C:\Users\nmdl-khb\ColBERT\centroidset 만들기"

# ── Centroid set: 100 × 128 ────────────────────────────────────────────────
centroids = np.load(os.path.join(OUT_DIR, "centroids_100x128.npy"))

col_names = [f"dim_{i}" for i in range(128)]
row_names = [f"centroid_{i}" for i in range(100)]

df_centroids = pd.DataFrame(centroids, index=row_names, columns=col_names)
df_centroids.index.name = "centroid_id"
df_centroids.to_csv(os.path.join(OUT_DIR, "centroids_100x128.csv"))
df_centroids.to_excel(os.path.join(OUT_DIR, "centroids_100x128.xlsx"))
print(f"centroids saved: {df_centroids.shape}")

# ── Query embeddings: 96 × 128 ─────────────────────────────────────────────
query_embs = np.load(os.path.join(OUT_DIR, "query_embs_96x128.npy"))

# 3 queries × 32 tokens → q0_t0 ~ q2_t31
row_names_q = [f"q{q}_t{t}" for q in range(3) for t in range(32)]

df_query = pd.DataFrame(query_embs, index=row_names_q, columns=col_names)
df_query.index.name = "query_token"
df_query.to_csv(os.path.join(OUT_DIR, "query_embs_96x128.csv"))
df_query.to_excel(os.path.join(OUT_DIR, "query_embs_96x128.xlsx"))
print(f"query_embs saved: {df_query.shape}")
