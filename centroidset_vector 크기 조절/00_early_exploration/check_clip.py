import pandas as pd
import numpy as np

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\00_early_exploration'
df = pd.read_excel(f'{BASE}/centroids_100x128_clip.xlsx', index_col=0)
vals = df.values.astype(float).flatten()
print("min   :", vals.min())
print("max   :", vals.max())
print("max_abs:", np.abs(vals).max())
print()
for p in [50, 75, 90, 95, 99, 100]:
    print(f"  {p}th percentile (abs): {np.percentile(np.abs(vals), p):.4f}")
