import pandas as pd
import numpy as np

BASE = '.'
Q_df = pd.read_excel(f'{BASE}/[clip99.9]query_embs_96x128.xlsx', index_col=0)
C_df = pd.read_excel(f'{BASE}/[clip99.9]centroids_100x128.xlsx', index_col=0)

Q = Q_df.values.astype(np.float64)
C = C_df.values.astype(np.float64)
print("Q shape", Q.shape, "C shape", C.shape)
print("Q range", Q.min(), Q.max())
print("C range", C.min(), C.max())

diff = Q[:,None,:] - C[None,:,:]
diff = diff.flatten()
print("diff count", diff.size)
print("diff min/max", diff.min(), diff.max())
print("diff mean/std", diff.mean(), diff.std())
for p in [1,5,25,50,75,95,99]:
    print(f"  pct{p}: {np.percentile(diff,p):.4f}")

absdiff = np.abs(diff)
print("absdiff mean/median/95pct/99pct/max:", absdiff.mean(), np.median(absdiff), np.percentile(absdiff,95), np.percentile(absdiff,99), absdiff.max())

np.save('q2_diff_flat.npy', diff)
