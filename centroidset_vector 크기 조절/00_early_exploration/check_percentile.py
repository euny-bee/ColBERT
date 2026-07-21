import pandas as pd
import numpy as np

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\00_early_exploration'

files = {
    'centroids_clip' : 'centroids_100x128_clip.xlsx',
    'query_embs_clip': 'query_embs_96x128_clip.xlsx',
}

for name, fname in files.items():
    vals = pd.read_excel(f'{BASE}/{fname}', index_col=0).values.astype(float).flatten()
    lo = np.percentile(vals, 0.5)
    hi = np.percentile(vals, 99.5)
    abs_99 = np.percentile(np.abs(vals), 99)
    print(f'{name}:')
    print(f'  99% 범위: [{lo:.4f}, {hi:.4f}]')
    print(f'  |x| 기준 99th percentile: +-{abs_99:.4f}')
    print()
