import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\00_early_exploration'
DATA_BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline'

cent  = pd.read_excel(f'{DATA_BASE}/[clip99.9]centroids_100x128.xlsx',  index_col=0).values.astype(float).flatten()
query = pd.read_excel(f'{DATA_BASE}/[clip99.9]query_embs_96x128.xlsx',  index_col=0).values.astype(float).flatten()
doc_df = pd.read_excel(f'{DATA_BASE}/[clip99.9]doc_embs_12919x128.xlsx', index_col=0)
dim_cols = [c for c in doc_df.columns if c.startswith('dim_')]
doc   = doc_df[dim_cols].values.astype(float).flatten()

bins = np.arange(-1.02, 1.04, 0.01)   # bin 0.01 단위

datasets = [
    (cent,  f'centroids  (100x128 = {len(cent):,}개)',  '#2196F3'),
    (doc,   f'doc_embs dims  (12919x128 = {len(doc):,}개)', '#4CAF50'),
    (query, f'query_embs  (96x128 = {len(query):,}개)',  '#E91E63'),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

for ax, (vals, title, color) in zip(axes, datasets):
    counts, edges, patches = ax.hist(vals, bins=bins, color=color, alpha=0.85,
                                     edgecolor='white', linewidth=0.2)

    # ±1 근방 bin 강조
    for patch, left in zip(patches, edges[:-1]):
        if abs(left + 0.005) >= 0.99:
            patch.set_facecolor('#FF5722')
            patch.set_alpha(1.0)

    ax.axvline(-1, color='red', linestyle='--', linewidth=1.3, label='±1')
    ax.axvline( 1, color='red', linestyle='--', linewidth=1.3)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('count')
    ax.legend(fontsize=9, loc='upper left')

    n_at_boundary = int((np.abs(vals) >= 0.99).sum())
    pct = n_at_boundary / len(vals) * 100
    ax.text(0.99, 0.88, f'|x|>=0.99: {n_at_boundary}개 ({pct:.3f}%)',
            transform=ax.transAxes, ha='right', fontsize=9, color='#FF5722')

axes[-1].set_xlabel('값', fontsize=11)
axes[-1].set_xlim(-1.05, 1.05)

plt.suptitle('[clip99.9] 값 분포 (bin=0.01)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}/distribution_clip99.9.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장: distribution_clip99.9.png")
