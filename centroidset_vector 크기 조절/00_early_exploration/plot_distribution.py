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

cent  = pd.read_excel(f'{BASE}/centroids_100x128_clip.xlsx',  index_col=0).values.astype(float).flatten()
query = pd.read_excel(f'{BASE}/query_embs_96x128_clip.xlsx',  index_col=0).values.astype(float).flatten()

bins = np.arange(-1.02, 1.04, 0.02)   # 구간 0.02 단위

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, vals, title, color in [
    (axes[0], cent,  f'centroids_clip  (100×128 = {len(cent):,}개)', '#2196F3'),
    (axes[1], query, f'query_embs_clip  (96×128 = {len(query):,}개)',  '#E91E63'),
]:
    counts, edges, patches = ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)

    # ±1 근방 강조
    for patch, left in zip(patches, edges[:-1]):
        if abs(left) >= 0.98:
            patch.set_facecolor('#FF5722')
            patch.set_alpha(1.0)

    ax.axvline(-1, color='red', linestyle='--', linewidth=1.2, label='±1')
    ax.axvline( 1, color='red', linestyle='--', linewidth=1.2)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('count')
    ax.legend(fontsize=10)

    # 극단값 개수 표시
    n_pos1 = int((vals >= 0.98).sum())
    n_neg1 = int((vals <= -0.98).sum())
    ax.text(0.98, 0.85, f'|x|≥0.98: {n_pos1+n_neg1}개',
            transform=ax.transAxes, ha='right', fontsize=9, color='#FF5722')

axes[1].set_xlabel('값', fontsize=11)
axes[1].set_xlim(-1.05, 1.05)

plt.suptitle('Clip 버전 값 분포 (bin=0.02)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}/distribution_clip.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장: distribution_clip.png")
