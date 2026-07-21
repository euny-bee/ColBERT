import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.font_manager as fm

available = {f.name for f in fm.fontManager.ttflist}
sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": sans, "font.size": 13,
    "font.weight": "bold", "axes.labelsize": 18, "axes.labelweight": "bold",
    "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.unicode_minus": False,
})

IN_BASE = r"C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin"
OUT = r"C:\Users\nmdl-khb\AppData\Local\Temp\claude\c--Users-nmdl-khb-ColBERT-TOPSW\20aa624b-a34d-44aa-8475-8732670e3e23\scratchpad\scatter_violin_TOPSW_blue.png"

# swapped: 'optA' now uses the TOPSW "this work" blue (#1565C0) instead of #2196F3
C3 = {'optA': '#1565C0', 'optC': '#FF7811'}

step6 = pd.read_excel(f'{IN_BASE}/[seed19]step6_all_results.xlsx', sheet_name=None)
q_id = 2
df_dig = step6[f'q{q_id}_digital'].set_index('pid')
df_A   = step6[f'q{q_id}_optA'].set_index('pid')
df_C   = step6[f'q{q_id}_optC'].set_index('pid')

common_A = [p for p in df_dig.index if p in df_A.index]
xA = np.array([int(df_dig.loc[p, 'rank_dig_f32']) for p in common_A])
yA = np.array([int(df_A.loc[p, 'rank_optA'])      for p in common_A])

common_C = [p for p in df_dig.index if p in df_C.index]
xC = np.array([int(df_dig.loc[p, 'rank_dig_f32']) for p in common_C])
yC = np.array([int(df_C.loc[p, 'rank_optC'])      for p in common_C])

resA = yA - xA
resC = yC - xC

fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.0))

ax.scatter(xA, yA, c=C3['optA'], s=22, alpha=0.85, zorder=4, clip_on=False, label='Vth comp')
ax.scatter(xC, yC, c=C3['optC'], s=22, alpha=0.85, zorder=4, marker='D', clip_on=False, label='No comp')

max_rank = max(list(xA) + list(yA) + list(xC) + list(yC) + [10])
lim = max_rank * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.4, label='y = x')
ax.set_xlabel('Digital rank')
ax.set_ylabel('Analog rank')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.grid(True, ls='--', alpha=0.3, lw=0.8)
ax.legend(loc='upper left', fontsize=13, frameon=False)

# inset violin (bottom-right), matching original composite layout
axins = inset_axes(ax, width="42%", height="42%", loc='lower right', borderpad=2.2)
data = [resA, resC]
colors = [C3['optA'], C3['optC']]
parts = axins.violinplot(data, positions=[0, 1], widths=0.7, showextrema=False)
for pc, col in zip(parts['bodies'], colors):
    pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor(col); pc.set_linewidth(1.2)
bp = axins.boxplot(data, positions=[0, 1], widths=0.15, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', lw=1.5))
for box, col in zip(bp['boxes'], colors):
    box.set_facecolor('white'); box.set_edgecolor(col); box.set_linewidth(1.2)
rng = np.random.default_rng(0)
for i, (d, col) in enumerate(zip(data, colors)):
    jitter = rng.uniform(-0.06, 0.06, size=len(d))
    axins.scatter(np.full(len(d), i) + jitter, d, s=8, color=col, alpha=0.5, zorder=3)
axins.axhline(0, color='k', ls='--', lw=0.7, alpha=0.4)
axins.set_xticks([0, 1]); axins.set_xticklabels(['Vth\ncomp', 'No\ncomp'], fontsize=10)
axins.tick_params(axis='y', labelsize=9)
axins.set_ylabel('ΔRank (analog-digital)', fontsize=9, labelpad=1)
for sp in axins.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.8)

axins.text(0.5, -0.38, f"Std dev\nVth comp = {resA.std(ddof=1):.1f}\nNo comp = {resC.std(ddof=1):.1f}",
           transform=axins.transAxes, fontsize=8, fontweight='bold', va='top', ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.6', alpha=0.9))

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT}")
