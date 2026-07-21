"""
MRR@10: Vth Compensation (This Work) vs No Compensation
Output:
  [graph_optC]mrr_final.png
  [graph_optC]mrr_legend.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
from matplotlib.transforms import ScaledTranslation
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

available = {f.name for f in fm.fontManager.ttflist}
sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    sans,
    "font.size":          13,
    "font.weight":        "bold",
    "axes.labelsize":     16.56,
    "axes.labelweight":   "bold",
    "axes.titlesize":     16,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    16.56,
    "ytick.labelsize":    16.56,
    "axes.unicode_minus": False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

pbs_labels = ["[0, 0.5]", "[0, 1.0]", "[0, 2.0]", "[0, 3.0]"]
x = np.arange(4)

vth_comp = np.array([78.8, 78.8, 78.8, 78.8])
no_comp  = np.array([78.1, 71.3, 27.8,  2.1])

COLOR_COMP   = "#1565C0"   # matches combined_final.py COLOR_COMP (darker blue)
COLOR_NOCOMP = "#FF7811"   # matches scatter+violin (plot_seed19_q2_r2.py C3['optC'])

# ── 메인 figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.9, 4.6))

ax.fill_between(x, vth_comp, no_comp, alpha=0.15, color=COLOR_NOCOMP)
ax.plot(x, no_comp,  marker='D', ms=6.5,color=COLOR_NOCOMP, lw=1.5, zorder=4)
ax.plot(x, vth_comp, marker='o', ms=12, color=COLOR_COMP,   lw=4.5, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(pbs_labels, fontsize=16.56, fontweight="bold")
ax.set_xlabel("PBS-Induced Vth Shift Range [V]", fontsize=16.56, fontweight="bold", labelpad=8)
ax.set_ylabel("MRR@10 (%)", fontsize=16.56, fontweight="bold", labelpad=4)
ax.set_xlim(-0.15, 3.15)
ax.set_ylim(0, 92)
ax.tick_params(axis='y', labelsize=16.56)
ax.grid(True, axis="y", ls="--", alpha=0.3, lw=0.8)
for spine in ax.spines.values():
    spine.set_visible(True)


for label in ax.get_yticklabels():
    if label.get_text() == '0':
        offset = ScaledTranslation(0, 5/72, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)
        break

fig.tight_layout(pad=0.3)
out = SCRIPT_DIR / "[graph_optC]mrr_final.png"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"Saved: {out}")

# ── 레전드 별도 파일 ──────────────────────────────────────────────────────────
handles = [
    mlines.Line2D([], [], color=COLOR_COMP,   marker='o', ms=10, lw=3.5,
                  label="Vth comp (this work)"),
    mlines.Line2D([], [], color=COLOR_NOCOMP, marker='D', ms=6.5,lw=1.5,
                  label="No comp"),
]

fig, ax = plt.subplots(figsize=(1, 1))
ax.axis("off")
legend = ax.legend(handles=handles, loc="center", fontsize=12, frameon=True,
                   framealpha=1.0, edgecolor="black", handlelength=2.5,
                   handletextpad=0.6, borderpad=0.8, labelspacing=0.5)
legend.get_frame().set_linewidth(1.2)
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
out = SCRIPT_DIR / "[graph_optC]mrr_legend.png"
fig.savefig(out, dpi=300, bbox_inches=bbox.padded(0.05))
plt.close(fig)
print(f"Saved: {out}")
