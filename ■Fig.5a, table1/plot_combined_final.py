"""
R@50 + MRR@10: Vth Compensation (This Work) vs No Compensation — 단일 figure
  실선: R@50 / 점선: MRR@10
Output:
  [graph_optC]combined_final.png
  [graph_optC]combined_legend.png
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
    "axes.labelsize":     23,
    "axes.labelweight":   "bold",
    "axes.titlesize":     16,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    20,
    "ytick.labelsize":    20,
    "axes.unicode_minus": False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

pbs_labels = ["[0, 0.5]", "[0, 1.0]", "[0, 2.0]", "[0, 3.0]"]
x = np.arange(4)

r50_comp  = np.array([98.8, 98.8, 98.8, 98.8])
r50_nocomp= np.array([98.8, 94.1, 58.8, 23.1])
mrr_comp  = np.array([78.8, 78.8, 78.8, 78.8])
mrr_nocomp= np.array([78.1, 71.3, 27.8,  2.1])

COLOR_COMP   = "#1565C0"
COLOR_NOCOMP = "#C62828"

# ── 메인 figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.9, 4.6))

# fill_between
ax.fill_between(x, r50_comp,  r50_nocomp,  alpha=0.10, color=COLOR_NOCOMP)
ax.fill_between(x, mrr_comp,  mrr_nocomp,  alpha=0.10, color=COLOR_NOCOMP)

# No Compensation (얇게)
ax.plot(x, r50_nocomp,  marker='o', ms=7,  color=COLOR_NOCOMP, lw=1.5, ls="-",  zorder=4)
ax.plot(x, mrr_nocomp,  marker='s', ms=7,  color=COLOR_NOCOMP, lw=1.5, ls="--", zorder=4)

# Vth Compensation (굵게)
ax.plot(x, r50_comp,  marker='o', ms=12, color=COLOR_COMP, lw=4.5, ls="-",  zorder=5)
ax.plot(x, mrr_comp,  marker='s', ms=10, color=COLOR_COMP, lw=4.5, ls="--", zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(pbs_labels, fontsize=20, fontweight="bold")
ax.set_xlabel("PBS-Induced Vth Shift Range [V]", fontsize=23, fontweight="bold", labelpad=8)
ax.set_ylabel("Score (%)", fontsize=23, fontweight="bold", labelpad=-6)
ax.set_xlim(-0.15, 3.15)
ax.set_ylim(0, 112)
ax.tick_params(axis='y', labelsize=20)
ax.grid(True, axis="y", ls="--", alpha=0.3, lw=0.8)
for spine in ax.spines.values():
    spine.set_visible(True)

ax.text(1.5, 105, "Robust to PBS-induced Vth variation",
        fontsize=12, fontweight="bold", color=COLOR_COMP,
        ha="center", va="bottom", style="italic")

for label in ax.get_yticklabels():
    if label.get_text() == '0':
        offset = ScaledTranslation(0, 5/72, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)
        break

fig.tight_layout(pad=0.3)
out = SCRIPT_DIR / "[graph_optC]combined_final.png"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"Saved: {out}")

# ── 레전드 별도 파일 ──────────────────────────────────────────────────────────
handles = [
    mlines.Line2D([], [], color=COLOR_COMP,   marker='o', ms=10, lw=3.5, ls="-",
                  label="Vth Compensation (This Work) — R@50"),
    mlines.Line2D([], [], color=COLOR_COMP,   marker='s', ms=9,  lw=3.5, ls="--",
                  label="Vth Compensation (This Work) — MRR@10"),
    mlines.Line2D([], [], color=COLOR_NOCOMP, marker='o', ms=7,  lw=1.5, ls="-",
                  label="No Compensation — R@50"),
    mlines.Line2D([], [], color=COLOR_NOCOMP, marker='s', ms=7,  lw=1.5, ls="--",
                  label="No Compensation — MRR@10"),
]

fig, ax = plt.subplots(figsize=(1, 1))
ax.axis("off")
legend = ax.legend(handles=handles, loc="center", fontsize=12, frameon=True,
                   framealpha=1.0, edgecolor="black", handlelength=2.5,
                   handletextpad=0.6, borderpad=0.8, labelspacing=0.5)
legend.get_frame().set_linewidth(1.5)
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
out = SCRIPT_DIR / "[graph_optC]combined_legend.png"
fig.savefig(out, dpi=300, bbox_inches=bbox.padded(0.05))
plt.close(fig)
print(f"Saved: {out}")
