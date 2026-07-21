"""
R@50: Vth Compensation (This Work) vs No Compensation — 최종 버전
  - Option 2 스타일: 선 굵기·마커 크기 차별화
  - "Benefit of Vth compensation" 텍스트 없음
  - 레전드 별도 파일로 분리
Output:
  [graph_optC]r50_final.png
  [graph_optC]r50_legend.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
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

vth_comp = np.array([98.8, 98.8, 98.8, 98.8])
no_comp  = np.array([98.8, 94.1, 58.8, 23.1])

COLOR_COMP   = "#1565C0"
COLOR_NOCOMP = "#C62828"

# ── 메인 figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.9, 4.6))

ax.fill_between(x, vth_comp, no_comp, alpha=0.15, color=COLOR_NOCOMP)
ax.plot(x, no_comp,  marker='o', ms=7,  color=COLOR_NOCOMP, lw=1.5, zorder=4)
ax.plot(x, vth_comp, marker='o', ms=12, color=COLOR_COMP,   lw=4.5, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(pbs_labels, fontsize=20, fontweight="bold")
ax.set_xlabel("PBS-Induced Vth Shift Range [V]", fontsize=23, fontweight="bold", labelpad=8)
ax.set_ylabel("R@50 (%)", fontsize=23, fontweight="bold", labelpad=-6)
ax.set_xlim(-0.15, 3.15)
ax.set_ylim(0, 112)
ax.tick_params(axis='y', labelsize=20)
ax.grid(True, axis="y", ls="--", alpha=0.3, lw=0.8)
for spine in ax.spines.values():
    spine.set_visible(True)

from matplotlib.transforms import ScaledTranslation
ax.text(1.5, 105, "Robust to PBS-induced Vth variation",
        fontsize=12, fontweight="bold", color=COLOR_COMP,
        ha="center", va="bottom", style="italic")

for label in ax.get_yticklabels():
    if label.get_text() == '0':
        offset = ScaledTranslation(0, 5/72, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)
        break

fig.tight_layout(pad=0.3)
out = SCRIPT_DIR / "[graph_optC]r50_final.png"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"Saved: {out}")

# ── 레전드 별도 파일 ──────────────────────────────────────────────────────────
handles = [
    mlines.Line2D([], [], color=COLOR_COMP,   marker='o', ms=10, lw=3.5,
                  label="Vth Compensation\n(This Work)"),
    mlines.Line2D([], [], color=COLOR_NOCOMP, marker='o', ms=7,  lw=1.5,
                  label="No Compensation"),

]

fig, ax = plt.subplots(figsize=(1, 1))
ax.axis("off")
legend = ax.legend(handles=handles, loc="center", fontsize=12, frameon=True,
                   framealpha=1.0, edgecolor="black", handlelength=2.5,
                   handletextpad=0.6, borderpad=0.8, labelspacing=0.5)
legend.get_frame().set_linewidth(1.5)
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
out = SCRIPT_DIR / "[graph_optC]r50_legend.png"
fig.savefig(out, dpi=300, bbox_inches=bbox.padded(0.05))
plt.close(fig)
print(f"Saved: {out}")
