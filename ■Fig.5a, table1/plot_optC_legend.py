"""
[graph_optC]option1_single.png 의 레전드 별도 파일
  - 실선 + 마커: Option C 각 지표
  - 점선: Option A baseline
Output: [graph_optC]legend.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent

available = {f.name for f in fm.fontManager.ttflist}
sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  sans,
    "font.size":        13,
    "font.weight":      "bold",
    "axes.unicode_minus": False,
})

COLORS = {
    "MRR@10":      "#D32F2F",
    "nDCG@10":     "#F57C00",
    "R@50":        "#388E3C",
    "R@1k":        "#1565C0",
    "Cand.Recall": "#6A1B9A",
}

handles = [
    # 실선 + 마커: Option C 각 지표
    *[mlines.Line2D([], [], color=c, marker='o', ms=7, lw=2.2, label=m)
      for m, c in COLORS.items()],
    # 구분선 역할 빈 항목
    mlines.Line2D([], [], color='none', label=''),
    # 점선: Option A baseline
    mlines.Line2D([], [], color='gray', ls=':', lw=1.8, label='Option A (baseline)'),
]

fig, ax = plt.subplots(figsize=(1, 1))
ax.axis("off")

legend = ax.legend(
    handles=handles,
    loc="center",
    fontsize=11,
    frameon=True,
    framealpha=1.0,
    edgecolor="gray",
    handlelength=2.2,
    handletextpad=0.6,
    borderpad=0.8,
    labelspacing=0.45,
)

fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
out = SCRIPT_DIR / "[graph_optC]legend.png"
fig.savefig(out, dpi=300, bbox_inches=bbox)
plt.close(fig)
print(f"Saved: {out}")
