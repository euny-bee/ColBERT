"""
R@50: Vth Compensation (This Work) vs No Compensation
Line + fill_between 으로 성능 차이 강조
Output: [graph_optC]r50_comparison.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

pbs_labels = ["[0, 0.5] V", "[0, 1.0] V", "[0, 2.0] V", "[0, 3.0] V"]
x = np.arange(4)

vth_comp = np.array([98.8, 98.8, 98.8, 98.8])  # Option A — PBS와 무관하게 일정
no_comp  = np.array([98.8, 94.1, 58.8, 23.1])  # Option C — PBS가 강할수록 하락

COLOR_COMP   = "#1565C0"  # 파란색: This Work
COLOR_NOCOMP = "#C62828"  # 빨간색: No Compensation

fig, ax = plt.subplots(figsize=(7.5, 5.4))

# 두 선 사이 음영 (손실 영역 강조)
ax.fill_between(x, vth_comp, no_comp, alpha=0.15, color=COLOR_NOCOMP)

# 선 그래프
ax.plot(x, vth_comp, marker='o', ms=9, color=COLOR_COMP,   lw=2.8, zorder=4,
        label="Vth Compensation (This Work)")
ax.plot(x, no_comp,  marker='o', ms=9, color=COLOR_NOCOMP, lw=2.8, zorder=4,
        label="No Compensation")

# 축 설정
ax.set_xticks(x)
ax.set_xticklabels(pbs_labels, fontsize=20, fontweight="bold")
ax.set_xlabel("PBS-Induced Vth Shift Range", fontsize=23, fontweight="bold")
ax.set_ylabel("R@50 (%)", fontsize=23, fontweight="bold")
ax.set_xlim(-0.4, 3.4)
ax.set_ylim(0, 112)
ax.tick_params(axis='y', labelsize=20)
ax.grid(True, axis="y", ls="--", alpha=0.3, lw=0.8)

# 4면 박스
for spine in ax.spines.values():
    spine.set_visible(True)

# 음영 중앙 텍스트 — x=2 지점 두 선 사이 중간
mid_y = (vth_comp[2] + no_comp[2]) / 2  # (98.8 + 58.8) / 2 = 78.8
ax.text(2.0, mid_y, "Benefit of Vth compensation",
        fontsize=13, fontweight="bold", color=COLOR_NOCOMP,
        ha="center", va="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))

# 범례
ax.legend(fontsize=13, loc="lower left", framealpha=0.9)

fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]r50_comparison.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
