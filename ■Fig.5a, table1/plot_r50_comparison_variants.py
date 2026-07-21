"""
R@50 비교 figure — 4가지 강조 방식 각각 별도 저장
  opt1: This Work 직접 라벨 + 화살표
  opt2: 선 굵기·마커 크기 차별화
  opt3: 파란 수평 밴드 (This Work 유지 영역)
  opt4: 별(★) 마커
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

vth_comp = np.array([98.8, 98.8, 98.8, 98.8])
no_comp  = np.array([98.8, 94.1, 58.8, 23.1])

COLOR_COMP   = "#1565C0"
COLOR_NOCOMP = "#C62828"

def base_ax(ax, lw_comp=2.8, ms_comp=9, lw_nocomp=2.8, ms_nocomp=9, marker_comp='o', zorder_comp=4):
    ax.fill_between(x, vth_comp, no_comp, alpha=0.15, color=COLOR_NOCOMP)
    ax.plot(x, no_comp,  marker='o', ms=ms_nocomp, color=COLOR_NOCOMP,
            lw=lw_nocomp, zorder=4, label="No Compensation")
    ax.plot(x, vth_comp, marker=marker_comp, ms=ms_comp, color=COLOR_COMP,
            lw=lw_comp, zorder=zorder_comp, label="Vth Compensation (This Work)")
    ax.set_xticks(x)
    ax.set_xticklabels(pbs_labels, fontsize=20, fontweight="bold")
    ax.set_xlabel("PBS-Induced Vth Shift Range", fontsize=23, fontweight="bold")
    ax.set_ylabel("R@50 (%)", fontsize=23, fontweight="bold")
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(0, 112)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, axis="y", ls="--", alpha=0.3, lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
    mid_y = (vth_comp[2] + no_comp[2]) / 2
    ax.text(2.0, mid_y, "Benefit of Vth compensation",
            fontsize=13, fontweight="bold", color=COLOR_NOCOMP,
            ha="center", va="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))

# ── Option 1: 직접 라벨 + 화살표 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))
base_ax(ax)
ax.legend(fontsize=13, loc="lower left", framealpha=0.9)
ax.annotate("This Work",
            xy=(3, vth_comp[3]), xytext=(2.3, 108),
            fontsize=14, fontweight="bold", color=COLOR_COMP,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=COLOR_COMP, lw=1.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_COMP, lw=1.5, alpha=0.9))
fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]r50_opt1_label.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Option 2: 선 굵기·마커 크기 차별화 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))
base_ax(ax, lw_comp=4.5, ms_comp=12, lw_nocomp=1.5, ms_nocomp=7)
ax.legend(fontsize=13, loc="lower left", framealpha=0.9)
fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]r50_opt2_thickness.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Option 3: 파란 수평 밴드 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))
ax.axhspan(94, 101.5, alpha=0.12, color=COLOR_COMP, zorder=0)
ax.text(3.35, 97.5, "Maintained\nzone", fontsize=11, fontweight="bold",
        color=COLOR_COMP, ha="right", va="center", style="italic")
base_ax(ax)
ax.legend(fontsize=13, loc="lower left", framealpha=0.9)
fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]r50_opt3_band.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Option 4: 별(★) 마커 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))
base_ax(ax, ms_comp=18, marker_comp='*', zorder_comp=6)
ax.legend(fontsize=13, loc="lower left", framealpha=0.9)
fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]r50_opt4_star.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

print("\n완료!")
