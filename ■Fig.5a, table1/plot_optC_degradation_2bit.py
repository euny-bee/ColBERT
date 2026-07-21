"""
Option C 성능 저하 시각화 — Digital 2bit baseline 포함 버전
기존 [graph_optC]option1/2/3 과 별도로 저장
출력:
  [graph_optC]option1_2bit.png
  [graph_optC]option2_2bit.png
  [graph_optC]option3_2bit.png
  [graph_optC]legend_2bit.png
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
    "font.family":       "sans-serif",
    "font.sans-serif":   sans,
    "font.size":         13,
    "font.weight":       "bold",
    "axes.labelsize":    16,
    "axes.labelweight":  "bold",
    "axes.titlesize":    16,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "axes.unicode_minus": False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

GRID_ALPHA = 0.3

# ── 데이터 ─────────────────────────────────────────────────────────────────────
pbs_labels = ["v1\n[0, 0.5] V", "v2\n[0, 1.0] V", "v3\n[0, 2.0] V", "v4\n[0, 3.0] V"]
x = np.arange(4)

full = {
    "MRR@10":      [78.1, 71.3, 27.8,  2.1],
    "nDCG@10":     [82.3, 75.9, 32.0,  3.4],
    "R@50":        [98.8, 94.1, 58.8, 23.1],
    "R@1k":        [99.6, 96.5, 67.1, 48.2],
    "Cand.Recall": [100.0, 96.5, 68.2, 51.0],
}
rankonly = {
    "MRR@10":  [77.4, 74.6, 35.0,  5.1],
    "nDCG@10": [81.8, 79.3, 41.9,  7.5],
    "R@50":    [99.2, 98.0, 86.3, 42.7],
    "R@1k":    [99.6, 99.6, 98.4, 89.8],
}
baselines = {
    "Option A":   {"MRR@10": 78.8, "nDCG@10": 83.4, "R@50": 98.8, "R@1k": 99.6, "Cand.Recall": 100.0},
    "Digital 2bit": {"MRR@10": 76.9, "nDCG@10": 81.9, "R@50": 98.8, "R@1k": 100.0, "Cand.Recall": 100.0},
}

COLORS = {
    "MRR@10":      "#D32F2F",
    "nDCG@10":     "#F57C00",
    "R@50":        "#388E3C",
    "R@1k":        "#1565C0",
    "Cand.Recall": "#6A1B9A",
}

def style_ax(ax, ylim=(0, 105), xlabel="PBS-Induced Vth Shift (No Compensation)", ylabel="Score (%)"):
    ax.set_xticks(x)
    ax.set_xticklabels(pbs_labels, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, axis="y", ls="--", alpha=GRID_ALPHA, lw=0.8)

def draw_baselines(ax, metrics):
    for metric in metrics:
        c = COLORS[metric]
        ax.axhline(baselines["Option A"][metric],    color=c, ls=":",  lw=1.4, alpha=0.60)
        ax.axhline(baselines["Digital 2bit"][metric], color=c, ls="--", lw=1.4, alpha=0.60)

# ── Option 1: Single line plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))

for metric, vals in full.items():
    ax.plot(x, vals, marker='o', ms=8, color=COLORS[metric], lw=2.4, zorder=3)

draw_baselines(ax, ["MRR@10", "nDCG@10", "R@50", "R@1k"])

style_ax(ax, ylim=(0, 112))
ax.set_title("Option C (NoComp) — Full Pipeline under PBS Stress", pad=10)

fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]option1_2bit.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Option 1 saved: {out}")

# ── Option 2: 2-panel ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

for metric in ["MRR@10", "nDCG@10"]:
    ax1.plot(x, full[metric], marker='o', ms=8, color=COLORS[metric], lw=2.4,
             label=f"Opt-C: {metric}", zorder=3)
draw_baselines(ax1, ["MRR@10", "nDCG@10"])

dummy_A   = mlines.Line2D([], [], color='gray', ls=':',  lw=1.5, label='Option A baseline')
dummy_2bt = mlines.Line2D([], [], color='gray', ls='--', lw=1.5, label='Digital 2bit baseline')
h, l = ax1.get_legend_handles_labels()
ax1.legend(h + [dummy_A, dummy_2bt], l + ['Option A baseline', 'Digital 2bit baseline'],
           fontsize=11, loc="upper right", framealpha=0.88)
ax1.set_title("Ranking Quality", pad=8)
style_ax(ax1, ylim=(0, 100))

for metric in ["R@50", "R@1k", "Cand.Recall"]:
    ax2.plot(x, full[metric], marker='o', ms=8, color=COLORS[metric], lw=2.4,
             label=f"Opt-C: {metric}", zorder=3)
draw_baselines(ax2, ["R@50", "R@1k"])

ax2.legend(fontsize=11, loc="upper right", framealpha=0.88)
ax2.set_title("Recall Metrics", pad=8)
style_ax(ax2, ylim=(0, 112), ylabel="Recall (%)")

fig.suptitle("Option C (NoComp) — Full Pipeline under PBS-Induced Vth Variation",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]option2_2bit.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Option 2 saved: {out}")

# ── Option 3: rank-only vs full pipeline ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

for (metric, ax, title) in [("MRR@10", axes[0], "MRR@10"), ("R@50", axes[1], "R@50")]:
    f_vals  = np.array(full[metric])
    ro_vals = np.array(rankonly[metric])

    ax.plot(x, f_vals,  marker='o', ms=8, color="#C62828", lw=2.4,
            label="Full Pipeline (Step 2+6)", zorder=4)
    ax.plot(x, ro_vals, marker='s', ms=8, color="#1565C0", lw=2.4, ls="--",
            label="Rank-Only (Step 6)", zorder=4)
    ax.fill_between(x, f_vals, ro_vals, alpha=0.12, color="#C62828",
                    label="IVF (Step 2) impact")
    ax.axhline(baselines["Option A"][metric],    color="black", ls=":",  lw=1.5, alpha=0.55,
               label="Option A (ref)")
    ax.axhline(baselines["Digital 2bit"][metric], color="black", ls="--", lw=1.5, alpha=0.45,
               label="Digital 2bit (ref)")

    ax.legend(fontsize=11, loc="upper right", framealpha=0.88)
    ax.set_title(title, pad=8)
    style_ax(ax, ylim=(0, 112))

fig.suptitle("Option C — Rank-Only vs. Full Pipeline: IVF (Step 2) Degradation",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]option3_2bit.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Option 3 saved: {out}")

# ── 레전드 ─────────────────────────────────────────────────────────────────────
handles = [
    *[mlines.Line2D([], [], color=c, marker='o', ms=7, lw=2.2, label=m)
      for m, c in COLORS.items()],
    mlines.Line2D([], [], color='none', label=''),
    mlines.Line2D([], [], color='gray', ls=':',  lw=1.8, label='Option A (baseline)'),
    mlines.Line2D([], [], color='gray', ls='--', lw=1.8, label='Digital 2bit (baseline)'),
]

fig, ax = plt.subplots(figsize=(1, 1))
ax.axis("off")
legend = ax.legend(handles=handles, loc="center", fontsize=11, frameon=True,
                   framealpha=1.0, edgecolor="gray", handlelength=2.2,
                   handletextpad=0.6, borderpad=0.8, labelspacing=0.45)
fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
out = SCRIPT_DIR / "[graph_optC]legend_2bit.png"
fig.savefig(out, dpi=300, bbox_inches=bbox)
plt.close(fig)
print(f"Legend saved: {out}")

print("\n완료!")
