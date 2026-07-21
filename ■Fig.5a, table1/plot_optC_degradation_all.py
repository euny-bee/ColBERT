"""
Option C 성능 저하 시각화 — 3가지 그래프 스타일
스타일 기준: [vth_v3_fixedcell]row0_scatter_newline.png 과 동일
  (sans-serif bold, FONT_BASE=13, FONT_AXIS=16, FONT_PANEL=16, tick=14)
출력:
  [graph_optC]option1_single.png
  [graph_optC]option2_twopanel.png
  [graph_optC]option3_degradation.png
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
    "axes.labelsize":    23,
    "axes.labelweight":  "bold",
    "axes.titlesize":    16,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   20,
    "ytick.labelsize":   20,
    "axes.unicode_minus": False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

GRID_ALPHA = 0.3

# ── 데이터 ─────────────────────────────────────────────────────────────────────
pbs_labels = ["[0, 0.5] V", "[0, 1.0] V", "[0, 2.0] V", "[0, 3.0] V"]
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
    "Digital":  {"MRR@10": 78.4, "nDCG@10": 83.1, "R@50": 98.8, "R@1k": 99.6},
    "Option A": {"MRR@10": 78.8, "nDCG@10": 83.4, "R@50": 98.8, "R@1k": 99.6},
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
    ax.set_xticklabels(pbs_labels, fontsize=20, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=23, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=23, fontweight="bold")
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, axis="y", ls="--", alpha=GRID_ALPHA, lw=0.8)

# ── Option 1: Single line plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))

for metric, vals in full.items():
    ax.plot(x, vals, marker='o', ms=8, color=COLORS[metric], lw=2.4, zorder=3)

for metric in ["MRR@10", "nDCG@10", "R@50", "R@1k"]:
    ax.axhline(baselines["Option A"][metric], color=COLORS[metric], ls=":", lw=1.4, alpha=0.55)

style_ax(ax, ylim=(0, 112))
ax.set_title("Option C (NoComp) — Full Pipeline under PBS Stress", pad=10)
for spine in ax.spines.values():
    spine.set_visible(True)

fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]option1_single.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Option 1 saved: {out}")

# ── Option 2: 2-panel (ranking | recall) ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

for metric in ["MRR@10", "nDCG@10"]:
    ax1.plot(x, full[metric], marker='o', ms=8, color=COLORS[metric], lw=2.4,
             label=f"Opt-C: {metric}", zorder=3)
    ax1.axhline(baselines["Option A"][metric], color=COLORS[metric], ls="--", lw=1.2, alpha=0.55)
    ax1.axhline(baselines["Digital"][metric],  color=COLORS[metric], ls=":",  lw=1.0, alpha=0.35)

dummy_A   = mlines.Line2D([], [], color='gray', ls='--', lw=1.5, label='Option A baseline')
dummy_dig = mlines.Line2D([], [], color='gray', ls=':',  lw=1.2, label='Digital baseline')
h, l = ax1.get_legend_handles_labels()
ax1.legend(h + [dummy_A, dummy_dig], l + ['Option A baseline', 'Digital baseline'],
           fontsize=11, loc="upper right", framealpha=0.88)
ax1.set_title("Ranking Quality", pad=8)
style_ax(ax1, ylim=(0, 100))

for metric in ["R@50", "R@1k", "Cand.Recall"]:
    ax2.plot(x, full[metric], marker='o', ms=8, color=COLORS[metric], lw=2.4,
             label=f"Opt-C: {metric}", zorder=3)
    ax2.axhline(baselines["Option A"].get(metric, 100.0), color=COLORS[metric],
                ls="--", lw=1.2, alpha=0.55)

ax2.legend(fontsize=11, loc="upper right", framealpha=0.88)
ax2.set_title("Recall Metrics", pad=8)
style_ax(ax2, ylim=(0, 112), ylabel="Recall (%)")

fig.suptitle("Option C (NoComp) — Full Pipeline under PBS-Induced $V_{th}$ Variation",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]option2_twopanel.png"
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
    ax.axhline(baselines["Option A"][metric], color="black", ls=":", lw=1.5, alpha=0.50,
               label=f"Option A (ref)")

    ax.legend(fontsize=11, loc="upper right", framealpha=0.88)
    ax.set_title(title, pad=8)
    style_ax(ax, ylim=(0, 112))

fig.suptitle("Option C — Rank-Only vs. Full Pipeline: IVF (Step 2) Degradation",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
out = SCRIPT_DIR / "[graph_optC]option3_degradation.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Option 3 saved: {out}")

print("\n완료!")
