"""
plot_table23_combined.py — Combined figure: Table 2 (Digital 2bit residual) + Table 3 (Option C decomposition)
Style: matches table1_vth_results.png (booktabs: top/mid/bottom rules, no vertical lines)
Output: table23_combined.png (300 dpi) in this directory
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table23_combined.png"

# ── Font: serif, matches LaTeX/ACL paper body ────────────────────────────────
available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  serif,
    "axes.unicode_minus": False,
})

COL_X      = [0.02, 0.395, 0.615, 0.715, 0.815, 0.915]
COL_X_NUM_C= [0.665, 0.765, 0.865, 0.965]
FS_HEAD    = 12.5
FS_SEC     = 11.0
FS_BODY    = 11.5
FS_CAP     = 10

def hline(ax, y_pos, lw):
    ax.plot([0, 1], [y_pos, y_pos], color="black", lw=lw, transform=ax.transAxes,
             solid_capstyle="butt", clip_on=False)

def draw_header(ax, y_top, col1_label, row_h):
    y_mrr = y_top - row_h * 0.42
    ax.text(COL_X[0], y_mrr, col1_label, fontsize=FS_HEAD, fontweight="bold", va="center")
    ax.text(COL_X[1], y_mrr, "Condition", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_X_NUM_C, ["MRR@10", "nDCG@10", "R@50", "R@1k"]):
        ax.text(cx, y_mrr, h, fontsize=FS_HEAD, fontweight="bold", va="center", ha="center")
    return y_mrr - row_h * 0.42

def draw_row(ax, y_pos, method, cond, mrr, ndcg, r50, r1k, bold):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y_pos, method, fontsize=FS_BODY, fontweight=fw, va="center")
    ax.text(COL_X[1], y_pos, cond,   fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [mrr, ndcg, r50, r1k]):
        ax.text(cx, y_pos, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

def draw_section(ax, y_pos, label):
    ax.text(0.5, y_pos, label, fontsize=FS_SEC, style="italic", va="center", ha="center")

# ── Content ──────────────────────────────────────────────────────────────────
t2_baseline = [
    ("Digital (float32)",       "--", "78.4", "83.1", "98.8", "99.6", False),
    ("Digital (2bit residual)", "--", "76.9", "81.9", "98.8", "100.0", False),
]
t2_analog = [
    ("Option A (VthComp)", "v1–v4 (PBS-robust)", "78.8", "83.4", "98.8", "99.6", True),
]

t3_rows = [
    ("Option C (own candidates)",      "v1  [0, 0.5] V", "4.0", "4.5", "10.2", "30.6", False),
    ("Option C (Option A candidates)", "v1  [0, 0.5] V", "5.7", "6.8", "17.3", "62.4", True),
    ("Option C (own candidates)",      "v2  [0, 1.0] V", "0.4", "0.5", "1.6",  "6.3",  False),
    ("Option C (Option A candidates)", "v2  [0, 1.0] V", "0.5", "0.6", "2.0",  "22.0", True),
    ("Option C (own candidates)",      "v3  [0, 2.0] V", "0.0", "0.0", "0.8",  "2.4",  False),
    ("Option C (Option A candidates)", "v3  [0, 2.0] V", "0.06","0.13","1.6", "10.2", True),
    ("Option C (own candidates)",      "v4  [0, 3.0] V", "0.0", "0.0", "0.0",  "2.0",  False),
    ("Option C (Option A candidates)", "v4  [0, 3.0] V", "0.0", "0.0", "0.8",  "9.0",  True),
]

ROW_H = 0.034
GAP   = 0.045   # vertical gap between the two tables

# Estimate total normalized-row count to size the figure
n2 = 1 + 1 + len(t2_baseline) + 1 + len(t2_analog)   # header + sec + rows + sec + rows
n3 = 1 + len(t3_rows)
total_rows = n2 + n3
fig_h = 2.6 + total_rows * 0.30
fig, ax = plt.subplots(figsize=(9.6, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

y = 1.0 - 0.018

# ============================== Table 2 ======================================
hline(ax, y, 1.8)
y = draw_header(ax, y, "Method", ROW_H)
hline(ax, y, 1.0)
y -= ROW_H * 0.55

draw_section(ax, y, "No PBS Stress — 2bit Residual Compression (PLAID-style)")
y -= ROW_H
for (m, c, mrr, ndcg, r50, r1k, b) in t2_baseline:
    draw_row(ax, y, m, c, mrr, ndcg, r50, r1k, b)
    y -= ROW_H

y -= ROW_H * 0.15
hline(ax, y + ROW_H * 0.45, 0.6)

draw_section(ax, y, "Under PBS-Induced V$_{th}$ Variation (for reference)")
y -= ROW_H
for (m, c, mrr, ndcg, r50, r1k, b) in t2_analog:
    draw_row(ax, y, m, c, mrr, ndcg, r50, r1k, b)
    y -= ROW_H

y_t2_bottom = y + ROW_H * 0.45
hline(ax, y_t2_bottom, 1.8)

ax.text(0.0, y_t2_bottom - 0.028,
         "Table 2: Effect of 2bit residual compression (PLAID-style, mean relative L2 reconstruction error\n"
         r"27.9%) on Digital MaxSim ranking, vs. Option A's analog $V_{th}$-compensated cell.",
         fontsize=FS_CAP, va="top", ha="left", transform=ax.transAxes)

y = y_t2_bottom - GAP - 0.085

# ============================== Table 3 ======================================
hline(ax, y, 1.8)
y = draw_header(ax, y, "Variant", ROW_H)
hline(ax, y, 1.0)
y -= ROW_H * 0.55

for i, (m, c, mrr, ndcg, r50, r1k, b) in enumerate(t3_rows):
    draw_row(ax, y, m, c, mrr, ndcg, r50, r1k, b)
    y -= ROW_H
    if i % 2 == 1 and i != len(t3_rows) - 1:
        hline(ax, y + ROW_H * 0.45, 0.4)

y_t3_bottom = y + ROW_H * 0.45
hline(ax, y_t3_bottom, 1.8)

ax.text(0.0, y_t3_bottom - 0.028,
         "Table 3: Decomposing Option C (NoComp) error into candidate-retrieval vs. ranking contributions.\n"
         "\"Option A candidates\" reuses Option A's (100% recall) candidate set as input to Option C's\n"
         "Step 6 (dead-zone MaxSim) ranking, isolating the ranking-only PBS effect. As PBS strengthens\n"
         "(v1→v4), perfect candidates alone no longer rescue ranking quality.",
         fontsize=FS_CAP, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
