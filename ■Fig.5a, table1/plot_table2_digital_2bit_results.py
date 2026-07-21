"""
plot_table2_digital_2bit_results.py — Table: Digital float32 vs 2bit residual compression vs Option A
Style: matches table1_vth_results.png (booktabs: top/mid/bottom rules, no vertical lines)
Output: table2_digital_2bit_results.png (300 dpi) in this directory
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table2_digital_2bit_results.png"

# ── Font: serif, matches LaTeX/ACL paper body ────────────────────────────────
available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  serif,
    "axes.unicode_minus": False,
})

# ── Table content ─────────────────────────────────────────────────────────────
# (method, condition, MRR@10, nDCG@10, R@50, R@1k, bold)
rows_baseline = [
    ("Digital (float32)",      "--", "78.4", "83.1", "98.8", "99.6", False),
    ("Digital (2bit residual)", "--", "76.9", "81.9", "98.8", "100.0", False),
]
rows_analog = [
    ("Option A (VthComp)", "v1–v4 (PBS-robust)", "78.8", "83.4", "98.8", "99.6", True),
]

COL_X      = [0.02, 0.345, 0.615, 0.715, 0.815, 0.915]   # left edge of each column
COL_X_NUM_C= [0.665, 0.765, 0.865, 0.965]                # center x for numeric cols
ROW_H      = 0.072
FS_HEAD    = 12.5
FS_SEC     = 11.5
FS_BODY    = 12

n_rows = 1 + len(rows_baseline) + 1 + len(rows_analog)
fig_h  = 1.55 + n_rows * 0.40
fig, ax = plt.subplots(figsize=(9.2, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

y = 1.0
top_margin = 0.045

def hline(y_pos, lw):
    ax.plot([0, 1], [y_pos, y_pos], color="black", lw=lw, transform=ax.transAxes,
             solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_mrr = y_top - 0.030
    ax.text(COL_X[0], y_mrr, "Method", fontsize=FS_HEAD, fontweight="bold", va="center")
    ax.text(COL_X[1], y_mrr, "Condition", fontsize=FS_HEAD, fontweight="bold", va="center")
    headers = ["MRR@10", "nDCG@10", "R@50", "R@1k"]
    for cx, h in zip(COL_X_NUM_C, headers):
        ax.text(cx, y_mrr, h, fontsize=FS_HEAD, fontweight="bold", va="center", ha="center")
    return y_mrr - 0.030

def draw_row(y_pos, method, cond, mrr, ndcg, r50, r1k, bold):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y_pos, method, fontsize=FS_BODY, fontweight=fw, va="center")
    ax.text(COL_X[1], y_pos, cond,   fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [mrr, ndcg, r50, r1k]):
        ax.text(cx, y_pos, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

def draw_section(y_pos, label):
    ax.text(0.5, y_pos, label, fontsize=FS_SEC, style="italic", va="center", ha="center")

# Top thick rule
y = 1.0 - top_margin
hline(y, 1.8)

# Header row
y_after_head = draw_header(y)
hline(y_after_head, 1.0)

y = y_after_head - ROW_H * 0.55

# Section: 2bit residual compression (no PBS)
draw_section(y, "No PBS Stress — 2bit Residual Compression (PLAID-style)")
y -= ROW_H
for (m, c, mrr, ndcg, r50, r1k, b) in rows_baseline:
    draw_row(y, m, c, mrr, ndcg, r50, r1k, b)
    y -= ROW_H

y -= ROW_H * 0.15
hline(y + ROW_H * 0.45, 0.6)

# Section: analog Vth compensation, for reference
draw_section(y, "Under PBS-Induced V$_{th}$ Variation (for reference)")
y -= ROW_H
for (m, c, mrr, ndcg, r50, r1k, b) in rows_analog:
    draw_row(y, m, c, mrr, ndcg, r50, r1k, b)
    y -= ROW_H

# Bottom thick rule
y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

# Caption
ax.text(0.0, y_bottom - 0.055,
         "Table 2: Effect of 2bit residual compression (PLAID-style, mean relative L2 reconstruction\n"
         r"error 27.9%) on Digital MaxSim ranking, vs. Option A's analog $V_{th}$-compensated cell. Despite"
         "\nsubstantial quantization error, the accuracy drop is small and remains far better than Option C\n"
         "under any PBS condition (see Table 1).",
         fontsize=10, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
