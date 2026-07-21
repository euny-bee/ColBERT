"""
plot_table1_vth_results.py — Table: Retrieval quality under PBS-induced Vth variation
Style: matches ColBERTv2 paper "Table 4" (booktabs: top/mid/bottom rules, no vertical lines)
Output: table1_vth_results.png (300 dpi) in this directory
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table1_vth_results.png"

# ── Font: serif, matches LaTeX/ACL paper body ────────────────────────────────
available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  serif,
    "axes.unicode_minus": False,
})

# ── Table content ─────────────────────────────────────────────────────────────
# (method, vth_condition, MRR@10, nDCG@10, R@50, R@1k, bold)
rows_baseline = [
    ("Digital", "--", "78.4", "83.1", "98.8", "99.6", False),
]
rows_pbs = [
    ("Option A (VthComp)", "v1–v4 (PBS-robust)", "78.8", "83.4", "98.8", "99.6", True),
    ("Option C (NoComp)",  "v1  [0, 0.5] V",          "4.0",  "4.5",  "10.2", "30.6", False),
    ("Option C (NoComp)",  "v2  [0, 1.0] V",          "0.4",  "0.5",  "1.6",  "6.3",  False),
    ("Option C (NoComp)",  "v3  [0, 2.0] V",          "0.0",  "0.0",  "0.8",  "2.4",  False),
    ("Option C (NoComp)",  "v4  [0, 3.0] V",          "0.0",  "0.0",  "0.0",  "2.0",  False),
]

COL_X      = [0.02, 0.345, 0.615, 0.715, 0.815, 0.915]   # left edge of each column
COL_X_NUM_C= [0.665, 0.765, 0.865, 0.965]                # center x for numeric cols
ROW_H      = 0.072
FS_HEAD    = 12.5
FS_SEC     = 11.5
FS_BODY    = 12

n_rows = 1 + len(rows_baseline) + 1 + len(rows_pbs)   # header + baseline + section + pbs rows
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
    ax.text(COL_X[1], y_mrr, "PBS Condition", fontsize=FS_HEAD, fontweight="bold", va="center")
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

# Section: baseline
draw_section(y, "No PBS Stress (Baseline)")
y -= ROW_H
for (m, c, mrr, ndcg, r50, r1k, b) in rows_baseline:
    draw_row(y, m, c, mrr, ndcg, r50, r1k, b)
    y -= ROW_H

y -= ROW_H * 0.15
hline(y + ROW_H * 0.45, 0.6)

# Section: PBS-induced Vth variation
draw_section(y, "Under PBS-Induced V$_{th}$ Variation")
y -= ROW_H
for (m, c, mrr, ndcg, r50, r1k, b) in rows_pbs:
    draw_row(y, m, c, mrr, ndcg, r50, r1k, b)
    y -= ROW_H

# Bottom thick rule
y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

# Caption
ax.text(0.0, y_bottom - 0.055,
         "Table 1: Retrieval quality (%) on MS MARCO (255 queries, 50k-passage pool) under PBS-induced\n"
         r"$V_{th}$ variation. Option A (VthComp) cancels $V_{th}$ regardless of stress level; Option C (NoComp)"
         "\ndegrades monotonically as the dead zone widens with PBS shift.",
         fontsize=10, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
