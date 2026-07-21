"""
plot_table3_optC_decomposition.py — Table: Option C candidate-retrieval vs ranking-only PBS decomposition
Style: matches table1_vth_results.png (booktabs: top/mid/bottom rules, no vertical lines)
Output: table3_optC_decomposition.png (300 dpi) in this directory
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table3_optC_decomposition.png"

# ── Font: serif, matches LaTeX/ACL paper body ────────────────────────────────
available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  serif,
    "axes.unicode_minus": False,
})

# ── Table content ─────────────────────────────────────────────────────────────
# (variant, pbs_condition, MRR@10, nDCG@10, R@50, R@1k, bold)
rows = [
    ("Option C (own candidates)",          "v1  [0, 0.5] V", "4.0", "4.5", "10.2", "30.6", False),
    ("Option C (Option A candidates)",     "v1  [0, 0.5] V", "5.7", "6.8", "17.3", "62.4", True),
    ("Option C (own candidates)",          "v2  [0, 1.0] V", "0.4", "0.5", "1.6",  "6.3",  False),
    ("Option C (Option A candidates)",     "v2  [0, 1.0] V", "0.5", "0.6", "2.0",  "22.0", True),
    ("Option C (own candidates)",          "v3  [0, 2.0] V", "0.0", "0.0", "0.8",  "2.4",  False),
    ("Option C (Option A candidates)",     "v3  [0, 2.0] V", "0.06","0.13","1.6",  "10.2", True),
    ("Option C (own candidates)",          "v4  [0, 3.0] V", "0.0", "0.0", "0.0",  "2.0",  False),
    ("Option C (Option A candidates)",     "v4  [0, 3.0] V", "0.0", "0.0", "0.8",  "9.0",  True),
]

COL_X      = [0.02, 0.395, 0.615, 0.715, 0.815, 0.915]   # left edge of each column
COL_X_NUM_C= [0.665, 0.765, 0.865, 0.965]                # center x for numeric cols
ROW_H      = 0.062
FS_HEAD    = 12.5
FS_SEC     = 11.0
FS_BODY    = 11.5

n_rows = 1 + len(rows)
fig_h  = 1.6 + n_rows * 0.36
fig, ax = plt.subplots(figsize=(9.6, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

top_margin = 0.040

def hline(y_pos, lw):
    ax.plot([0, 1], [y_pos, y_pos], color="black", lw=lw, transform=ax.transAxes,
             solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_mrr = y_top - 0.028
    ax.text(COL_X[0], y_mrr, "Variant", fontsize=FS_HEAD, fontweight="bold", va="center")
    ax.text(COL_X[1], y_mrr, "PBS Condition", fontsize=FS_HEAD, fontweight="bold", va="center")
    headers = ["MRR@10", "nDCG@10", "R@50", "R@1k"]
    for cx, h in zip(COL_X_NUM_C, headers):
        ax.text(cx, y_mrr, h, fontsize=FS_HEAD, fontweight="bold", va="center", ha="center")
    return y_mrr - 0.028

def draw_row(y_pos, method, cond, mrr, ndcg, r50, r1k, bold):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y_pos, method, fontsize=FS_BODY, fontweight=fw, va="center")
    ax.text(COL_X[1], y_pos, cond,   fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [mrr, ndcg, r50, r1k]):
        ax.text(cx, y_pos, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

# Top thick rule
y = 1.0 - top_margin
hline(y, 1.8)

# Header row
y_after_head = draw_header(y)
hline(y_after_head, 1.0)

y = y_after_head - ROW_H * 0.55

# Rows, with a thin separator every 2 rows (per PBS condition group)
for i, (m, c, mrr, ndcg, r50, r1k, b) in enumerate(rows):
    draw_row(y, m, c, mrr, ndcg, r50, r1k, b)
    y -= ROW_H
    if i % 2 == 1 and i != len(rows) - 1:
        hline(y + ROW_H * 0.45, 0.4)

# Bottom thick rule
y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

# Caption
ax.text(0.0, y_bottom - 0.050,
         "Table 3: Decomposing Option C (NoComp) error into candidate-retrieval vs. ranking contributions.\n"
         "\"Option A candidates\" reuses Option A's (near-perfect, 100% recall) candidate set as input to\n"
         "Option C's Step 6 (dead-zone MaxSim) ranking, isolating the ranking-only PBS effect. As PBS\n"
         "strengthens (v1→v4), perfect candidates alone no longer rescue ranking quality — the dead-zone-\n"
         "corrupted MaxSim ranking becomes the dominant error source.",
         fontsize=10, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
