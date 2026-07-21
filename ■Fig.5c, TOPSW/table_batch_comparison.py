"""
Table 5 -- GPU (RTX 3070 Ti) energy/query vs batch size, and efficiency ratio vs analog (Option A, cell-only),
at two array scales (stacked layout, narrower width).
Output: table_batch_comparison.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_batch_comparison.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# batch, energy, latency, power, ratio
proto_rows = [
    ("1",   "2,404.5 μJ", "93.77 μs", "25.6 W",  "402,769×"),
    ("32",  "343.4 μJ",   "3.28 μs",  "104.9 W", "57,525×"),
    ("256", "200.6 μJ",   "0.96 μs",  "209.4 W", "33,601×"),
]
sys_rows = [
    ("1",   "11,773.5 μJ", "118.83 μs", "99.1 W",  "42,964×"),
    ("32",  "4,369.5 μJ",  "18.94 μs",  "230.7 W", "15,945×"),
    ("256", "3,949.7 μJ",  "15.50 μs",  "254.9 W", "14,413×"),
]

COL_X       = [0.03]
COL_X_NUM_C = [0.34, 0.53, 0.70, 0.87]
TABLE_RIGHT = 0.98
ROW_H = 0.068
FS_HEAD, FS_SEC, FS_BODY = 11.5, 11.2, 11.3

n_rows = 1 + (1 + len(proto_rows)) + (1 + len(sys_rows))
fig, ax = plt.subplots(figsize=(7.6, 2.2 + n_rows * 0.40))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
top_margin = 0.028

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_r = y_top - 0.036
    ax.text(COL_X[0], y_r, "Batch size", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_X_NUM_C, ["Energy\n/query", "Latency\n/query", "Avg.\npower", "vs.\nanalog"]):
        ax.text(cx, y_r, h, fontsize=FS_HEAD-1, fontweight="bold", va="center", ha="center", linespacing=1.25)
    return y_r - 0.042

def draw_row(y, batch, e, lat, p, ratio, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y, batch, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [e, lat, p, ratio]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

def draw_section(y, label):
    ax.text(TABLE_RIGHT / 2, y, label, fontsize=FS_SEC, style="italic", va="center", ha="center")

y = 1.0 - top_margin
hline(y, 1.8)
y_after_head = draw_header(y)
hline(y_after_head, 1.0)
y = y_after_head - ROW_H * 0.55

draw_section(y, "Prototype scale (100 centroids, vs. 5.97 nJ analog)")
y -= ROW_H
for row in proto_rows:
    draw_row(y, *row)
    y -= ROW_H

y -= ROW_H * 0.2
hline(y + ROW_H * 0.45, 1.0)

draw_section(y, "System-level scale (2,048 centroids, vs. 274.03 nJ analog)")
y -= ROW_H
for row in sys_rows:
    draw_row(y, *row)
    y -= ROW_H

y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.042,
        "Table 5: GPU (RTX 3070 Ti, measured) per-query energy, latency, and power vs. batch size (queries\n"
        "issued as one kernel call), and the resulting efficiency ratio vs. analog cell-only energy, at two\n"
        "array scales. Larger batches amortize per-call CUDA launch overhead, sharply reducing per-query\n"
        "energy; the ratio vs. analog shrinks correspondingly but stays 4-5 orders of magnitude in analog's favor.",
        fontsize=9.3, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
