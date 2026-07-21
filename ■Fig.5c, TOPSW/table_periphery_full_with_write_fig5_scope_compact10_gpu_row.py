"""
Table 4-Full-Periphery + Write (scope-consistent, system-level only) -- Same content
as the system-level block of the original table_periphery_full_with_write.py, but the
prototype-scale comparison block has been dropped (this variant is presented standalone,
not as a scale comparison) and the former per-block subheading is promoted to a table
title above the column headers.

Write total (with driver, from periphery_energy_model_v2.py), amortized over N_queries=255:
    System-level: 3,133,779.9995 nJ / 255 = 12,289.33 nJ/query

Full-periphery search-only cumulative values (from periphery_energy_model_v2.py):
    System-level: cell 274.0271, +TIA 19,045.8671, +ADC 19,671.5951, +driver 209,233.1951,
                  +MLbuf(final) 296,835.1151 nJ
Output: table_periphery_full_with_write_fig5_scope_compact10_gpu_row.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_full_with_write_fig5_scope_compact10_gpu_row.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

TITLE = "System-level scale used in Table 1 and Fig. 5a (2,048 centroids + 255 queries + 10,988 documents)"

# ── Table content: scenario, search/query, +write(255q), total/query, energy eff., vs.GPU ──
rows = [
    ("Cell only",            "274.03 nJ",     "+12,289.33 nJ", "12,563.36 nJ",  "8.499 TOPS/W",  "314.4×"),
    ("+ TIA",                "+18,771.84 nJ",  "shared",        "31,335.20 nJ",  "3.408 TOPS/W",  "126.1×"),
    ("+ ADC",                "+625.73 nJ",     "shared",        "31,960.93 nJ",  "3.341 TOPS/W",  "123.6×"),
    ("+ DAC",                "+2.13 nJ",       "shared",        "31,963.06 nJ",  "3.341 TOPS/W",  "123.6×"),
    ("+ Driver",             "+189,561.60 nJ", "shared",        "221,524.66 nJ", "0.4821 TOPS/W", "17.8×"),
    ("+ ML buffer (final)",  "+87,601.92 nJ",  "shared",        "309,126.58 nJ", "0.3455 TOPS/W", "12.8×"),
    ("GPU baseline (RTX 3070 Ti)", "—", "—", "3,949,707.8 nJ", "0.0270 TOPS/W", "1×"),
]

COL_X   = [0.02]
COL_NUM = [0.266, 0.401, 0.536, 0.657, 0.770]
FS_TITLE, FS_HEAD, FS_BODY = 13, 11.5, 11.5
ROW_H     = 0.072

n_rows  = len(rows)
n_extra = 3.4  # title + header + rule allowance
fig, ax = plt.subplots(figsize=(11.8, 0.9 + n_extra * 0.44 + n_rows * ROW_H * 6.1), dpi=300)

AX_MARGIN = 0.013
ax.set_position([AX_MARGIN, AX_MARGIN, 1 - 2 * AX_MARGIN, 1 - 2 * AX_MARGIN])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_frame_on(False)
ax.patch.set_visible(False)

TABLE_RIGHT = 0.803

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_col_headers(y):
    ax.text(COL_X[0], y, "Scenario (cumulative)", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_NUM, ["Search /query", "Write /query", "Cumulative total\nenergy /query", "Energy eff.", "vs. GPU"]):
        ax.text(cx, y, h, fontsize=FS_HEAD - 1.5, fontweight="bold", va="center", ha="center")

def draw_row(y, scenario, s, w, t, eff, r, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0] + 0.02, y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_NUM, [s, w, t, eff, r]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

top_margin = 0.016
y = 1.0 - top_margin
ax.text(TABLE_RIGHT / 2, y, TITLE, fontsize=FS_TITLE, fontweight="bold", va="center", ha="center")
y -= 0.038
hline(y, 1.8)
y -= 0.052
draw_col_headers(y)
y -= 0.041
hline(y, 1.2)
y -= ROW_H * 0.8

for i, row in enumerate(rows):
    if i == len(rows) - 1:
        hline(y + ROW_H * 0.5, 0.8)
    draw_row(y, *row, bold=(i == len(rows) - 2))
    y -= ROW_H

y_bottom = y + ROW_H * 0.35
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.028,
        "Table 4-Full-Periphery + Write (scope-consistent): incremental full-periphery search energy/query\n"
        "(cell -> +TIA -> +ADC -> +DAC -> +driver -> +ML buffer, Option A) plus shared one-time write-phase energy (index build:\n"
        "cell + row/col write-driver, Option A) amortized over an assumed index lifetime of 255 queries.\n"
        "Write total: 3,133,780.0 nJ (2,048 centroids, 10,988 documents). Search entries are component increments; DAC uses 0.52 pJ/conversion.\n"
        "Other search components are from periphery_energy_model_v2.py; total energy is cumulative. Search and write include their own row/col driver, so vs.-GPU\n"
        "is scope-consistent. GPU baseline: RTX 3070 Ti, batch=256, measured by nvidia-smi power polling.",
        fontsize=8.6, va="top", ha="left", transform=ax.transAxes)

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

# Trim any leftover canvas below the caption to the actual content bbox (see prior
# table scripts' AX_MARGIN note: bbox_inches="tight" always keeps the full axes rect).
from PIL import Image, ImageOps
im = Image.open(OUTPUT).convert("RGB")
gray = ImageOps.invert(im.convert("L"))
bbox = gray.getbbox()
if bbox:
    side_margin = bbox[0]
    x0, y0, x1, y1 = bbox
    im.crop((0, max(0, y0 - side_margin), min(im.width, x1 + side_margin),
             min(im.height, y1 + side_margin))).save(OUTPUT)

print(f"Saved: {OUTPUT}")
