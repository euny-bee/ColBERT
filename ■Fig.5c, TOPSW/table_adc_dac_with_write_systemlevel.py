"""
Table 4-ADC/DAC + Write, system-level scale only. Redesigned so the cumulative Cell ->
+ADC -> +DAC ladder (search-only, unchanged from Table 4-ADC/DAC) is visually separated from
the write addition: write is a single flat, one-time cost (NOT part of the per-row cumulative
chain -- it does not multiply or re-accumulate across the three scenario rows), shown as its
own line between the search subtotal and a final bold "Total" row. Earlier version put
"+ Write" in its own column repeated on every row, which read as if write accumulated once
per row (e.g. 12,289.33 x 3) -- it does not; it is added exactly once.
Output: table_adc_dac_with_write_systemlevel.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_adc_dac_with_write_systemlevel.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

TITLE = "System-level scale (2,048 centroids)"

# Search-only cumulative rows (unchanged from Table 4-ADC/DAC)
search_rows = [
    ("Cell only",     "274.03 nJ", "389.71 TOPS/W", "14,414×"),
    ("+ ADC",         "899.76 nJ", "118.69 TOPS/W", "4,390×"),
    ("+ DAC (final)", "901.89 nJ", "118.41 TOPS/W", "4,379×"),
]
WRITE_LABEL = "+ Write  (one-time index-build cost, amortized over 255 queries)"
WRITE_VAL   = "+12,289.33 nJ"
total_row   = ("Total /query  (search final + write)", "13,191.22 nJ", "8.096 TOPS/W", "299.4×")

COL_X   = [0.02]
COL_NUM = [0.50, 0.72, 0.90]
FS_HEAD, FS_BODY = 11.5, 11.5
ROW_H = 0.079

n_rows = len(search_rows) + 1  # + total row
fig, ax = plt.subplots(figsize=(9.6, 0.9 + 3.4 * 0.44 + n_rows * ROW_H * 6.6), dpi=300)

AX_MARGIN = 0.015
ax.set_position([AX_MARGIN, AX_MARGIN, 1 - 2 * AX_MARGIN, 1 - 2 * AX_MARGIN])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_frame_on(False)
ax.patch.set_visible(False)

TABLE_RIGHT = 1.0

def hline(y, lw, x0=0, x1=None, ls="-"):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, ls=ls, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_col_headers(y):
    ax.text(COL_X[0], y, "Scenario", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_NUM, ["Energy /query", "Energy eff.", "vs. GPU"]):
        ax.text(cx, y, h, fontsize=FS_HEAD - 1, fontweight="bold", va="center", ha="center")

def draw_row(y, scenario, e, eff, r, bold=False, indent=0.02):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0] + indent, y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_NUM, [e, eff, r]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

top_margin = 0.028
y = 1.0 - top_margin
hline(y, 1.8)
y -= 0.065
ax.text(COL_X[0], y, TITLE, fontsize=FS_HEAD, fontweight="bold", va="center")
y -= 0.075
draw_col_headers(y)
y -= 0.035
hline(y, 1.2)
y -= ROW_H * 0.85

for row in search_rows:
    draw_row(y, *row)
    y -= ROW_H

# write: single flat one-time addend, visually separated (dashed rule, italic, no eff./ratio
# columns -- it is not itself an energy-efficiency scenario, just an addend to the total)
y -= ROW_H * 0.12
hline(y, 0.7, COL_X[0], TABLE_RIGHT, ls=(0, (4, 3)))
y -= ROW_H * 0.78
ax.text(COL_X[0] + 0.02, y, WRITE_LABEL, fontsize=FS_BODY - 1.3, style="italic", va="center", color="0.25")
ax.text(COL_NUM[0], y, WRITE_VAL, fontsize=FS_BODY, va="center", ha="center", style="italic", color="0.25")
y -= ROW_H * 0.95

hline(y + ROW_H * 0.28, 1.0)
y -= ROW_H * 0.12
draw_row(y, *total_row, bold=True, indent=0.0)
y -= ROW_H

y_bottom = y + ROW_H * 0.42
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.045,
        "Table 4-ADC/DAC + Write, system-level scale (headline scope). The Cell -> +ADC -> +DAC (final) rows are\n"
        "the cumulative search-only breakdown (TIA/driver/ML buffer excluded, Option A), unchanged from Table\n"
        "4-ADC/DAC. Write is a separate one-time index-build cost (cell + row/col write-driver, Option A), added\n"
        "exactly once -- not accumulated per row -- and amortized over an assumed index lifetime of 255 queries.\n"
        "The Total row is the DAC-final search cost plus this single amortized write addend. Because write includes\n"
        "its own driver while the search rows above exclude TIA/driver/ML buffer, this table is not fully scope-\n"
        "consistent -- see the full-periphery + write companion table for an apples-to-apples version.",
        fontsize=8.6, va="top", ha="left", transform=ax.transAxes)

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

from PIL import Image, ImageOps
im = Image.open(OUTPUT).convert("RGB")
gray = ImageOps.invert(im.convert("L"))
bbox = gray.getbbox()
if bbox:
    side_margin = bbox[0]
    x0, y0, x1, y1 = bbox
    im.crop((0, max(0, y0 - side_margin), im.width, min(im.height, y1 + side_margin))).save(OUTPUT)

print(f"Saved: {OUTPUT}")
