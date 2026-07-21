"""
Table 4-ADC/DAC + Write (headline scope) -- Adds one-time write-phase energy (index build,
Option A, cell + row/col driver) amortized over N_queries=255 to the headline Cell/+ADC/+DAC
search scenario from table_periphery_adc_dac.py. table_periphery_adc_dac.py itself is left
untouched -- this is a new, separate figure.

Write total (with driver, from periphery_energy_model_v2.py):
    Prototype (100 centroids):    328,342.0803 nJ  (one-time)
    System-level (2,048 centroids): 3,133,779.9995 nJ  (one-time)
N_queries = 255 for both scales (same assumed index lifetime, per user decision -- the
prototype scale's own historical query count was only 3, a device-validation subset, not a
deployment scenario, so it is not used here).
    Write/query (amortized) = Write_total / 255:
        Prototype:    1,287.62 nJ/query
        System-level: 12,289.33 nJ/query

Search-only values (Cell/+ADC/+DAC) are unchanged from table_periphery_adc_dac.py. Energy
eff. (TOPS/W) and vs.-GPU are recomputed against the new Total/query (search + write).
NOTE: this headline search figure excludes TIA/driver/ML-buffer periphery, while the write
figure includes its own row/col driver -- see table_periphery_full_with_write.py for a
scope-consistent version where both search and write include full periphery.
Output: table_periphery_adc_dac_with_write.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_adc_dac_with_write.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content: scenario, search/query, +write(255q), total/query, energy eff., vs.GPU ──
blocks = [
    ("Prototype scale (100 centroids)  —  write 328,342.1 nJ ÷ 255 queries = 1,287.62 nJ/query", [
        ("Cell only",     "5.97 nJ",  "+1,287.62 nJ", "1,293.59 nJ", "1.798 TOPS/W", "155.1×"),
        ("+ ADC",         "19.60 nJ", "+1,287.62 nJ", "1,307.22 nJ", "1.780 TOPS/W", "153.5×"),
        ("+ DAC (final)", "21.73 nJ", "+1,287.62 nJ", "1,309.35 nJ", "1.777 TOPS/W", "153.2×"),
    ]),
    ("System-level scale (2,048 centroids)  —  write 3,133,780.0 nJ ÷ 255 queries = 12,289.33 nJ/query", [
        ("Cell only",     "274.03 nJ", "+12,289.33 nJ", "12,563.36 nJ", "8.499 TOPS/W", "314.4×"),
        ("+ ADC",         "899.76 nJ", "+12,289.33 nJ", "13,189.09 nJ", "8.097 TOPS/W", "299.5×"),
        ("+ DAC (final)", "901.89 nJ", "+12,289.33 nJ", "13,191.22 nJ", "8.096 TOPS/W", "299.4×"),
    ]),
]

COL_X   = [0.02]
COL_NUM = [0.335, 0.485, 0.635, 0.775, 0.91]
FS_HEAD, FS_GRP, FS_BODY = 11.5, 11, 11.5
ROW_H     = 0.079
BLOCK_GAP = 0.032

n_rows  = sum(len(rows) for _, rows in blocks)
n_extra = 2 + len(blocks) * 1.9
fig, ax = plt.subplots(figsize=(11.6, 0.9 + n_extra * 0.44 + n_rows * ROW_H * 6.1), dpi=300)

AX_MARGIN = 0.013
ax.set_position([AX_MARGIN, AX_MARGIN, 1 - 2 * AX_MARGIN, 1 - 2 * AX_MARGIN])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_frame_on(False)
ax.patch.set_visible(False)

TABLE_RIGHT = 1.0

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_col_headers(y):
    ax.text(COL_X[0], y, "Scenario (cumulative)", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_NUM, ["Search /query", "+ Write\n(amortized, 255q)", "Total /query", "Energy eff.", "vs. GPU"]):
        ax.text(cx, y, h, fontsize=FS_HEAD - 1.5, fontweight="bold", va="center", ha="center")

def draw_block_title(y, title):
    ax.text(COL_X[0], y, title, fontsize=FS_GRP, fontweight="bold", va="center")

def draw_row(y, scenario, s, w, t, eff, r, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0] + 0.02, y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_NUM, [s, w, t, eff, r]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

top_margin = 0.018
y = 1.0 - top_margin
hline(y, 1.8)
y -= 0.032
draw_col_headers(y)
y -= 0.026
hline(y, 1.2)
y -= ROW_H * 0.8

for bi, (title, rows) in enumerate(blocks):
    y -= 0.006
    draw_block_title(y, title)
    y -= 0.016
    hline(y, 0.6, COL_X[0], TABLE_RIGHT)
    y -= ROW_H * 0.85
    for i, row in enumerate(rows):
        draw_row(y, *row, bold=(i == len(rows) - 1))
        y -= ROW_H
    if bi < len(blocks) - 1:
        y -= BLOCK_GAP

y_bottom = y + ROW_H * 0.35
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.032,
        "Table 4-ADC/DAC + Write (headline scope): search energy/query (Cell + ADC + DAC, TIA/driver/ML buffer\n"
        "excluded, Option A) plus one-time write-phase energy (index build: cell + row/col write-driver, Option A)\n"
        "amortized over an assumed index lifetime of 255 queries, same query count for both scales. Write totals:\n"
        "328,342.1 nJ (prototype), 3,133,780.0 nJ (system-level). Search-only values unchanged from Table 4-ADC/DAC.\n"
        "Energy eff. (TOPS/W) and vs.-GPU are recomputed against Total /query (search + amortized write). Because\n"
        "write includes its own driver while the search figure here excludes TIA/driver/ML buffer, this table is not\n"
        "fully scope-consistent -- see the full-periphery + write companion table for an apples-to-apples version.",
        fontsize=8.6, va="top", ha="left", transform=ax.transAxes)

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

# Trim leftover dead space below the caption (see table_periphery_adc_dac_stacked.py note).
from PIL import Image, ImageOps
im = Image.open(OUTPUT).convert("RGB")
gray = ImageOps.invert(im.convert("L"))
bbox = gray.getbbox()
if bbox:
    side_margin = bbox[0]
    x0, y0, x1, y1 = bbox
    im.crop((0, max(0, y0 - side_margin), im.width, min(im.height, y1 + side_margin))).save(OUTPUT)

print(f"Saved: {OUTPUT}")
