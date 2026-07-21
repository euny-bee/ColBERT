"""
Table 4-ADC/DAC (stacked) -- Same content as table_periphery_adc_dac.py (Option A, Vth
compensation: cell-only -> cell+ADC+DAC only, TIA/driver/ML buffer excluded, energy/query
and energy efficiency vs. GPU), but laid out as two stacked row-blocks (Prototype scale,
then System-level scale) sharing one set of column headers, instead of two column-groups
side by side. Much narrower, for contexts where the wide two-column-group table doesn't fit.
Output: table_periphery_adc_dac_stacked.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_adc_dac_stacked.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content ─────────────────────────────────────────────────────────────
blocks = [
    ("Prototype scale (100 centroids)", [
        ("Cell only",     "5.97 nJ",  "389.70 TOPS/W", "33,602×"),
        ("+ ADC",         "19.60 nJ", "118.70 TOPS/W", "10,234×"),
        ("+ DAC (final)", "21.73 nJ", "107.07 TOPS/W", "9,231×"),
    ]),
    ("System-level scale (2,048 centroids)", [
        ("Cell only",     "274.03 nJ", "389.71 TOPS/W", "14,414×"),
        ("+ ADC",         "899.76 nJ", "118.69 TOPS/W", "4,390×"),
        ("+ DAC (final)", "901.89 nJ", "118.41 TOPS/W", "4,379×"),
    ]),
]

COL_X    = [0.02]
COL_NUM  = [0.47, 0.685, 0.87]
FS_HEAD, FS_GRP, FS_BODY = 12, 12, 12
ROW_H     = 0.079 * 0.7
BLOCK_GAP = 0.028 * 0.7

n_rows  = sum(len(rows) for _, rows in blocks)
n_extra = 2 + len(blocks) * 1.6  # header + per-block title/rule/gap allowance
fig, ax = plt.subplots(figsize=(6.6, 0.75 + n_extra * 0.44 * 0.7 + n_rows * ROW_H * 6.1), dpi=300)

# Same symmetric-margin construction as table_periphery_adc_dac.py: axes fills the
# figure except an equal margin on every side, so bbox_inches="tight" (which always
# ends up cropping to the axes' own position rect) yields equal left/right margins.
AX_MARGIN = 0.014
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
    for cx, h in zip(COL_NUM, ["Energy /query", "Energy eff.", "vs. GPU"]):
        ax.text(cx, y, h, fontsize=FS_HEAD - 1, fontweight="bold", va="center", ha="center")

def draw_block_title(y, title):
    ax.text(COL_X[0], y, title, fontsize=FS_GRP, fontweight="bold", va="center")

def draw_row(y, scenario, e, eff, r, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0] + 0.02, y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_NUM, [e, eff, r]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

top_margin = 0.022 * 0.7
y = 1.0 - top_margin
hline(y, 1.8)
y -= 0.045 * 0.7
draw_col_headers(y)
y -= 0.032 * 0.7
hline(y, 1.2)
y -= ROW_H * 0.75

for bi, (title, rows) in enumerate(blocks):
    y -= 0.006 * 0.7
    draw_block_title(y, title)
    y -= 0.018
    hline(y, 0.6, COL_X[0], TABLE_RIGHT)
    y -= ROW_H * 0.85
    for i, row in enumerate(rows):
        draw_row(y, *row, bold=(i == len(rows) - 1))
        y -= ROW_H
    if bi < len(blocks) - 1:
        y -= BLOCK_GAP

y_bottom = y + ROW_H * 0.35
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.035 * 0.7,
        "Table 4-ADC/DAC: Periphery-energy scenario restricted to the two conversion blocks around the analog core\n"
        "(A->D readout ADC, D->A query-injection DAC) -- TIA, driver and ML buffer excluded -- for Option A (Vth\n"
        "compensation) search, per query, and the resulting efficiency ratio vs. GPU (RTX 3070 Ti, batch=256,\n"
        "measured). ADC: 1.5 pJ/conversion, 8-bit 45 nm (Andrulis et al., as-cited), N_rows_total x N_TOKENS events.\n"
        "DAC: 0.52 pJ/conversion, 8-bit (Hong & Lee 2007, 65 fJ/step x8, via Saberi et al. TCAS-I 2011), N_COLS x\n"
        "N_TOKENS events -- energy independent of row count, so identical (2.13 nJ) at both scales. Energy eff.\n"
        "(TOPS/W) = FLOPs/query / Energy/query, same FLOPs/query definition used for the GPU baseline, so vs.-GPU\n"
        "ratios are identical whether read from Energy/query or Energy eff.",
        fontsize=8.6, va="top", ha="left", transform=ax.transAxes)

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

# The symmetric-margin axes rect (see AX_MARGIN note above) always survives
# bbox_inches="tight" cropping in full, so when content is shorter than the
# figsize allotted for it, dead space is left below the caption. Trim that off
# by re-cropping to the actual ink bounding box plus a margin matching the sides.
from PIL import Image, ImageOps
im = Image.open(OUTPUT).convert("RGB")
gray = ImageOps.invert(im.convert("L"))
bbox = gray.getbbox()
if bbox:
    side_margin = bbox[0]  # left margin already set symmetrically by AX_MARGIN
    x0, y0, x1, y1 = bbox
    im.crop((0, max(0, y0 - side_margin), im.width, min(im.height, y1 + side_margin))).save(OUTPUT)

print(f"Saved: {OUTPUT}")
