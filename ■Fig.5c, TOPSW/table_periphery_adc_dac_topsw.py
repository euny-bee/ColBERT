"""
Table 4-ADC/DAC (TOPS/W variant) -- Same scenario breakdown as table_periphery_adc_dac.py
(Option A, Vth compensation: cell-only -> cell+ADC+DAC only, TIA/driver/ML buffer excluded),
but expressed as energy efficiency (TOPS/W) instead of energy/query (nJ).
TOPS/W = FLOPs/query / Energy/query, where FLOPs/query = N_rows_total x N_TOKENS x 256
(256 FLOPs/dot-product = 128-dim MAC, 2*dim convention). Same FLOPs used for GPU baseline,
so vs.-GPU ratio is unchanged from table_periphery_adc_dac.py (FLOPs cancel in the ratio).
Output: table_periphery_adc_dac_topsw.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_adc_dac_topsw.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content ─────────────────────────────────────────────────────────────
# scenario, proto_topsw, proto_ratio, sys_topsw, sys_ratio
rows = [
    ("Cell only",        "389.70 TOPS/W",  "33,602×", "389.71 TOPS/W",  "14,414×"),
    ("+ ADC",             "118.70 TOPS/W",  "10,234×", "118.69 TOPS/W",  "4,390×"),
    ("+ DAC (final)",     "107.07 TOPS/W",  "9,231×",  "118.41 TOPS/W",  "4,379×"),
]

gpu_row = ("GPU (RTX 3070 Ti, measured)", "0.0116 TOPS/W", "1×", "0.0270 TOPS/W", "1×")

COL_X       = [0.02]
COL_X_NUM_C = [0.44, 0.615, 0.79, 0.945]
TABLE_RIGHT = 0.99
ROW_H = 0.072
FS_HEAD, FS_GRP, FS_BODY = 11.5, 11.5, 11.5

n_rows = 2 + len(rows) + 1
fig, ax = plt.subplots(figsize=(10.2, 1.8 + n_rows * 0.44))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
top_margin = 0.036

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_grp = y_top - 0.032
    proto_cx = (COL_X_NUM_C[0] + COL_X_NUM_C[1]) / 2
    ax.text(proto_cx, y_grp, "Prototype scale (100 centroids)",
            fontsize=FS_GRP, fontweight="bold", va="center", ha="center")
    hline(y_grp - 0.018, 0.5, COL_X_NUM_C[0] - 0.09, COL_X_NUM_C[1] + 0.075)

    sys_cx = (COL_X_NUM_C[2] + COL_X_NUM_C[3]) / 2
    ax.text(sys_cx, y_grp, "System-level scale (2,048 centroids)",
            fontsize=FS_GRP, fontweight="bold", va="center", ha="center")
    hline(y_grp - 0.018, 0.5, COL_X_NUM_C[2] - 0.09, COL_X_NUM_C[3] + 0.075)

    y_r = y_grp - 0.042
    ax.text(COL_X[0], y_r, "Scenario (cumulative)", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_X_NUM_C, ["Energy eff. /query", "vs. GPU", "Energy eff. /query", "vs. GPU"]):
        ax.text(cx, y_r, h, fontsize=FS_HEAD-1, fontweight="bold", va="center", ha="center")
    return y_r - 0.040

def draw_row(y, scenario, pe, pr, se, sr, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [pe, pr, se, sr]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

y = 1.0 - top_margin
hline(y, 1.8)
y_after_head = draw_header(y)
hline(y_after_head, 1.0)
y = y_after_head - ROW_H * 0.6

for i, row in enumerate(rows):
    draw_row(y, *row, bold=(i == len(rows)-1))
    y -= ROW_H

hline(y + ROW_H * 0.4, 0.8)
y -= ROW_H * 0.15
draw_row(y, *gpu_row)
y -= ROW_H

y_bottom = y + ROW_H * 0.4
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.048,
        "Table 4-ADC/DAC (TOPS/W variant): Same scenario as Table 4-ADC/DAC, expressed as energy efficiency\n"
        "instead of energy/query, for Option A (Vth compensation) search. TOPS/W = FLOPs/query / Energy/query,\n"
        "with FLOPs/query = N_rows_total x N_TOKENS x 256 (128-dim dot-product = 128 mult + 128 add, 2xdim\n"
        "convention) -- 2,326,528 FLOPs/query (prototype), 106,790,912 FLOPs/query (system-level). Same FLOPs\n"
        "definition applied to the GPU (RTX 3070 Ti, batch=256, measured) baseline, so vs.-GPU ratios are\n"
        "identical to the nJ-based table (FLOPs cancel in the ratio). GPU row: same FLOPs / GPU energy/query\n"
        "(200,600.7 nJ prototype, 3,949,707.8 nJ system-level) -- shown as the 1x reference point. ADC/DAC\n"
        "unit-energy assumptions unchanged (1.5 pJ/conv. Andrulis et al., as-cited; 0.52 pJ/conv. Hong & Lee\n"
        "2007 via Saberi et al. TCAS-I 2011).",
        fontsize=9.1, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
