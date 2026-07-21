"""
Table 4-ADC/DAC -- Periphery energy scenario breakdown (Option A, Vth compensation): cell-only -> cell+ADC+DAC only
(TIA / driver / ML buffer excluded), and the resulting efficiency ratio vs. GPU (RTX 3070 Ti, batch=256, measured),
at two array scales. Companion to table_periphery_scenario.py (Table 4), isolating just the two conversion
blocks (A->D ADC, D->A query-injection DAC) needed around the analog core.
Output: table_periphery_adc_dac.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_adc_dac.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content ─────────────────────────────────────────────────────────────
# scenario, proto_energy, proto_topsw, proto_ratio, sys_energy, sys_topsw, sys_ratio
rows = [
    ("Cell only",    "5.97 nJ",  "389.70 TOPS/W", "33,602×", "274.03 nJ", "389.71 TOPS/W", "14,414×"),
    ("+ ADC",        "19.60 nJ", "118.70 TOPS/W", "10,234×", "899.76 nJ", "118.69 TOPS/W", "4,390×"),
    ("+ DAC (final)", "21.73 nJ", "107.07 TOPS/W", "9,231×", "901.89 nJ", "118.41 TOPS/W", "4,379×"),
]

COL_X       = [0.0205]
COL_P       = [0.2594, 0.4018, 0.5239]
COL_S       = [0.6541, 0.7965, 0.9186]
COL_X_NUM_C = COL_P + COL_S
TABLE_RIGHT = 1.0
ROW_H = 0.072
FS_HEAD, FS_GRP, FS_BODY = 11.5, 11.5, 11.5

n_rows = 2 + len(rows)
fig, ax = plt.subplots(figsize=(10.65, 0.75 + n_rows * 0.44), dpi=300)
# Axes fills the figure except for a small, EQUAL margin on every side (figure-fraction).
# get_tightbbox() always includes the axes' own position rect regardless of content
# (a matplotlib quirk), so bbox_inches="tight" ends up cropping to *this* rect -- by
# making the rect itself symmetric, the saved PNG's left/right (and top/bottom)
# margins come out symmetric too, independent of how wide the text/lines happen to be.
AX_MARGIN = 0.012
ax.set_position([AX_MARGIN, AX_MARGIN, 1 - 2 * AX_MARGIN, 1 - 2 * AX_MARGIN])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_frame_on(False)
ax.patch.set_visible(False)
top_margin = 0.036

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_grp = y_top - 0.032
    proto_cx = (COL_P[0] + COL_P[2]) / 2
    ax.text(proto_cx, y_grp, "Prototype scale (100 centroids)",
            fontsize=FS_GRP, fontweight="bold", va="center", ha="center")

    sys_cx = (COL_S[0] + COL_S[2]) / 2
    ax.text(sys_cx, y_grp, "System-level scale (2,048 centroids)",
            fontsize=FS_GRP, fontweight="bold", va="center", ha="center")

    y_r = y_grp - 0.042
    ax.text(COL_X[0], y_r, "Scenario (cumulative)", fontsize=FS_HEAD, fontweight="bold", va="center")
    headers = ["Energy /query", "Energy eff.", "vs. GPU"] * 2
    header_texts = []
    for cx, h in zip(COL_X_NUM_C, headers):
        t = ax.text(cx, y_r, h, fontsize=FS_HEAD-1, fontweight="bold", va="center", ha="center")
        header_texts.append(t)

    # measure the actual rendered extent of each group's 3 column headers, so the
    # underline runs exactly from the left edge of "Energy /query" to the right
    # edge of "vs. GPU" -- no manual offset guessing.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    pad = 0.003
    y_line = y_grp - 0.018
    for grp_texts in (header_texts[:3], header_texts[3:]):
        x0_disp = min(t.get_window_extent(renderer).x0 for t in grp_texts)
        x1_disp = max(t.get_window_extent(renderer).x1 for t in grp_texts)
        x0a = inv.transform((x0_disp, 0))[0]
        x1a = inv.transform((x1_disp, 0))[0]
        hline(y_line, 0.5, x0a - pad, x1a + pad)

    return y_r - 0.040

def draw_row(y, scenario, pe, pt, pr, se, st, sr, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [pe, pt, pr, se, st, sr]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

y_top0 = 1.0 - top_margin
y_after_head = draw_header(y_top0)
y = y_after_head - ROW_H * 0.6

for i, row in enumerate(rows):
    draw_row(y, *row, bold=(i == len(rows)-1))
    y -= ROW_H

y_bottom = y + ROW_H * 0.4

hline(y_top0, 1.8, x1=TABLE_RIGHT)
hline(y_after_head, 1.0, x1=TABLE_RIGHT)
hline(y_bottom, 1.8, x1=TABLE_RIGHT)

ax.text(0.0, y_bottom - 0.048,
        "Table 4-ADC/DAC: Periphery-energy scenario restricted to the two conversion blocks around the analog core\n"
        "(A->D readout ADC, D->A query-injection DAC) -- TIA, driver and ML buffer excluded -- for Option A (Vth\n"
        "compensation) search, per query, and the resulting efficiency ratio vs. GPU (RTX 3070 Ti, batch=256,\n"
        "measured). ADC: 1.5 pJ/conversion, 8-bit 45 nm (Andrulis et al., as-cited), N_rows_total x N_TOKENS events.\n"
        "DAC: 0.52 pJ/conversion, 8-bit (Hong & Lee 2007, 65 fJ/step x8, via Saberi et al. TCAS-I 2011), N_COLS x\n"
        "N_TOKENS events -- energy independent of row count, so identical (2.13 nJ) at both scales. Energy eff.\n"
        "(TOPS/W) = FLOPs/query / Energy/query, same FLOPs/query definition used for the GPU baseline, so vs.-GPU\n"
        "ratios are identical whether read from Energy/query or Energy eff. Process node not normalized: ADC\n"
        "(45 nm), DAC (as-cited process), analog cell (IGZO TFT, not a standard CMOS node), and GPU (Samsung\n"
        "8 nm) are each taken as-is from their own source. Precision not matched either: GPU measured in FP32\n"
        "(PyTorch default, no fp16/int8 cast); analog uses 8-bit ADC/DAC quantization.",
        fontsize=9.1, va="top", ha="left", transform=ax.transAxes)

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
