"""
Table 6 -- Periphery-energy scenario (cell -> +TIA -> +ADC -> +driver -> +ML buffer) x GPU batch size (1/32/256),
efficiency ratio vs. GPU (RTX 3070 Ti, measured), at two array scales (stacked layout).
Output: table_periphery_batch_combined.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_batch_combined.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# scenario, analog_energy, ratio_b1, ratio_b32, ratio_b256
proto_rows = [
    ("Cell only",           "5.97 nJ",     "402,776×", "57,526×", "33,602×"),
    ("+ TIA",               "414.9 nJ",    "5,795×",   "828×",    "483.5×"),
    ("+ ADC",               "428.6 nJ",    "5,611×",   "801×",    "468.1×"),
    ("+ Driver",            "6,361.4 nJ",  "378×",     "54×",     "31.5×"),
    ("+ ML buffer (final)", "8,269.8 nJ",  "291×",     "42×",     "24.3×"),
]
sys_rows = [
    ("Cell only",           "274.0 nJ",     "42,965×", "15,946×", "14,414×"),
    ("+ TIA",               "19,045.9 nJ",  "618×",    "229×",    "207.4×"),
    ("+ ADC",               "19,671.6 nJ",  "599×",    "222×",    "200.8×"),
    ("+ Driver",            "209,233.2 nJ", "56×",     "21×",     "18.9×"),
    ("+ ML buffer (final)", "296,835.1 nJ", "40×",     "15×",     "13.3×"),
]

COL_X       = [0.02]
COL_X_NUM_C = [0.375, 0.565, 0.735, 0.905]
TABLE_RIGHT = 0.98
ROW_H = 0.062
FS_HEAD, FS_SEC, FS_BODY = 11.3, 11.0, 11.0

n_rows = 1 + (1 + len(proto_rows)) + (1 + len(sys_rows))
fig, ax = plt.subplots(figsize=(9.6, 2.3 + n_rows * 0.38))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
top_margin = 0.026

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_grp = y_top - 0.030
    ratio_cx = (COL_X_NUM_C[1] + COL_X_NUM_C[3]) / 2
    ax.text(ratio_cx, y_grp, "vs. GPU (RTX 3070 Ti, measured), by batch size",
            fontsize=FS_HEAD-1, fontweight="bold", va="center", ha="center")
    hline(y_grp - 0.016, 0.5, COL_X_NUM_C[1] - 0.09, COL_X_NUM_C[3] + 0.05)

    y_r = y_grp - 0.038
    ax.text(COL_X[0], y_r, "Scenario (cumulative)", fontsize=FS_HEAD, fontweight="bold", va="center")
    ax.text(COL_X_NUM_C[0], y_r, "Analog\nenergy/query", fontsize=FS_HEAD-1.3, fontweight="bold", va="center", ha="center", linespacing=1.2)
    for cx, h in zip(COL_X_NUM_C[1:], ["batch=1", "batch=32", "batch=256"]):
        ax.text(cx, y_r, h, fontsize=FS_HEAD-1, fontweight="bold", va="center", ha="center")
    return y_r - 0.040

def draw_row(y, scenario, e, r1, r32, r256, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [e, r1, r32, r256]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

def draw_section(y, label):
    ax.text(TABLE_RIGHT / 2, y, label, fontsize=FS_SEC, style="italic", va="center", ha="center")

y = 1.0 - top_margin
hline(y, 1.8)
y_after_head = draw_header(y)
hline(y_after_head, 1.0)
y = y_after_head - ROW_H * 0.55

draw_section(y, "Prototype scale (100 centroids)")
y -= ROW_H
for i, row in enumerate(proto_rows):
    draw_row(y, *row, bold=(i == len(proto_rows)-1))
    y -= ROW_H

y -= ROW_H * 0.2
hline(y + ROW_H * 0.45, 1.0)

draw_section(y, "System-level scale (2,048 centroids)")
y -= ROW_H
for i, row in enumerate(sys_rows):
    draw_row(y, *row, bold=(i == len(sys_rows)-1))
    y -= ROW_H

y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.040,
        "Table 6: Cumulative periphery-energy scenario for Option A (Vth compensation) search, per query, vs. GPU\n"
        "efficiency ratio at three batch sizes. Analog energy/query is batch-independent; only the GPU denominator\n"
        "changes with batch size. TIA/ML-buffer power scaled 200x down from literature (9 mW / 42 mW at 5 ns) to\n"
        "match this work's 1 μs sensing window; driver (0.45 mW) and ADC (1.5 pJ/conversion) taken as-cited.",
        fontsize=9.2, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
