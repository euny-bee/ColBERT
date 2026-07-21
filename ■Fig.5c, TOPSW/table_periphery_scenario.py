"""
Table 4 -- Periphery energy scenario breakdown (Option A, Vth compensation): cell-only -> full system,
and the resulting efficiency ratio vs. GPU (RTX 3070 Ti, batch=256, measured), at two array scales.
Output: table_periphery_scenario.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_scenario.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content ─────────────────────────────────────────────────────────────
# scenario, proto_energy, proto_ratio, sys_energy, sys_ratio
rows = [
    ("Cell only",            "5.97 nJ",    "33,602×", "274.03 nJ",   "14,414×"),
    ("+ TIA",                 "414.9 nJ",   "484×",    "19,045.9 nJ", "207×"),
    ("+ ADC",                 "428.6 nJ",   "468×",    "19,671.6 nJ", "201×"),
    ("+ Driver",               "6,361.4 nJ", "31.5×",  "209,233.2 nJ","18.9×"),
    ("+ ML buffer (final)",   "8,269.8 nJ", "24.3×",   "296,835.1 nJ","13.3×"),
]

COL_X       = [0.02]
COL_X_NUM_C = [0.42, 0.575, 0.755, 0.91]
TABLE_RIGHT = 0.98
ROW_H = 0.072
FS_HEAD, FS_GRP, FS_BODY = 11.5, 11.5, 11.5

n_rows = 2 + len(rows)
fig, ax = plt.subplots(figsize=(9.6, 1.8 + n_rows * 0.44))
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
    hline(y_grp - 0.018, 0.5, COL_X_NUM_C[0] - 0.075, COL_X_NUM_C[1] + 0.075)

    sys_cx = (COL_X_NUM_C[2] + COL_X_NUM_C[3]) / 2
    ax.text(sys_cx, y_grp, "System-level scale (2,048 centroids)",
            fontsize=FS_GRP, fontweight="bold", va="center", ha="center")
    hline(y_grp - 0.018, 0.5, COL_X_NUM_C[2] - 0.075, COL_X_NUM_C[3] + 0.075)

    y_r = y_grp - 0.042
    ax.text(COL_X[0], y_r, "Scenario (cumulative)", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_X_NUM_C, ["Energy /query", "vs. GPU", "Energy /query", "vs. GPU"]):
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

y_bottom = y + ROW_H * 0.4
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.048,
        "Table 4: Cumulative periphery-energy scenario for Option A (Vth compensation) search, per query, and the\n"
        "resulting efficiency ratio vs. GPU (RTX 3070 Ti, batch=256, measured). TIA/ML-buffer power scaled down 200x\n"
        "from literature (9 mW / 42 mW at 5 ns) to match this work's 1 μs sensing window; driver (0.45 mW) and ADC\n"
        "(1.5 pJ/conversion, 8-bit 45 nm) power taken as-cited. Write-phase driver energy (not shown) is a separate\n"
        "one-time, index-build cost amortized over queries.",
        fontsize=9.3, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
