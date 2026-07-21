"""
Table 4v2 -- Periphery energy scenario breakdown (Option A, Vth compensation): cell-only -> full system,
with energy/query, latency/query, avg. power, and the resulting efficiency ratio vs. GPU
(RTX 3070 Ti, batch=256, measured), at two array scales. (Copy of table_periphery_scenario.py with
Latency/Power columns added -- original left untouched.)
Output: table_periphery_scenario_v2.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_scenario_v2.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content ─────────────────────────────────────────────────────────────
# scenario, energy, latency, power, ratio  (latency is 64.0 us for every row, at both scales)
proto_rows = [
    ("Cell only",           "5.97 nJ",    "64.0 μs", "93.3 μW",   "33,602×"),
    ("+ TIA",                "414.9 nJ",   "64.0 μs", "6.48 mW",   "484×"),
    ("+ ADC",                "428.6 nJ",   "64.0 μs", "6.70 mW",   "468×"),
    ("+ Driver",             "6,361.4 nJ", "64.0 μs", "99.4 mW",   "31.5×"),
    ("+ ML buffer (final)",  "8,269.8 nJ", "64.0 μs", "129.2 mW",  "24.3×"),
]
sys_rows = [
    ("Cell only",           "274.03 nJ",     "64.0 μs", "4.28 mW",    "14,414×"),
    ("+ TIA",                "19,045.9 nJ",  "64.0 μs", "297.6 mW",   "207×"),
    ("+ ADC",                "19,671.6 nJ",  "64.0 μs", "307.4 mW",   "201×"),
    ("+ Driver",             "209,233.2 nJ", "64.0 μs", "3.27 W",     "18.9×"),
    ("+ ML buffer (final)",  "296,835.1 nJ", "64.0 μs", "4.64 W",     "13.3×"),
]

COL_X       = [0.02]
COL_X_NUM_C = [0.40, 0.565, 0.72, 0.885]
TABLE_RIGHT = 0.98
ROW_H = 0.070
FS_HEAD, FS_SEC, FS_BODY = 11.3, 11.0, 11.1

n_rows = 1 + (1 + len(proto_rows)) + (1 + len(sys_rows))
fig, ax = plt.subplots(figsize=(10.6, 2.3 + n_rows * 0.40))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
top_margin = 0.026

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_r = y_top - 0.036
    ax.text(COL_X[0], y_r, "Scenario (cumulative)", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_X_NUM_C, ["Energy\n/query", "Latency\n/query", "Avg. power\n(=Energy÷Latency)", "vs.\nGPU"]):
        ax.text(cx, y_r, h, fontsize=FS_HEAD-1, fontweight="bold", va="center", ha="center", linespacing=1.25)
    return y_r - 0.042

def draw_row(y, scenario, e, lat, p, ratio, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [e, lat, p, ratio]):
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

ax.text(0.0, y_bottom - 0.042,
        "Table 4v2: Cumulative periphery-energy scenario for Option A (Vth compensation) search, per query, and\n"
        "the resulting efficiency ratio vs. GPU (RTX 3070 Ti, batch=256, measured). Latency/query (64 μs = 32\n"
        "coarse-search + 32 scoring token-steps x 1 μs sensing window) is the same across all rows; avg. power =\n"
        "energy / latency. TIA/ML-buffer power scaled down 200x from literature (9 mW / 42 mW at 5 ns) to match\n"
        "this work's 1 μs sensing window; driver (0.45 mW) and ADC (1.5 pJ/conversion, 8-bit 45 nm) power taken\n"
        "as-cited. Write-phase driver energy (not shown) is a separate one-time, index-build cost amortized over queries.",
        fontsize=9.1, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
