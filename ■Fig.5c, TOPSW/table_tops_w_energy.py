"""
Table 3 -- Per-query energy/latency/power: Analog (Option A/C) vs GPU baselines
Two scales: (i) prototype (100 centroids, seed19 subset) and (ii) system-level
(2,048 centroids, matches the Table 2 retrieval-quality experiment).
Output: table_tops_w_energy.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_tops_w_energy.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content ─────────────────────────────────────────────────────────────
# method, write_energy, search_energy, latency, power, vs_optA  (units in the string itself)

# (i) Prototype scale: 100 centroids, candidates A=184 / C=193
# GPU rows measured with batch=256 (queries batched together, matching standard serving practice)
proto_analog = [
    ("Vth compensation", "11.82 nJ",   "5.97 nJ", "64.0 μs", "93.3 μW", "1×"),
    ("No compensation",  "0.00009 nJ", "5.70 nJ", "64.0 μs", "89.1 μW", "1×"),
]
proto_gpu = [
    ("RTX 3070 Ti (batch=256), Vth comp", "--", "200.6 μJ", "0.958 μs", "209.4 W", "33,601×"),
    ("RTX 3070 Ti (batch=256), No comp",  "--", "224.9 μJ", "0.997 μs", "225.5 W", "39,451×"),
]
proto_gpu_est = [
    ("A100 (est.), Vth comp", "--", "82.5 μJ", "0.286 μs", "288.8 W", "13,820×"),
    ("A100 (est.), No comp",  "--", "92.5 μJ", "0.297 μs", "311.0 W", "16,222×"),
    ("H100 (est.), Vth comp", "--", "87.9 μJ", "0.174 μs", "505.4 W", "14,721×"),
    ("H100 (est.), No comp",  "--", "98.5 μJ", "0.181 μs", "544.3 W", "17,279×"),
]

# (ii) System-level scale: 2,048 centroids (matches Table 2), candidates A=10,988 / C(v4)=10,292
sys_analog = [
    ("Vth compensation", "242.1 nJ",  "274.0 nJ", "64.0 μs", "4,282 μW", "1×"),
    ("No compensation",  "0.0019 nJ", "240.0 nJ", "64.0 μs", "3,750 μW", "1×"),
]
sys_gpu = [
    ("RTX 3070 Ti (batch=256), Vth comp", "--", "3,949.7 μJ", "15.50 μs", "254.8 W", "14,413×"),
    ("RTX 3070 Ti (batch=256), No comp",  "--", "4,283.0 μJ", "14.87 μs", "287.9 W", "17,846×"),
]
sys_gpu_est = [
    ("A100 (est.), Vth comp", "--", "1,624.4 μJ", "4.622 μs", "351.4 W", "5,928×"),
    ("A100 (est.), No comp",  "--", "1,760.8 μJ", "4.434 μs", "397.1 W", "7,337×"),
    ("H100 (est.), Vth comp", "--", "1,730.2 μJ", "2.813 μs", "615.0 W", "6,314×"),
    ("H100 (est.), No comp",  "--", "1,875.5 μJ", "2.699 μs", "694.9 W", "7,814×"),
]

COL_X       = [0.02, 0.29]
COL_X_NUM_C = [0.475, 0.625, 0.755, 0.885]
TABLE_RIGHT = 0.98
ROW_H = 0.058
FS_HEAD, FS_SEC, FS_BODY = 11.5, 11.0, 11.2

n_rows = (1 + len(proto_analog) + 1 + len(proto_gpu) + 1 + len(proto_gpu_est) + 1
          + 1 + len(sys_analog) + 1 + len(sys_gpu) + 1 + len(sys_gpu_est))
fig, ax = plt.subplots(figsize=(10.4, 2.6 + n_rows * 0.38))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
top_margin = 0.028

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_r = y_top - 0.036
    ax.text(COL_X[0], y_r, "Method", fontsize=FS_HEAD, fontweight="bold", va="center")
    ax.text(COL_X[1], y_r, "Write energy\n(one-time)", fontsize=FS_HEAD-1.5, fontweight="bold", va="center", ha="center", linespacing=1.3)
    for cx, h in zip(COL_X_NUM_C, ["Search energy\n/query", "Latency\n/query", "Avg.\npower", "vs. analog\n(same row pair)"]):
        ax.text(cx, y_r, h, fontsize=FS_HEAD-1.5, fontweight="bold", va="center", ha="center", linespacing=1.3)
    return y_r - 0.040

def draw_row(y, method, write_e, search_e, lat, power, ratio, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y, method, fontsize=FS_BODY, fontweight=fw, va="center")
    ax.text(COL_X[1], y, write_e, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")
    for cx, val in zip(COL_X_NUM_C, [search_e, lat, power, ratio]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

def draw_section(y, label, italic=True):
    ax.text(TABLE_RIGHT / 2, y, label, fontsize=FS_SEC, style="italic" if italic else "normal",
            fontweight="normal" if italic else "bold", va="center", ha="center")

y = 1.0 - top_margin
hline(y, 1.8)
y_after_head = draw_header(y)
hline(y_after_head, 1.0)
y = y_after_head - ROW_H * 0.55

draw_section(y, "(i) Prototype scale -- 100 coarse-search centroids, candidates A=184 / C=193", italic=False)
y -= ROW_H
draw_section(y, "Analog in-memory distance computing (this work, SPICE-simulated)")
y -= ROW_H
for row in proto_analog:
    draw_row(y, *row)
    y -= ROW_H
y -= ROW_H * 0.1
draw_section(y, "GPU baseline -- measured (nvidia-smi power, this workstation)")
y -= ROW_H
for row in proto_gpu:
    draw_row(y, *row)
    y -= ROW_H
y -= ROW_H * 0.1
draw_section(y, "GPU baseline -- estimated (TDP x measured RTX 3070 Ti utilization, not measured)")
y -= ROW_H
for row in proto_gpu_est:
    draw_row(y, *row)
    y -= ROW_H

y -= ROW_H * 0.2
hline(y + ROW_H * 0.45, 1.0)

draw_section(y, "(ii) System-level scale -- 2,048 centroids, candidates A=10,988 / C(v4)=10,292 (matches Table 2)", italic=False)
y -= ROW_H
draw_section(y, "Analog in-memory distance computing (this work, SPICE-simulated)")
y -= ROW_H
for row in sys_analog:
    draw_row(y, *row)
    y -= ROW_H
y -= ROW_H * 0.1
draw_section(y, "GPU baseline -- measured (nvidia-smi power, this workstation)")
y -= ROW_H
for row in sys_gpu:
    draw_row(y, *row)
    y -= ROW_H

y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.040,
        "Table 3: Per-query energy, latency, and power -- analog Vth-compensated/no-compensation cells vs. GPU, at two scales.\n"
        "Search energy/query: 32 query tokens x (coarse-search centroids + candidate scoring), 1 μs sensing window per token-step.\n"
        "Write energy: one-time cost to program the coarse-search table (centroids x 128 dims), amortized over all queries sharing\n"
        "the index. Analog latency assumes per-token-serial / per-row-parallel array operation (32 coarse + 32 scoring steps).\n"
        "System-level candidate counts (A=10,988 avg., C=10,292 avg. under PBS v4=[0,3]V) measured directly from the 255-query,\n"
        "50,000-passage, 2,048-centroid, nprobe=2 pipeline (phase3_vth_pipeline.py) underlying Table 2.\n"
        "GPU rows use batch=256 (256 queries issued as one kernel call), matching standard serving practice and amortizing\n"
        "per-call CUDA launch overhead; unbatched (batch=1) GPU energy is 1-2 orders of magnitude higher due to that overhead.",
        fontsize=9.2, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
