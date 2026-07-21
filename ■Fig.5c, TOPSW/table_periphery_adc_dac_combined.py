"""
Table 4-ADC/DAC (combined) -- Supplementary figure for Fig. 5c. Shows the full derivation
chain from raw event counts down to TOPS/W in one figure:
    Cell (Xyce SPICE) + ADC (1.5 pJ/conv. x events) + DAC (0.52 pJ/conv. x events)
        = Energy/query (nJ)  --x1000, /FLOPs/query-->  Energy efficiency (TOPS/W)
Top panel: worked example for the headline result ("+DAC (final)", system-level scale,
2,048 centroids), each term's own event-count formula shown directly under its box.
Bottom panel: table_periphery_adc_dac.py (nJ) and table_periphery_adc_dac_topsw.py (TOPS/W)
merged into one table for both scales, with the "vs. GPU" ratio shown once per scale since
it is numerically identical whichever unit you read it from (FLOPs cancel).
Source values transcribed from "TOPSW 분석.pptx" slide 7 (ADC/DAC event-count derivation,
GPU measured energy/query) and TOPS_W_analysis_summary.md (Cell/Xyce methodology).
Output: table_periphery_adc_dac_combined.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_periphery_adc_dac_combined.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
sans  = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

BLUE = "#1565C0"
INK  = "0.25"

# ── Table content (scenario, proto_energy_nJ, proto_topsw, proto_ratio,
#                              sys_energy_nJ,   sys_topsw,   sys_ratio) ─────────
rows = [
    ("Cell only",     "5.97 nJ",  "389.70 TOPS/W", "33,602×", "274.03 nJ", "389.71 TOPS/W", "14,414×"),
    ("+ ADC",         "19.60 nJ", "118.70 TOPS/W", "10,234×", "899.76 nJ", "118.69 TOPS/W", "4,390×"),
    ("+ DAC (final)", "21.73 nJ", "107.07 TOPS/W", "9,231×",  "901.89 nJ", "118.41 TOPS/W", "4,379×"),
]
gpu_row = ("GPU (RTX 3070 Ti, measured)", "200,600.7 nJ", "0.0116 TOPS/W", "1×",
           "3,949,707.8 nJ", "0.0270 TOPS/W", "1×")

# ── Figure / axes layout ────────────────────────────────────────────────────
FS_HEAD, FS_GRP, FS_BODY = 11, 11, 11
ROW_H = 0.115
n_rows = 2 + len(rows) + 1
table_h_in = 2.15 + n_rows * 0.40
top_h_in = 5.3
fig_h = top_h_in + table_h_in
fig = plt.figure(figsize=(13.6, fig_h))
gs = fig.add_gridspec(2, 1, height_ratios=[top_h_in, table_h_in], hspace=0.0)
ax_top = fig.add_subplot(gs[0]); ax_top.set_xlim(0, 1.05); ax_top.set_ylim(0, 1); ax_top.axis("off")
ax_tab = fig.add_subplot(gs[1]); ax_tab.set_xlim(0, 1.18); ax_tab.set_ylim(0, 1); ax_tab.axis("off")

# ── 1. Derivation schematic: Cell + ADC + DAC = Energy/query -> TOPS/W ──────
ax_top.text(0.5, 0.965, "How Energy/query is built up, and how it becomes energy efficiency (TOPS/W)",
            fontsize=14.5, fontweight="bold", ha="center", va="center", family=sans[0])
ax_top.text(0.5, 0.90,
            "System-level scale (2,048 centroids). N_row_total = N_centroid + N_candidate = 2,048 + 10,988 = 13,036 rows,\n"
            "N_TOKENS = 32 tokens/query,  N_COLS = 128 (embedding dim)",
            fontsize=10.5, ha="center", va="center", color=INK, family=sans[0])

box_y = 0.62
term_boxes = [
    ("Cell (Xyce)\n274.03 nJ", 0.07),
    ("ADC\n625.73 nJ", 0.30),
    ("DAC\n2.13 nJ", 0.53),
]
sum_box = ("Energy/query\n901.89 nJ", 0.745)
out_box = ("118.41\nTOPS/W", 0.975)

bw_t, bh_t = 0.13, 0.20
bw_s, bh_s = 0.185, 0.24
bw_o, bh_o = 0.12, 0.20

def draw_box(txt, cx, cy, bw, bh, hi=True, fs=10.2):
    fc = "#E8F0FE" if hi else "white"
    ec = BLUE if hi else "0.45"
    rect = plt.Rectangle((cx - bw/2, cy - bh/2), bw, bh, transform=ax_top.transData,
                          facecolor=fc, edgecolor=ec, lw=1.3, zorder=2)
    ax_top.add_patch(rect)
    ax_top.text(cx, cy, txt, fontsize=fs, ha="center", va="center",
                fontweight="bold", family=sans[0], zorder=3)

for txt, cx in term_boxes:
    draw_box(txt, cx, box_y, bw_t, bh_t)
draw_box(sum_box[0], sum_box[1], box_y, bw_s, bh_s)
draw_box(out_box[0], out_box[1], box_y, bw_o, bh_o)

# "+" between the three term boxes, "=" before the sum box
plus_positions = [(term_boxes[0][1] + term_boxes[1][1]) / 2,
                   (term_boxes[1][1] + term_boxes[2][1]) / 2]
for px in plus_positions:
    ax_top.text(px, box_y, "+", fontsize=17, fontweight="bold", ha="center", va="center", color=INK)
eq_x = (term_boxes[2][1] + bw_t/2 + sum_box[1] - bw_s/2) / 2
ax_top.text(eq_x, box_y, "=", fontsize=17, fontweight="bold", ha="center", va="center", color=INK)

# arrow: Energy/query -> TOPS/W, with the unit-conversion formula as its label
arr_x0 = sum_box[1] + bw_s/2 + 0.012
arr_x1 = out_box[1] - bw_o/2 - 0.012
ax_top.annotate("", xy=(arr_x1, box_y), xytext=(arr_x0, box_y),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
arr_cx = (arr_x0 + arr_x1) / 2
ax_top.text(arr_cx, box_y + 0.135, "× 1000\n(nJ → pJ)", fontsize=8.5, ha="center", va="bottom",
            color=BLUE, family=sans[0])
ax_top.text(arr_cx, box_y - 0.135, "÷ FLOPs/query\n106,790,912", fontsize=8.5, ha="center", va="top",
            color=BLUE, family=sans[0])

# ── sub-notes: how each term's energy was derived ────────────────────────────
subnotes = [
    (term_boxes[0][1], "656.90 fJ/eval × 417,152 evals\n(Xyce SPICE search\nenergy, Option A)"),
    (term_boxes[1][1], "1.5 pJ/conv. × 417,152 events\n(events = N_row_total\n× N_TOKENS)"),
    (term_boxes[2][1], "0.52 pJ/conv. × 4,096 events\n(events = N_COLS\n× N_TOKENS)"),
]
sub_y0 = box_y - bh_t/2 - 0.05
for cx, note in subnotes:
    ax_top.text(cx, sub_y0, note, fontsize=8.0, ha="center", va="top", color=INK, family=sans[0])

ax_top.text(sum_box[1], box_y - bh_s/2 - 0.05,
            "sum of the three terms\nto the left",
            fontsize=8.0, ha="center", va="top", color=INK, family=sans[0])

# ── bottom note ───────────────────────────────────────────────────────────────
ax_top.text(0.5, 0.235,
        "Worked example for the headline result (“+DAC (final)”, system-level scale, 2,048 centroids). The prototype\n"
        "scale (100 centroids) follows the same formulas with N_row_total = 100 + 184 = 284 (see table below).\n"
        "Because the analog engine and GPU baseline share the same FLOPs/query definition, FLOPs cancel in the\n"
        "ratio, so “vs. GPU” is identical whether read from Energy/query or TOPS/W.",
        fontsize=9.3, ha="center", va="top", style="italic", color=INK)

# ── 2. Merged table ──────────────────────────────────────────────────────────
COL_X = [0.02]
COL_P = [0.335, 0.485, 0.60]
COL_S = [0.815, 0.965, 1.10]
COL_X_NUM_C = COL_P + COL_S
TABLE_RIGHT = 1.18

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax_tab.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax_tab.transAxes,
                solid_capstyle="butt", clip_on=False)

y = 0.985
hline(y, 1.8)

y_grp = y - 0.062
proto_cx = (COL_P[0] + COL_P[2]) / 2
ax_tab.text(proto_cx, y_grp, "Prototype scale (100 centroids)  —  FLOPs/query = 2,326,528",
            fontsize=FS_GRP - 0.6, fontweight="bold", va="center", ha="center")
hline(y_grp - 0.032, 0.5, COL_P[0] - 0.075, COL_P[2] + 0.055)

sys_cx = (COL_S[0] + COL_S[2]) / 2
ax_tab.text(sys_cx, y_grp, "System-level scale (2,048 centroids)  —  FLOPs/query = 106,790,912",
            fontsize=FS_GRP - 0.6, fontweight="bold", va="center", ha="center")
hline(y_grp - 0.032, 0.5, COL_S[0] - 0.075, COL_S[2] + 0.055)

y_r = y_grp - 0.085
ax_tab.text(COL_X[0], y_r, "Scenario (cumulative)", fontsize=FS_HEAD, fontweight="bold", va="center")
headers = ["Energy/query", "Energy eff.", "vs. GPU"] * 2
for cx, h in zip(COL_X_NUM_C, headers):
    ax_tab.text(cx, y_r, h, fontsize=FS_HEAD - 1, fontweight="bold", va="center", ha="center")
for grp in (COL_P, COL_S):
    ax_tab.annotate("", xy=(grp[1] - 0.058, y_r), xytext=(grp[0] + 0.048, y_r),
                     arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1))

y_after_head = y_r - 0.065
hline(y_after_head, 1.0)
y = y_after_head - ROW_H * 0.75

def draw_row(y, scenario, pe, pt, pr, se, st, sr, bold=False):
    fw = "bold" if bold else "normal"
    ax_tab.text(COL_X[0], y, scenario, fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [pe, pt, pr, se, st, sr]):
        ax_tab.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

for i, row in enumerate(rows):
    draw_row(y, *row, bold=(i == len(rows) - 1))
    y -= ROW_H

hline(y + ROW_H * 0.45, 0.8)
y -= ROW_H * 0.2
draw_row(y, *gpu_row)
y -= ROW_H

y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

ax_tab.text(0.0, y_bottom - 0.075,
        "Table 4-ADC/DAC (combined): Periphery-energy scenario restricted to the two conversion blocks around\n"
        "the analog core (A→D readout ADC, D→A query-injection DAC) — TIA, driver and ML buffer excluded —\n"
        "for Option A (Vth compensation) search, per query. ADC: 1.5 pJ/conversion, 8-bit 45 nm (Andrulis et al.,\n"
        "as-cited), N_rows_total × N_TOKENS events. DAC: 0.52 pJ/conversion, 8-bit (Hong & Lee 2007, 65 fJ/step\n"
        "×8, via Saberi et al. TCAS-I 2011), N_COLS × N_TOKENS events — energy independent of row count, so\n"
        "identical (2.13 nJ) at both scales. FLOPs/query = N_rows_total × N_TOKENS × 256 (128-dim dot-product =\n"
        "128 mult + 128 add, 2×dim convention), applied identically to the GPU (RTX 3070 Ti, batch=256, measured)\n"
        "baseline. “vs. GPU” is a single shared column per scale because the Energy/query-based and TOPS/W-based\n"
        "ratios are numerically identical (FLOPs cancel). Process node not normalized: ADC (45 nm), DAC (as-cited\n"
        "process), analog cell (IGZO TFT, not a standard CMOS node), and GPU (Samsung 8 nm) are each taken\n"
        "as-is from their own source. Precision not matched either: GPU measured in FP32 (PyTorch default, no\n"
        "fp16/int8 cast); analog uses 8-bit ADC/DAC quantization.",
        fontsize=8.8, va="top", ha="left", transform=ax_tab.transAxes)

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"Saved: {OUTPUT}")
