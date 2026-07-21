"""
Standalone derivation schematic, split out of table_periphery_adc_dac_combined.py --
just the top panel (Cell + ADC + DAC = Energy/query -> TOPS/W worked example), without
the merged nJ/TOPS/W table underneath. Same content/numbers, own figure.
Output: fig_energy_derivation_schematic.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "fig_energy_derivation_schematic.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
sans  = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

BLUE = "#1565C0"
INK  = "0.25"

fig, ax_top = plt.subplots(figsize=(13.6, 3.55), dpi=300)
ax_top.set_xlim(0, 1.05); ax_top.set_ylim(0, 1); ax_top.axis("off")

# ── Derivation schematic: Cell + ADC + DAC = Energy/query -> TOPS/W ──────────
ax_top.text(0.5, 0.955, "How Energy/query is built up, and how it becomes energy efficiency (TOPS/W)",
            fontsize=14.5, fontweight="bold", ha="center", va="center", family=sans[0])
ax_top.text(0.5, 0.865,
            "System-level scale (2,048 centroids). N_row_total = N_centroid + N_candidate = 2,048 + 10,988 = 13,036 rows,\n"
            "N_TOKENS = 32 tokens/query,  N_COLS = 128 (embedding dim)",
            fontsize=10.5, ha="center", va="center", color=INK, family=sans[0])

box_y = 0.56
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
ax_top.text(0.5, 0.135,
        "Worked example for the headline result (“+DAC (final)”, system-level scale, 2,048 centroids). The prototype\n"
        "scale (100 centroids) follows the same formulas with N_row_total = 100 + 184 = 284 (see Table 4-ADC/DAC).\n"
        "Because the analog engine and GPU baseline share the same FLOPs/query definition, FLOPs cancel in the\n"
        "ratio, so “vs. GPU” is identical whether read from Energy/query or TOPS/W.",
        fontsize=9.3, ha="center", va="top", style="italic", color=INK)

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"Saved: {OUTPUT}")
