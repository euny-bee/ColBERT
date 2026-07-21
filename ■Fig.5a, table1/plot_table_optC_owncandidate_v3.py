"""
Table 2 — Option C (NoComp), 자체 후보검색 full pipeline
  2단계 컬럼 헤더: Scoring (MRR/nDCG/R@50/R@1k) | Coarse search (Cand.recall)
Output: table_optC_owncandidate_v3.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_optC_owncandidate_v3.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content ─────────────────────────────────────────────────────────────
rows_baseline = [
    ("Digital (no quantization)",  "--", "78.4", "83.1", "98.8",  "99.6", "100.0"),
    ("Digital (2bit quantization)","--", "76.9", "81.9", "98.8", "100.0", "100.0"),
    ("Vth compensation",           "--", "78.8", "83.4", "98.8",  "99.6", "100.0"),
]
rows_optC = [
    ("No compensation", "[0, 0.5] V", "78.1", "82.3", "98.8", "99.6", "100.0"),
    ("",                "[0, 1.0] V", "71.3", "75.9", "94.1", "96.5",  "96.5"),
    ("",                "[0, 2.0] V", "27.8", "32.0", "58.8", "67.1",  "68.2"),
    ("",                "[0, 3.0] V",  "2.1",  "3.4", "23.1", "48.2",  "51.0"),
]

# 7 columns: method, cond, MRR, nDCG, R50, R1k, cand_recall
COL_X       = [0.02, 0.23]
COL_X_NUM_C = [0.42, 0.512, 0.604, 0.696, 0.788]
TABLE_RIGHT = 0.86
ROW_H = 0.068
FS_HEAD, FS_GRP, FS_SEC, FS_BODY = 11.5, 11.0, 11.0, 11.5

n_rows = 1 + len(rows_baseline) + 1 + len(rows_optC)
fig, ax = plt.subplots(figsize=(9.0, 2.1 + n_rows * 0.40))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
top_margin = 0.040

def hline(y, lw, x0=0, x1=None):
    if x1 is None:
        x1 = TABLE_RIGHT
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    # ── Level 1: 그룹 헤더 ──────────────────────────────────────────────────
    y_grp = y_top - 0.028

    scoring_cx = (COL_X_NUM_C[0] + COL_X_NUM_C[3]) / 2
    ax.text(scoring_cx, y_grp, "Scoring",
            fontsize=FS_GRP, fontweight="bold", va="center", ha="center")
    hline(y_grp - 0.016, 0.5, COL_X_NUM_C[0] - 0.036, COL_X_NUM_C[3] + 0.036)

    ax.text(COL_X_NUM_C[4], y_grp, "Coarse search",
            fontsize=FS_GRP, fontweight="bold", va="center", ha="center")
    hline(y_grp - 0.016, 0.5, COL_X_NUM_C[4] - 0.048, COL_X_NUM_C[4] + 0.048)

    # ── Level 2: 컬럼 헤더 ──────────────────────────────────────────────────
    y_r = y_grp - 0.038
    ax.text(COL_X[0], y_r, "Method",          fontsize=FS_HEAD, fontweight="bold", va="center")
    ax.text(COL_X[1], y_r, "Vth shift range", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_X_NUM_C, ["MRR@10", "nDCG@10", "R@50", "R@1k", "Cand. recall"]):
        ax.text(cx, y_r, h, fontsize=FS_HEAD-1, fontweight="bold", va="center", ha="center")
    return y_r - 0.038

def draw_row(y, method, cond, mrr, ndcg, r50, r1k, recall, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y, method, fontsize=FS_BODY, fontweight=fw, va="center")
    ax.text(COL_X[1], y, cond,   fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [mrr, ndcg, r50, r1k, recall]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

def draw_section(y, label):
    ax.text(TABLE_RIGHT / 2, y, label, fontsize=FS_SEC, style="italic", va="center", ha="center")

y = 1.0 - top_margin
hline(y, 1.8)
y_after_head = draw_header(y)
hline(y_after_head, 1.0)
y = y_after_head - ROW_H * 0.55

draw_section(y, "PBS-invariant baseline")
y -= ROW_H
for row in rows_baseline:
    draw_row(y, *row)
    y -= ROW_H

y -= ROW_H * 0.15
hline(y + ROW_H * 0.45, 0.6)

draw_section(y, "Under PBS-induced Vth variation")
y -= ROW_H
for row in rows_optC:
    draw_row(y, *row)
    y -= ROW_H

y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.045,
        "Table 2: Retrieval quality under PBS-induced Vth variation — full pipeline.\n"
        "Option C dead-zone circuit applied to both Coarse search (Phase 1) and Scoring (Phase 2).\n"
        "Cand. recall: fraction of queries where the true passage was retrieved.",
        fontsize=10, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
