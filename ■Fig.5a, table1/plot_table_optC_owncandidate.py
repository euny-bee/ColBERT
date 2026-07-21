"""
Table 2 — Option C (NoComp), 자체 후보검색 full pipeline
  Step2/3(centroid 선택, IVF lookup)부터 Step6(MaxSim)까지 전부 dead-zone 회로로 수행
  셀단위 고정 Vth(1+2 조합) 기준 결과
Output: table_optC_owncandidate.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT     = SCRIPT_DIR / "table_optC_owncandidate.png"

available = {f.name for f in fm.fontManager.ttflist}
serif = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in available] or ["DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": serif, "axes.unicode_minus": False})

# ── Table content ─────────────────────────────────────────────────────────────
# (method, pbs_cond, MRR@10, nDCG@10, R@50, R@1k, cand_recall)
rows_baseline = [
    ("Digital (float32)",       "--", "78.4", "83.1", "98.8",  "99.6", "  --"),
    ("Digital (2bit residual)", "--", "76.9", "81.9", "98.8", "100.0", "  --"),
    ("Option A (VthComp)",      "--", "78.8", "83.4", "98.8",  "99.6", "  --"),
]
rows_optC = [
    ("Option C (NoComp)", "v1  [0, 0.5] V", "78.1", "82.3", "98.8", "99.6", "100.0%"),
    ("Option C (NoComp)", "v2  [0, 1.0] V", "71.3", "75.9", "94.1", "96.5",  "96.5%"),
    ("Option C (NoComp)", "v3  [0, 2.0] V", "27.8", "32.0", "58.8", "67.1",  "68.2%"),
    ("Option C (NoComp)", "v4  [0, 3.0] V",  "2.1",  "3.4", "23.1", "48.2",  "51.0%"),
]

# 7 columns: method, cond, MRR, nDCG, R50, R1k, cand_recall
COL_X       = [0.02, 0.33, 0.56, 0.645, 0.730, 0.815, 0.900]
COL_X_NUM_C = [0.603, 0.688, 0.773, 0.858, 0.943]
ROW_H = 0.072
FS_HEAD, FS_SEC, FS_BODY = 11.5, 11.0, 11.5

n_rows = 1 + len(rows_baseline) + 1 + len(rows_optC)
fig, ax = plt.subplots(figsize=(10.0, 1.55 + n_rows * 0.40))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
top_margin = 0.045

def hline(y, lw):
    ax.plot([0,1],[y,y], color="black", lw=lw, transform=ax.transAxes,
            solid_capstyle="butt", clip_on=False)

def draw_header(y_top):
    y_r = y_top - 0.030
    ax.text(COL_X[0], y_r, "Method",        fontsize=FS_HEAD, fontweight="bold", va="center")
    ax.text(COL_X[1], y_r, "PBS Condition", fontsize=FS_HEAD, fontweight="bold", va="center")
    for cx, h in zip(COL_X_NUM_C, ["MRR@10","nDCG@10","R@50","R@1k","Cand.\nRecall"]):
        ax.text(cx, y_r, h, fontsize=FS_HEAD-1, fontweight="bold", va="center", ha="center")
    return y_r - 0.035

def draw_row(y, method, cond, mrr, ndcg, r50, r1k, recall, bold=False):
    fw = "bold" if bold else "normal"
    ax.text(COL_X[0], y, method, fontsize=FS_BODY, fontweight=fw, va="center")
    ax.text(COL_X[1], y, cond,   fontsize=FS_BODY, fontweight=fw, va="center")
    for cx, val in zip(COL_X_NUM_C, [mrr, ndcg, r50, r1k, recall]):
        ax.text(cx, y, val, fontsize=FS_BODY, fontweight=fw, va="center", ha="center")

def draw_section(y, label):
    ax.text(0.5, y, label, fontsize=FS_SEC, style="italic", va="center", ha="center")

y = 1.0 - top_margin
hline(y, 1.8)
y_after_head = draw_header(y)
hline(y_after_head, 1.0)
y = y_after_head - ROW_H * 0.55

draw_section(y, "No PBS Stress (Baseline)")
y -= ROW_H
for row in rows_baseline:
    draw_row(y, *row)
    y -= ROW_H

y -= ROW_H * 0.15
hline(y + ROW_H * 0.45, 0.6)

draw_section(y, r"Under PBS-Induced $V_{th}$ Variation  —  Full Pipeline (Step 2/3 + Step 6)")
y -= ROW_H
for row in rows_optC:
    draw_row(y, *row)
    y -= ROW_H

y_bottom = y + ROW_H * 0.45
hline(y_bottom, 1.8)

ax.text(0.0, y_bottom - 0.055,
        "Table 2: Retrieval quality (%) under PBS-induced $V_{th}$ variation — full pipeline variant.\n"
        "Option C's dead-zone circuit is applied to both candidate retrieval (Step 2/3) and MaxSim reranking (Step 6)\n"
        "with per-cell fixed $V_{th}$. Candidate recall reflects the fraction of queries for which the true passage was retrieved.",
        fontsize=10, va="top", ha="left", transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
