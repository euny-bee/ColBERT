from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

OUT = Path(r"C:\Users\nmdl-khb\AppData\Local\Temp\claude\c--Users-nmdl-khb-ColBERT-TOPSW\20aa624b-a34d-44aa-8475-8732670e3e23\scratchpad")

available = {f.name for f in fm.fontManager.ttflist}
sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": sans, "font.size": 13, "font.weight": "bold",
    "axes.labelsize": 23, "axes.labelweight": "bold", "xtick.labelsize": 20, "ytick.labelsize": 20,
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
})

pbs_labels = ["[0, 0.5]", "[0, 1.0]", "[0, 2.0]", "[0, 3.0]"]
x = np.arange(4)
vth_comp = np.array([78.8, 78.8, 78.8, 78.8])
no_comp  = np.array([78.1, 71.3, 27.8,  2.1])

COLOR_COMP   = "#2196F3"
COLOR_NOCOMP = "#FF7811"
GRAY_FILL    = "#9E9E9E"

def make(variant, fill_color, fill_alpha, use_fill=True):
    fig, ax = plt.subplots(figsize=(5.9, 4.6))
    if use_fill:
        ax.fill_between(x, vth_comp, no_comp, alpha=fill_alpha, color=fill_color)
    ax.plot(x, no_comp,  marker='o', ms=7,  color=COLOR_NOCOMP, lw=1.5, zorder=4)
    ax.plot(x, vth_comp, marker='o', ms=12, color=COLOR_COMP,   lw=4.5, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(pbs_labels, fontsize=20, fontweight="bold")
    ax.set_xlabel("PBS-Induced Vth Shift Range [V]", fontsize=23, fontweight="bold", labelpad=8)
    ax.set_ylabel("MRR@10 (%)", fontsize=23, fontweight="bold", labelpad=4)
    ax.set_xlim(-0.15, 3.15); ax.set_ylim(0, 92)
    ax.grid(True, axis="y", ls="--", alpha=0.3, lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout(pad=0.3)
    out = OUT / f"mrr_variant_{variant}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out}")

# A: neutral gray fill instead of orange (decouple annotation-shading from series color)
make("A_grayfill", GRAY_FILL, 0.13)
# B: no fill at all, just the two lines
make("B_nofill", None, 0, use_fill=False)
# C: orange fill but much lower alpha (current color, quieter)
make("C_lowalpha", COLOR_NOCOMP, 0.07)
