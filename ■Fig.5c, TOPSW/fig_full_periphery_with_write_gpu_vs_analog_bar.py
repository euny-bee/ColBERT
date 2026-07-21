"""GPU vs. full-periphery analog energy efficiency at the system-level scale.

The original Cell+ADC+DAC figures are left untouched. This standalone variant uses
GPU=0.0270 TOPS/W and analog=0.3455 TOPS/W, including full search periphery and the
amortized one-time index-write cost.
Output: fig_full_periphery_with_write_topsw_bar.png
"""
from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = Path(__file__).resolve().parent

def sci_fmt(v, sig=3):
    """3-sig-fig scientific notation, e.g. 3949707.8 -> '3.95×10^6' (mathtext superscript,
    so the minus sign renders correctly regardless of the body font's glyph coverage)."""
    exp = math.floor(math.log10(abs(v)))
    mantissa = round(v / 10**exp, sig - 1)
    if mantissa >= 10:            # rounding carried over (e.g. 9.996 -> 10.0)
        mantissa /= 10
        exp += 1
    return f"{mantissa:.{sig-1}f}" + r"$\times\mathdefault{10^{%d}}$" % exp

available = {f.name for f in fm.fontManager.ttflist}
sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    sans,
    "font.size":          13,
    "font.weight":        "bold",
    "axes.labelsize":     23,
    "axes.labelweight":   "bold",
    "axes.titlesize":     16,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    18,
    "ytick.labelsize":    18,
    "axes.unicode_minus": False,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
})

# ── same red/blue convention as r50_opt4_star (red = baseline/worse, blue = this work/better) ──
COLOR_GPU    = "#C62828"   # red  -- GPU baseline
COLOR_ANALOG = "#1565C0"   # blue -- analog (this work)

labels = ["GPU\n(RTX 3070 Ti,\nmeasured)", "Analog\n(Full periphery + write,\nthis work)"]

RATIO_TEXT = "Analog is 12.8× more\nenergy-efficient than the GPU baseline"
CAPTION = (
    "GPU: RTX 3070 Ti, batch=256, measured (nvidia-smi power polling). Analog: Xyce SPICE-simulated\n"
    "cell search + amortized index-write energy + TIA + ADC + DAC + RWL/SBL search driver + ML buffer.\n"
    "System-level scale (2,048 centroids, 10,988 documents, 255 queries), Option A, full periphery.\n"
    "Process node not normalized: ADC (45 nm), DAC (as-cited process), analog cell (IGZO TFT, not a\n"
    "standard CMOS node), and GPU (Samsung 8 nm) are each taken as-is from their own source. Precision\n"
    "not matched either: GPU measured in FP32 (PyTorch default, no fp16/int8 cast); analog uses 8-bit\n"
    "ADC/DAC quantization."
)

def make_figure(vals, ylabel, val_fmt, title, out_name,
                show_values=True, xlabels=None, ymax_mult=20,
                xtick_fs=20, ylabel_fs=25, ytick_fs=24, yscale="log",
                yticks=None, ytick_labels=None, linear_ymax=None, value_fs=21.6,
                figsize=(5.8, 6.2), spine_lw=1.2, xlim_margin=0.5):
    xlabels = labels if xlabels is None else xlabels
    fig, ax = plt.subplots(figsize=figsize)
    x = [0, 1]
    bars = ax.bar(x, vals, width=0.5, color=[COLOR_GPU, COLOR_ANALOG], zorder=3)
    ax.set_yscale(yscale)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=xtick_fs, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=ylabel_fs, fontweight="bold", labelpad=2)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.tick_params(axis="y", labelsize=ytick_fs)

    ax.grid(True, axis="y", which="major", ls="--", alpha=0.3, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(spine_lw)

    ymax = max(vals)
    if show_values:
        for rect, v in zip(bars, vals):
            label_y = (v * 1.4 if yscale == "log" else
                       v + (linear_ymax if linear_ymax is not None else ymax * ymax_mult) * 0.04)
            ax.text(rect.get_x() + rect.get_width() / 2, label_y, val_fmt(v),
                    ha="center", va="bottom", fontsize=value_fs, fontweight="bold")
    if yscale == "log":
        ax.set_ylim(min(vals) / 6, ymax * ymax_mult)
    else:
        ax.set_ylim(0, linear_ymax if linear_ymax is not None else ymax * ymax_mult)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels, fontsize=ytick_fs)
    ax.set_xlim(-xlim_margin, 1 + xlim_margin)

    ax.text(0.5, 1.16, RATIO_TEXT, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=13, fontweight="bold", style="italic",
            color=COLOR_ANALOG)

    fig.text(0.02, -0.04, CAPTION, fontsize=8.6, ha="left", va="top")

    out = SCRIPT_DIR / out_name
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out}")

make_figure(
    [3_949_707.8, 309_126.58], "Energy / query (nJ)",
    lambda v: f"{v:,.1f} nJ" if v >= 1_000_000 else f"{v:,.2f} nJ",
    "", "fig_full_periphery_with_write_energy_bar.png",
    show_values=True, ymax_mult=3, xlabels=["GPU", "This work"],
)
make_figure(
    [3_949_707.8, 309_126.58], "Energy / query (×10⁶ nJ)",
    lambda v: f"{v:,.1f} nJ" if v >= 1_000_000 else f"{v:,.2f} nJ",
    "", "fig_full_periphery_with_write_energy_bar_linear.png",
    show_values=True, xlabels=["GPU", "This work"], yscale="linear",
    yticks=[0, 1_000_000, 2_000_000, 3_000_000, 4_000_000],
    ytick_labels=["0", "1", "2", "3", "4"], linear_ymax=4_800_000,
    xtick_fs=24, value_fs=26.136, spine_lw=0.8,
    # bigger value_fs widens "3,949,707.8 nJ" enough that, centered on the GPU
    # bar at the default xlim margin, it clips the left spine -- widen the
    # margin (symmetrically, so the bars stay centered) to clear it.
    xlim_margin=0.62,
)
make_figure(
    [0.0270, 0.3455], "Energy efficiency (TOPS/W)",
    sci_fmt,
    "Energy efficiency", "fig_full_periphery_with_write_topsw_bar.png",
    show_values=False, xlabels=["GPU", "This work"],
    xtick_fs=16, ylabel_fs=20, ytick_fs=19.2,
)
make_figure(
    [0.0270, 0.3455], "Energy efficiency (TOPS/W)",
    sci_fmt,
    "Energy efficiency", "fig_full_periphery_with_write_topsw_bar_axes15smaller.png",
    show_values=False, xlabels=["GPU", "This work"],
    xtick_fs=13.6, ylabel_fs=17, ytick_fs=16.32,
)
make_figure(
    [0.0270, 0.3455], "Energy efficiency (TOPS/W)",
    sci_fmt,
    "Energy efficiency", "fig_full_periphery_with_write_topsw_bar_linear.png",
    show_values=False, xlabels=["GPU", "This work"], ymax_mult=1.25,
    xtick_fs=13.6, ylabel_fs=17, ytick_fs=16.32, yscale="linear",
)
make_figure(
    [0.0270, 0.3455], "Energy efficiency (TOPS/W)",
    sci_fmt,
    "Energy efficiency", "fig_full_periphery_with_write_topsw_bar_linear_ticks0p1.png",
    show_values=False, xlabels=["GPU", "This work"],
    xtick_fs=16.56, ylabel_fs=16.56, ytick_fs=16.56, yscale="linear",
    yticks=[0.0, 0.1, 0.2, 0.3, 0.4], linear_ymax=0.42,
    # axes box (the black-bordered plot rect) sized in inches to match
    # [graph_optC]mrr_final.png's axes box (5.03 x 3.88 in), solved iteratively
    # since tight_layout's margins are ~fixed in inches, not a fixed fraction.
    # Re-solved after matching tick/label font sizes to mrr_final.png (16.56pt),
    # since larger fonts widen the margins and shrink the box at a fixed figsize.
    figsize=(6.062, 5.924),
    # mrr_final.png never overrides spine linewidth, so it uses matplotlib's
    # default of 0.8pt -- match that here instead of the 1.2pt used elsewhere.
    spine_lw=0.8,
)
