"""
GPU vs. Analog (Cell+ADC+DAC, Option A final) at system-level scale (2,048 centroids),
shown two ways -- Energy/query (nJ) and Energy efficiency (TOPS/W) -- both encoding the same
underlying 4,379x gap (verified: 3,949,707.8/901.89 = 4379.37, and 118.41/0.0270 = 4379.4,
same number both ways since FLOPs/query cancels in the ratio).
Style matched to [graph_optC]r50_opt4_star.png (Arial, bold labels, large tick fonts, dashed grid,
full box spines, red=baseline/blue=this-work color convention).
Rendered as two SEPARATE standalone figures (one meant as main, the other as an inset).
Outputs: fig_adc_dac_energy_bar.png, fig_adc_dac_topsw_bar.png
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

labels = ["GPU\n(RTX 3070 Ti,\nmeasured)", "Analog\n(Cell+ADC+DAC,\nthis work)"]

RATIO_TEXT = "Analog is 4,379× more\nenergy-efficient than the GPU baseline"
CAPTION = (
    "GPU: RTX 3070 Ti, batch=256, measured (nvidia-smi power polling). Analog: Xyce SPICE-simulated\n"
    "cell energy + ADC (1.5 pJ/conv., Andrulis et al., as-cited) + DAC (0.52 pJ/conv., Hong & Lee 2007\n"
    "via Saberi et al. TCAS-I 2011). System-level scale (2,048 centroids), Option A, Cell+ADC+DAC (final).\n"
    "Process node not normalized: ADC (45 nm), DAC (as-cited process), analog cell (IGZO TFT, not a\n"
    "standard CMOS node), and GPU (Samsung 8 nm) are each taken as-is from their own source. Precision\n"
    "not matched either: GPU measured in FP32 (PyTorch default, no fp16/int8 cast); analog uses 8-bit\n"
    "ADC/DAC quantization."
)

def make_figure(vals, ylabel, val_fmt, title, out_name,
                show_values=True, xlabels=None, ymax_mult=20):
    xlabels = labels if xlabels is None else xlabels
    fig, ax = plt.subplots(figsize=(5.8, 6.2))
    x = [0, 1]
    bars = ax.bar(x, vals, width=0.5, color=[COLOR_GPU, COLOR_ANALOG], zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=20, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=25, fontweight="bold", labelpad=2)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.tick_params(axis="y", labelsize=24)

    ax.grid(True, axis="y", which="major", ls="--", alpha=0.3, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ymax = max(vals)
    if show_values:
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v * 1.4, val_fmt(v),
                    ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylim(min(vals) / 6, ymax * ymax_mult)
    ax.set_xlim(-0.5, 1.5)

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
    [3_949_707.8, 901.89], "Energy / query (nJ)",
    sci_fmt,
    "", "fig_adc_dac_energy_bar.png",
    show_values=False, ymax_mult=3, xlabels=["GPU", "This work"],
)
make_figure(
    [0.0270, 118.41], "Energy efficiency (TOPS/W)",
    sci_fmt,
    "Energy efficiency", "fig_adc_dac_topsw_bar.png",
)
make_figure(
    [0.0270, 118.41], "Energy efficiency (TOPS/W)",
    sci_fmt,
    "Energy efficiency", "fig_adc_dac_topsw_bar_simple.png",
    show_values=False, xlabels=["GPU", "This work"],
)
