"""
plot_fig1.py — fig1_residual_distribution: residual value histogram (float16 vs 2-bit)

Data  : colbert_float16_correct.xlsx + colbert_2bit_correct.csv (same dir)
Output: fig1_residual_distribution.png (300 dpi) in the same directory
Style : matches [vth_v3]row0_scatter.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
F16_XLSX   = SCRIPT_DIR / "colbert_float16_correct.xlsx"
BIT2_CSV   = SCRIPT_DIR / "colbert_2bit_correct.csv"
OUTPUT      = SCRIPT_DIR / "fig1_residual_distribution.png"
OUTPUT_LEG  = SCRIPT_DIR / "fig1_legend.png"

DIM           = 128
RESIDUAL_COLS = [f"residual_dim_{i}" for i in range(DIM)]

# ── Style (matches [vth_v3]row0_scatter.png) ─────────────────────────────────
FONT_BASE  = 16
FONT_PANEL = 18
FONT_LABEL = 26
FONT_TICK  = 24
FONT_LEG   = 17
GRID_ALPHA = 0.3

COLOR_F16  = "#4878CF"
COLOR_2BIT = "#E8604C"
ALPHA_HIST = 0.75

MAX_SAMPLE = 500_000


def _setup_matplotlib() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    sans,
        "font.size":          FONT_BASE,
        "font.weight":        "bold",
        "axes.labelsize":     FONT_LABEL,
        "axes.labelweight":   "bold",
        "axes.titlesize":     FONT_PANEL,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    FONT_TICK,
        "ytick.labelsize":    FONT_TICK,
        "axes.unicode_minus": False,
    })


def _style_ax(ax) -> None:
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8, ls="--")
    ax.minorticks_off()


def load_data():
    print("Loading colbert_float16_correct.xlsx ...")
    df_f16  = pd.read_excel(F16_XLSX)
    print("Loading colbert_2bit_correct.csv ...")
    df_2bit = pd.read_csv(BIT2_CSV, encoding="cp949")

    flat_f16  = df_f16[RESIDUAL_COLS].to_numpy(dtype=np.float32).flatten()
    flat_2bit = df_2bit[RESIDUAL_COLS].to_numpy(dtype=np.float32).flatten()

    n = len(df_f16)
    print(f"  {n:,} tokens loaded  ({len(flat_f16):,} values per distribution)")

    f16_min, f16_max = float(flat_f16.min()), float(flat_f16.max())

    rng = np.random.default_rng(42)
    if len(flat_f16) > MAX_SAMPLE:
        idx = rng.choice(len(flat_f16), MAX_SAMPLE, replace=False)
        flat_f16  = flat_f16[idx]
        flat_2bit = flat_2bit[idx]

    return flat_f16, flat_2bit, n, f16_min, f16_max


def plot_fig1(flat_f16: np.ndarray, flat_2bit: np.ndarray, n: int,
              font_label: int = FONT_LABEL, font_tick: int = FONT_TICK,
              output: Path = OUTPUT,
              f16_min: float = None, f16_max: float = None,
              xlim: tuple = None, xtick_step: float = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(flat_f16,  bins=300, density=True, alpha=ALPHA_HIST,
            color=COLOR_F16,  edgecolor="none",
            label="float16 (continuous)")
    ax.hist(flat_2bit, bins=20,  density=True, alpha=ALPHA_HIST + 0.1,
            color=COLOR_2BIT, edgecolor="none",
            label="2-bit (4 discrete values)")

    unique_vals = np.unique(flat_2bit)
    for v in unique_vals:
        ax.axvline(v, color=COLOR_2BIT, linestyle="--", linewidth=1.5, alpha=0.65)

    # float16 min/max 파란색 점선
    _min = f16_min if f16_min is not None else float(flat_f16.min())
    _max = f16_max if f16_max is not None else float(flat_f16.max())
    ax.axvline(_min, color=COLOR_F16, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(_max, color=COLOR_F16, linestyle="--", linewidth=1.5, alpha=0.8)

    # bin 경계선: 인접한 2-bit 레벨의 중간값
    bin_boundaries = [(unique_vals[i] + unique_vals[i + 1]) / 2
                      for i in range(len(unique_vals) - 1)]
    for b in bin_boundaries:
        ax.axvline(b, color=COLOR_F16, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Residual", fontsize=font_label)
    ax.set_ylabel("Density",  fontsize=font_label, labelpad=8)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(_min - 0.04, _max + 0.04)
    if xtick_step is not None:
        x1 = xlim[1] if xlim is not None else (_max + 0.04)
        n = int(np.floor(x1 / xtick_step))
        ax.set_xticks(np.arange(-n * xtick_step, n * xtick_step + xtick_step * 0.01, xtick_step))
    ax.tick_params(axis="x", labelsize=font_tick)
    ax.tick_params(axis="y", labelsize=font_tick)

    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output}")

    # ── 범례 단독 figure ─────────────────────────────────────────────────────
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=COLOR_2BIT, alpha=ALPHA_HIST + 0.1, label="2-bit quantized"),
        mpatches.Patch(color=COLOR_F16,  alpha=ALPHA_HIST,        label="no quantized"),
    ]

    fig_leg, ax_leg = plt.subplots(figsize=(4, 1.4))
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, fontsize=FONT_LEG,
                  loc="center", frameon=True)
    fig_leg.savefig(OUTPUT_LEG, dpi=300, bbox_inches="tight")
    plt.close(fig_leg)
    print(f"  → {OUTPUT_LEG}")


def main() -> None:
    _setup_matplotlib()
    flat_f16, flat_2bit, n, f16_min, f16_max = load_data()
    print("Plotting Figure 1 ...")
    plot_fig1(flat_f16, flat_2bit, n, f16_min=f16_min, f16_max=f16_max)

    print("Plotting Figure 1 (large fonts) ...")
    plot_fig1(flat_f16, flat_2bit, n,
              font_label=int(FONT_LABEL * 1.2),
              font_tick=int(FONT_TICK * 1.2),
              output=SCRIPT_DIR / "fig1_residual_distribution_large.png",
              f16_min=f16_min, f16_max=f16_max,
              xlim=(-0.3, 0.3))
    print("Plotting Figure 1 (large fonts, wide xlim) ...")
    plot_fig1(flat_f16, flat_2bit, n,
              font_label=int(FONT_LABEL * 1.2),
              font_tick=int(FONT_TICK * 1.2),
              output=SCRIPT_DIR / "fig1_residual_distribution_large2.png",
              f16_min=f16_min, f16_max=f16_max,
              xlim=(-0.22, 0.22))
    print("Plotting Figure 1 (large3) ...")
    plot_fig1(flat_f16, flat_2bit, n,
              font_label=int(FONT_LABEL * 1.2),
              font_tick=int(FONT_TICK * 1.2),
              output=SCRIPT_DIR / "fig1_residual_distribution_large3.png",
              f16_min=f16_min, f16_max=f16_max,
              xlim=(-0.21, 0.21))
    for suffix, lim, xstep in [("large4", 0.20, None), ("large5", 0.19, None),
                                ("large6", 0.18, 0.06), ("large7", 0.17, 0.06), ("large8", 0.16, 0.06)]:
        print(f"Plotting Figure 1 ({suffix}) ...")
        plot_fig1(flat_f16, flat_2bit, n,
                  font_label=int(FONT_LABEL * 1.2),
                  font_tick=int(FONT_TICK * 1.2),
                  output=SCRIPT_DIR / f"fig1_residual_distribution_{suffix}.png",
                  f16_min=f16_min, f16_max=f16_max,
                  xlim=(-lim, lim), xtick_step=xstep)
    print("Done.")


if __name__ == "__main__":
    main()
