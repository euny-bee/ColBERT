"""
fig5_norm_distribution: 2-bit reconstruction norm ||centroid + residual||² distribution.

Reads colbert_float16_correct.csv and colbert_2bit_correct.csv from the ColBERT root.
Saves fig5_norm_distribution.png (300 dpi) in the same directory as this script.
Run from 논문 figure/: python plot_fig5.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "centroidset 만들기"
OUTPUT_DIR = SCRIPT_DIR

EMB_CSV      = DATA_DIR / "doc_embs_12919x128.csv"
RES_F16_CSV  = DATA_DIR / "residuals_float32.csv"
RES_2BIT_CSV = DATA_DIR / "residuals_2bit.csv"

DIM      = 128
DIM_COLS = [f"dim_{i}" for i in range(DIM)]

# ── Style constants ──────────────────────────────────────────────────────────
FONT_BASE      = 13
FONT_AXIS      = 14
FONT_PANEL     = 15
FONT_LABEL     = 18   # xlabel / ylabel (matches scatter script)
FONT_TICK      = 17   # tick label size (matches scatter script)
FONT_LEGEND    = 14
FONT_LEG_TITLE = 16
FONT_SUPTITLE  = 15

COLOR_F16  = "#4878CF"   # steel blue
COLOR_2BIT = "#E8604C"   # tomato-red

ALPHA_HIST = 0.75
GRID_ALPHA = 0.3
LW_VLINE   = 1.8


# ── Matplotlib setup ─────────────────────────────────────────────────────────
def _setup_matplotlib() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]

    plt.rcParams.update(
        {
            "font.family":        "sans-serif",
            "font.sans-serif":    sans,
            "font.size":          FONT_BASE,
            "font.weight":        "bold",
            "axes.labelsize":     FONT_AXIS,
            "axes.labelweight":   "bold",
            "axes.titlesize":     FONT_PANEL,
            "axes.titleweight":   "bold",
            "xtick.labelsize":    FONT_BASE,
            "ytick.labelsize":    FONT_BASE,
            "axes.unicode_minus": False,
        }
    )


def _style_ax(ax: plt.Axes, *, grid: bool = True) -> None:
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    if grid:
        ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8, ls="--")
        ax.minorticks_off()


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    print(f"Loading {EMB_CSV.name} ...")
    df_emb  = pd.read_csv(EMB_CSV)
    print(f"Loading {RES_F16_CSV.name} ...")
    df_r16  = pd.read_csv(RES_F16_CSV)
    print(f"Loading {RES_2BIT_CSV.name} ...")
    df_r2   = pd.read_csv(RES_2BIT_CSV)

    # emb = centroid + residual_f16  →  centroid = emb - residual_f16
    emb          = df_emb[DIM_COLS].to_numpy(dtype=np.float32)
    residual_f16 = df_r16[DIM_COLS].to_numpy(dtype=np.float32)
    residual_2bit = df_r2[DIM_COLS].to_numpy(dtype=np.float32)

    centroid = emb - residual_f16
    v_f16    = emb                      # = centroid + residual_f16
    v_2bit   = centroid + residual_2bit

    n = len(v_f16)
    print(f"  {n:,} tokens loaded")
    return v_f16, v_2bit, n


# ── Figure 5: ||c + r||² distribution ────────────────────────────────────────
def plot_fig5(v_f16: np.ndarray, v_2bit: np.ndarray, n: int) -> None:
    norm_2bit = (v_2bit ** 2).sum(axis=1)
    mean_2bit = float(norm_2bit.mean())

    x_min = float(np.floor(norm_2bit.min() * 10) / 10)
    x_max = float(np.ceil(max(norm_2bit.max(), 1.05) * 10) / 10)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        norm_2bit, bins=100, density=True,
        alpha=ALPHA_HIST, color=COLOR_2BIT,
        label=f"2-bit  (mean = {mean_2bit:.4f})",
        edgecolor="none",
    )
    ax.axvline(
        1.0, color=COLOR_F16, linestyle="--", linewidth=LW_VLINE,
        label="float16 = 1.0 (exact)",
    )
    ax.axvline(
        mean_2bit, color=COLOR_2BIT, linestyle="--", linewidth=LW_VLINE,
        label=f"2-bit mean = {mean_2bit:.4f}",
    )

    # x축 눈금: 0.1 간격
    x_ticks = np.arange(round(x_min, 1), round(x_max + 0.05, 1), 0.1)
    ax.set_xticks(x_ticks)
    ax.set_xlim(x_min - 0.02, x_max + 0.02)

    # y축 눈금: 1.0 간격
    y_max = ax.get_ylim()[1]
    y_ceil = int(np.ceil(y_max))
    ax.set_yticks(range(0, y_ceil + 1))

    ax.set_xlabel("||centroid + residual||²", fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND,
              title="Quantization\nComparison", title_fontsize=FONT_LEG_TITLE)

    ax.set_title(
        f"Figure 5: 2-bit Reconstruction Norm  ||c + r||²  [N = {n:,} tokens]\n"
        "float16 = exactly 1.0  ·  2-bit deviates from unit sphere",
        fontsize=FONT_PANEL,
        fontweight="bold",
    )
    _style_ax(ax)
    fig.tight_layout()
    save_figure(fig, "fig5_norm_distribution")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_matplotlib()
    v_f16, v_2bit, n = load_data()

    print("Plotting Figure 5 ...")
    plot_fig5(v_f16, v_2bit, n)

    print(f"Done. Output in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
