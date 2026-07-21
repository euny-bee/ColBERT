"""
Residual quantization figures: fig1 value distribution, fig2a per-token norm.

Reads residuals_float32.csv and residuals_2bit.csv from centroidset 만들기/.
Saves fig1 and fig2a as PNG (300 dpi) in the same directory as this script.
Run from 논문 figure/: python plot_residual_figures.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR.parent / "centroidset 만들기"
OUTPUT_DIR  = SCRIPT_DIR

F16_CSV  = DATA_DIR / "residuals_float32.csv"
BIT2_CSV = DATA_DIR / "residuals_2bit.csv"

DIM      = 128
DIM_COLS = [f"dim_{i}" for i in range(DIM)]

# ── Style constants ──────────────────────────────────────────────────────────
FONT_BASE     = 10
FONT_AXIS     = 11
FONT_PANEL    = 12
FONT_SUPTITLE = 14

COLOR_F16  = "#4878CF"   # steel blue
COLOR_2BIT = "#E8604C"   # tomato-red

ALPHA_HIST  = 0.6
ALPHA_VLINE = 0.65
GRID_ALPHA  = 0.3
LW_VLINE    = 1.0

MAX_SAMPLE_FIG1 = 500_000


# ── Matplotlib setup ─────────────────────────────────────────────────────────
def _setup_matplotlib() -> None:
    # rebuild font cache if Arial not found yet
    available = {f.name for f in fm.fontManager.ttflist}
    sans = ["Arial", "Helvetica", "DejaVu Sans"]
    sans = [f for f in sans if f in available] or ["DejaVu Sans"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "font.size": FONT_BASE,
            "axes.labelsize": FONT_AXIS,
            "axes.titlesize": FONT_PANEL,
            "xtick.labelsize": FONT_BASE,
            "ytick.labelsize": FONT_BASE,
        }
    )


def _style_ax(ax: plt.Axes, *, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    if grid:
        ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8)
        ax.minorticks_off()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data() -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (residual_f16, residual_2bit, n_tokens) as float32 arrays (N, 128)."""
    print(f"Loading {F16_CSV.name} ...")
    df_f16  = pd.read_csv(F16_CSV)
    print(f"Loading {BIT2_CSV.name} ...")
    df_2bit = pd.read_csv(BIT2_CSV)

    residual_f16  = df_f16[DIM_COLS].to_numpy(dtype=np.float32)
    residual_2bit = df_2bit[DIM_COLS].to_numpy(dtype=np.float32)

    n = len(residual_f16)
    print(f"  {n:,} tokens, {DIM} dims each")
    return residual_f16, residual_2bit, n


# ── Figure 1: Residual value distribution ────────────────────────────────────
def plot_fig1(residual_f16: np.ndarray, residual_2bit: np.ndarray, n: int) -> None:
    flat_f16  = residual_f16.flatten()
    flat_2bit = residual_2bit.flatten()

    rng = np.random.default_rng(42)
    if len(flat_f16) > MAX_SAMPLE_FIG1:
        idx      = rng.choice(len(flat_f16), MAX_SAMPLE_FIG1, replace=False)
        flat_f16  = flat_f16[idx]
        flat_2bit = flat_2bit[idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(flat_f16,  bins=300, density=True, alpha=ALPHA_HIST,
            color=COLOR_F16,  label="float16 (continuous)", edgecolor="none")
    ax.hist(flat_2bit, bins=20,  density=True, alpha=ALPHA_HIST + 0.1,
            color=COLOR_2BIT, label="2-bit (4 discrete values)", edgecolor="none")

    unique_vals = np.unique(flat_2bit)
    for v in unique_vals:
        ax.axvline(v, color=COLOR_2BIT, linestyle="--",
                   linewidth=LW_VLINE, alpha=ALPHA_VLINE)

    ax.set_xlabel("Residual value")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.35, 0.35)
    ax.legend(fontsize=FONT_BASE)
    fig.suptitle(
        f"Figure 1: Residual Value Distribution\n"
        f"float16 (continuous) vs 2-bit (4 discrete values)  [N={n:,} tokens]",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )
    _style_ax(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig1_residual_distribution_new")


# ── Figure 2a: Per-token residual norm ||r||² ─────────────────────────────────
def plot_fig2a(residual_f16: np.ndarray, residual_2bit: np.ndarray, n: int) -> None:
    norm_f16  = (residual_f16  ** 2).sum(axis=1)
    norm_2bit = (residual_2bit ** 2).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(norm_f16,  bins=100, density=True, alpha=ALPHA_HIST,
            color=COLOR_F16,
            label=f"float16  (mean={norm_f16.mean():.4f})",
            edgecolor="none")
    ax.hist(norm_2bit, bins=100, density=True, alpha=ALPHA_HIST,
            color=COLOR_2BIT,
            label=f"2-bit    (mean={norm_2bit.mean():.4f})",
            edgecolor="none")

    ax.set_xlabel("||r||²  per token")
    ax.set_ylabel("Density")
    ax.legend(fontsize=FONT_BASE)
    fig.suptitle(
        f"Figure 2a: Per-token Residual Norm  ||r||²\n"
        f"float16 vs 2-bit  [N={n:,} tokens]",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )
    _style_ax(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig2a_residual_norm_new")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_matplotlib()
    residual_f16, residual_2bit, n = load_data()

    print("Plotting Figure 1 ...")
    plot_fig1(residual_f16, residual_2bit, n)

    print("Plotting Figure 2a ...")
    plot_fig2a(residual_f16, residual_2bit, n)

    print(f"Done. Outputs in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
