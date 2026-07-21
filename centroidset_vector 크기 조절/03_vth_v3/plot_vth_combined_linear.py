"""
Vth mismatch combined linear plot.

2nd panel (Option A mismatch + individual comp) and
4th panel (Option C mismatch, no comp) overlaid on one axes — linear scale only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent

# ── Style constants (plot_pbs_array.py 기준) ──────────────────────────────────
FONT_BASE     = 13
FONT_AXIS     = 14
FONT_PANEL    = 16
FONT_SUPTITLE = 18
GRID_ALPHA    = 0.3
LW_MAIN       = 2.0
LW_SUB        = 1.4

C_M0_A  = '#2196F3'   # blue  — Vth Comp M0
C_M3_A  = '#F44336'   # red   — Vth Comp M3
C_TOT_A = '#4CAF50'   # green — Vth Comp Total
C_M0_C  = '#90CAF9'   # light blue  — No Comp M0
C_M3_C  = '#EF9A9A'   # light red   — No Comp M3
C_TOT_C = '#A5D6A7'   # light green — No Comp Total

# ── Device parameters (die4) ──────────────────────────────────────────────────
L_p = 6.1120; K_p = 2.7319; B_p = -10.7139; VSAT = 2.88; SLOPE = 4.0332e-5
VDD = 1.7

Vth_orig = 0.151
Vth_M0   = 0.1
Vth_M3   = 0.5


# ── Matplotlib setup ──────────────────────────────────────────────────────────
def _setup_matplotlib() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "font.size": FONT_BASE,
            "font.weight": "bold",
            "axes.labelsize": FONT_AXIS,
            "axes.labelweight": "bold",
            "axes.titlesize": FONT_PANEL,
            "axes.titleweight": "bold",
            "xtick.labelsize": FONT_BASE,
            "ytick.labelsize": FONT_BASE,
        }
    )


def _style_ax(ax: plt.Axes) -> None:
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8)
    ax.minorticks_off()


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ── Device model ──────────────────────────────────────────────────────────────
def I_single(vgs: np.ndarray, vth: float) -> np.ndarray:
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v   = np.minimum(vgs, VSAT)
    return 10 ** (B_p + L_p / (1 + np.exp(-K_p * (v - vth)))) + np.maximum(0, vgs - VSAT) * SLOPE


def cell_optA(d, vth_m0, vth_m3, stored_m0, stored_m3):
    d = np.asarray(d, dtype=float)
    vgd_m0 = d + stored_m0
    I_m0 = np.where(d >= 0,
                    np.maximum(I_single(vgd_m0, vth_m0) - I_single(vgd_m0 - VDD, vth_m0), 0.0),
                    0.0)
    vgd_m3 = -d + stored_m3
    I_m3 = np.where(d < 0,
                    np.maximum(I_single(vgd_m3, vth_m3) - I_single(vgd_m3 - VDD, vth_m3), 0.0),
                    0.0)
    clip = 1e-14
    return (np.clip(I_m0, clip, None) * 1e9,
            np.clip(I_m3, clip, None) * 1e9,
            np.clip(I_m0 + I_m3, clip, None) * 1e9)


def cell_optC(d, vth_m0, vth_m3):
    d = np.asarray(d, dtype=float)
    vgd_m0 = d
    I_m0 = np.where(d >= 0,
                    np.maximum(I_single(vgd_m0, vth_m0) - I_single(vgd_m0 - VDD, vth_m0), 0.0),
                    0.0)
    vgd_m3 = -d
    I_m3 = np.where(d < 0,
                    np.maximum(I_single(vgd_m3, vth_m3) - I_single(vgd_m3 - VDD, vth_m3), 0.0),
                    0.0)
    clip = 1e-14
    return (np.clip(I_m0, clip, None) * 1e9,
            np.clip(I_m3, clip, None) * 1e9,
            np.clip(I_m0 + I_m3, clip, None) * 1e9)


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_combined() -> None:
    d_arr = np.linspace(-2.0, 2.0, 1000)

    # Panel 2: Option A mismatch + individual comp
    Im0_A, Im3_A, Itot_A = cell_optA(d_arr, Vth_M0, Vth_M3, Vth_M0, Vth_M3)
    # Panel 4: Option C mismatch, no comp
    Im0_C, Im3_C, Itot_C = cell_optC(d_arr, Vth_M0, Vth_M3)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    # Option A (Vth Comp)
    ax.plot(d_arr, Im0_A,  color=C_M0_A,  lw=LW_SUB, ls="--",
            label=f"Vth Comp,  M0  {Vth_M0:.1f}V shift")
    ax.plot(d_arr, Im3_A,  color=C_M3_A,  lw=LW_SUB, ls=":",
            label=f"Vth Comp,  M3  {Vth_M3:.1f}V shift")
    ax.plot(d_arr, Itot_A, color=C_TOT_A, lw=LW_MAIN, ls="-",
            label="Vth Comp,  Total")

    # Option C (No Comp)
    ax.plot(d_arr, Im0_C,  color=C_M0_C,  lw=LW_SUB, ls="--",
            label=f"No Comp,  M0  {Vth_M0:.1f}V shift")
    ax.plot(d_arr, Im3_C,  color=C_M3_C,  lw=LW_SUB, ls=":",
            label=f"No Comp,  M3  {Vth_M3:.1f}V shift")
    ax.plot(d_arr, Itot_C, color=C_TOT_C, lw=LW_MAIN, ls="-",
            label="No Comp,  Total")

    ax.set_xlabel("d = V1 - V2  [V]")
    ax.set_ylabel("IDS  [nA]")
    ax.set_xlim(-2.1, 2.1)
    ax.legend(fontsize=FONT_BASE - 2, loc="upper center", ncol=1)
    _style_ax(ax)

    fig.suptitle(
        f"Vth Comp vs No Comp  (M0: {Vth_M0}V shift,  M3: {Vth_M3}V shift)  [linear]",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "plot_vth_combined_linear")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_matplotlib()
    print("Plotting ...")
    plot_combined()
    print(f"Done. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
