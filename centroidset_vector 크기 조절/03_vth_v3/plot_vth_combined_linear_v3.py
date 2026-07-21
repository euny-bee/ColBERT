"""
v3_fixed: Option A (mismatch + individual comp) vs Option C (mismatch, no comp) — linear scale.

Panels 2 and 4 from the second row of plot_vth_mismatch_v3_fixed.png, overlaid on one axes.
Model: fixed I_single with overdrive-based VSAT clip (from pipeline_vth_v3.py).
Mismatch: Vth_M0=0.1V, Vth_M3=2.5V (v3 range [0,3]V, std=0.90 representative).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent

# ── Style constants (plot_pbs_array.py 기준, 1.3x bold) ───────────────────────
FONT_BASE     = 16
FONT_AXIS     = 17
FONT_PANEL    = 19
FONT_SUPTITLE = 22
GRID_ALPHA    = 0.3
LW_MAIN       = 2.0
LW_SUB        = 1.4

C_M0_A  = '#2196F3'   # blue       — Vth Comp M0
C_M3_A  = '#F44336'   # red        — Vth Comp M3
C_TOT_A = '#2196F3'   # Material Blue 500 — Vth Comp Total
C_M0_C  = '#90CAF9'   # light blue — No Comp M0
C_M3_C  = '#EF9A9A'   # light red  — No Comp M3
C_TOT_C = '#F57C00'   # orange      — No Comp Total

# ── Device parameters (die4, v3_fixed model) ──────────────────────────────────
L_p      =  6.112020
K_p      =  2.731949
B_p      = -10.713911
VSAT     =  2.88
SLOPE    =  4.0332e-5
VDD      =  1.7
Vth_orig =  0.151045
VSAT_OD  =  VSAT - Vth_orig   # 2.729V — overdrive-based VSAT constant

Vth_M0 = 0.5
Vth_M3 = 3.0


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
            "mathtext.fontset": "dejavusans",
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


# ── Device model (v3_fixed: overdrive-based VSAT clip) ────────────────────────
def I_single(vgs: np.ndarray, vth: float) -> np.ndarray:
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    vsat_abs = vth + VSAT_OD
    v = np.minimum(vgs, vsat_abs)
    return (10 ** (B_p + L_p / (1 + np.exp(-K_p * (v - vth))))
            + np.maximum(0.0, vgs - vsat_abs) * SLOPE)


def cell_optA(d, vth_m0, vth_m3):
    """Option A: individual Vth compensation (stored = actual Vth per device)."""
    d = np.asarray(d, dtype=float)
    vgd_m0 = d + vth_m0
    I_m0 = np.where(d >= 0,
                    np.maximum(I_single(vgd_m0, vth_m0) - I_single(vgd_m0 - VDD, vth_m0), 0.0),
                    0.0)
    vgd_m3 = -d + vth_m3
    I_m3 = np.where(d < 0,
                    np.maximum(I_single(vgd_m3, vth_m3) - I_single(vgd_m3 - VDD, vth_m3), 0.0),
                    0.0)
    clip = 1e-14
    return (np.clip(I_m0, clip, None) * 1e9,
            np.clip(I_m3, clip, None) * 1e9,
            np.clip(I_m0 + I_m3, clip, None) * 1e9)


def cell_optC(d, vth_m0, vth_m3):
    """Option C: no compensation (stored = 0)."""
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
    Im0_A, Im3_A, Itot_A = cell_optA(d_arr, Vth_M0, Vth_M3)
    # Panel 4: Option C mismatch, no comp
    Im0_C, Im3_C, Itot_C = cell_optC(d_arr, Vth_M0, Vth_M3)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax.plot(d_arr, Itot_A * 1e-3, color=C_TOT_A, lw=LW_MAIN, ls="-",
            label="Vth Comp")
    ax.plot(d_arr, Itot_C * 1e-3, color=C_TOT_C, lw=LW_MAIN, ls="--",
            label=f"No Comp\nM0 {Vth_M0:.1f}V shift\nM3 {Vth_M3:.1f}V shift")

    ax.set_xlabel("V1 - V2  [V]")
    ax.set_ylabel("IDS  [uA]")
    ax.set_xlim(-2.1, 2.1)
    _style_ax(ax)

    fig.tight_layout()
    save_figure(fig, "plot_vth_combined_linear_v3")

    # ── Legend only figure ────────────────────────────────────────────────────
    fig_leg, ax_leg = plt.subplots(figsize=(3, 1.2))
    ax_leg.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax_leg.legend(handles, labels, fontsize=FONT_BASE - 2, loc="center", ncol=1,
                        frameon=True,
                        title=r"$\mathrm{V_{th,M0} \neq V_{th,M3}}$",
                        title_fontsize=FONT_BASE - 1)
    fig_leg.tight_layout()
    save_figure(fig_leg, "plot_vth_combined_linear_v3_legend")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_matplotlib()
    print("Plotting ...")
    plot_combined()
    print(f"Done. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
