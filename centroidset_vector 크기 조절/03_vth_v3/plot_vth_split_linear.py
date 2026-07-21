"""
Vth Comp / No Comp split into two separate figures (linear scale).
Baseline Vth_orig = 0.03V; M0 shifted by +0.5V, M3 shifted by +1.5V
  -> Vth_M0 = 0.03 + 0.5 = 0.53V, Vth_M3 = 0.03 + 1.5 = 1.53V
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent

FONT_BASE = 16
FONT_AXIS = 17
LW_MAIN   = 2.2

C_VTHCOMP = '#2196F3'   # blue
C_NOCOMP  = '#F57C00'   # orange

L_p      =  6.112020
K_p      =  2.731949
B_p      = -10.713911
VSAT     =  2.88
SLOPE    =  4.0332e-5
VDD      =  1.7

Vth_orig   = 0.03
Shift_M0   = 0.5
Shift_M3   = 1.5
Vth_M0     = Vth_orig + Shift_M0   # 0.53
Vth_M3     = Vth_orig + Shift_M3   # 1.53
VSAT_OD    = VSAT - Vth_orig


def _setup_matplotlib() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": FONT_BASE,
        "font.weight": "bold", "axes.labelsize": FONT_AXIS, "axes.labelweight": "bold",
        "xtick.labelsize": FONT_BASE, "ytick.labelsize": FONT_BASE,
    })


def I_single(vgs, vth):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    vsat_abs = vth + VSAT_OD
    v = np.minimum(vgs, vsat_abs)
    return (10 ** (B_p + L_p / (1 + np.exp(-K_p * (v - vth))))
            + np.maximum(0.0, vgs - vsat_abs) * SLOPE)


def cell_optA(d, vth_m0, vth_m3):
    d = np.asarray(d, dtype=float)
    vgd_m0 = d + vth_m0
    I_m0 = np.where(d >= 0, np.maximum(I_single(vgd_m0, vth_m0) - I_single(vgd_m0 - VDD, vth_m0), 0.0), 0.0)
    vgd_m3 = -d + vth_m3
    I_m3 = np.where(d < 0, np.maximum(I_single(vgd_m3, vth_m3) - I_single(vgd_m3 - VDD, vth_m3), 0.0), 0.0)
    clip = 1e-14
    return np.clip(I_m0 + I_m3, clip, None) * 1e9


def cell_optC(d, vth_m0, vth_m3):
    d = np.asarray(d, dtype=float)
    vgd_m0 = d
    I_m0 = np.where(d >= 0, np.maximum(I_single(vgd_m0, vth_m0) - I_single(vgd_m0 - VDD, vth_m0), 0.0), 0.0)
    vgd_m3 = -d
    I_m3 = np.where(d < 0, np.maximum(I_single(vgd_m3, vth_m3) - I_single(vgd_m3 - VDD, vth_m3), 0.0), 0.0)
    clip = 1e-14
    return np.clip(I_m0 + I_m3, clip, None) * 1e9


def _style_ax(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.8)
    ax.minorticks_off()
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("V1 - V2  [V]")
    ax.set_ylabel("IDS  [uA]")
    ax.set_xlim(-2.1, 2.1)


def main():
    _setup_matplotlib()
    d_arr = np.linspace(-2.0, 2.0, 1000)

    Itot_A = cell_optA(d_arr, Vth_M0, Vth_M3)
    Itot_C = cell_optC(d_arr, Vth_M0, Vth_M3)

    def save_legend(handles, labels, stem):
        fig_leg, ax_leg = plt.subplots(figsize=(3, 1.2))
        ax_leg.axis("off")
        ax_leg.legend(handles, labels, fontsize=FONT_BASE - 2, loc="center",
                      frameon=True, edgecolor="black")
        fig_leg.tight_layout()
        out = OUTPUT_DIR / f"{stem}_legend.png"
        fig_leg.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig_leg)
        print(f"Saved: {out}")

    # ── Panel 1: Vth Compensation only ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(d_arr, Itot_A * 1e-3, color=C_VTHCOMP, lw=LW_MAIN, ls="-", label="Vth comp")
    _style_ax(ax)
    handles1, labels1 = ax.get_legend_handles_labels()
    fig.tight_layout()
    out1 = OUTPUT_DIR / "plot_vth_split_linear_vthcomp.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")
    save_legend(handles1, labels1, "plot_vth_split_linear_vthcomp")

    # ── Panel 2: No Compensation (solid) + Vth Comp (dashed) overlay ──────────
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(d_arr, Itot_A * 1e-3, color=C_VTHCOMP, lw=LW_MAIN, ls="--", label="Vth comp")
    ax.plot(d_arr, Itot_C * 1e-3, color=C_NOCOMP, lw=LW_MAIN, ls="-",
            label=f"No comp\nM0 {Shift_M0:.1f}V shift\nM3 {Shift_M3:.1f}V shift")

    # dead zone shading: [-Vth_M3, +Vth_M0] = [-1.53, +0.53]
    ax.axvspan(-Vth_M3, Vth_M0, color="violet", alpha=0.18, zorder=0)

    _style_ax(ax)
    handles2, labels2 = ax.get_legend_handles_labels()
    fig.tight_layout()
    out2 = OUTPUT_DIR / "plot_vth_split_linear_nocomp.png"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")
    save_legend(handles2, labels2, "plot_vth_split_linear_nocomp")

    print(f"Vth_M0={Vth_M0:.2f}V (orig {Vth_orig}+shift {Shift_M0}), "
          f"Vth_M3={Vth_M3:.2f}V (orig {Vth_orig}+shift {Shift_M3})")
    print(f"No Comp dead zone: [-{Vth_M3:.2f}, +{Vth_M0:.2f}]")


if __name__ == "__main__":
    main()
