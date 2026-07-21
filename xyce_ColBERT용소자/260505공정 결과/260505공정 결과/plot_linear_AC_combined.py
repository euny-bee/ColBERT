"""
Option A (With Vth Compensation) vs Option C (No Compensation) — linear scale only.

Reads die4 CSV, fits logistic model, plots both linear-scale panels side by side.
Run from this directory: python plot_linear_AC_combined.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
from scipy.optimize import curve_fit

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR
CSV_FILE   = DATA_DIR / "IGZO_TR [die4_ovl20_R0(20) ; 5_6_2026 6_08_20 PM].csv"

# ── Constants ─────────────────────────────────────────────────────────────────
FONT_BASE     = 13
FONT_AXIS     = 14
FONT_PANEL    = 16
FONT_SUPTITLE = 18

VTH_ORIG  = 0.151          # V
VDS       = 1.7            # V
SHIFTS    = (0.5, 3.0)
COLORS    = ["darkorange", "#FF5722"]
LSTYLES   = ["--", "--"]
GRID_ALPHA = 0.3
LW_CURVE   = 1.2


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
    print(f"  → {out}")


# ── Data loading & fitting ────────────────────────────────────────────────────
def parse_csv(path: Path):
    gate, id_ = [], []
    in_data = sweep_done = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("DataName"):
                if in_data:
                    sweep_done = True
                in_data = True
                continue
            if sweep_done:
                break
            if not in_data or not line.startswith("DataValue"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                gate.append(float(parts[1]))
                id_.append(float(parts[3]))
            except ValueError:
                continue
    return np.array(gate), np.array(id_)


def logistic(vgs, L, K, V0, B):
    return B + L / (1 + np.exp(-K * (vgs - V0)))


def fit_params(path: Path) -> Dict:
    gate, id_ = parse_csv(path)
    id_abs = np.abs(id_)
    valid  = id_abs > 0
    vgs_f  = gate[valid]
    log_ids = np.log10(id_abs[valid])

    noise  = gate <= -2.0
    ioff   = max(np.median(id_abs[noise]) if np.any(noise) else 1e-11, 1e-14)
    Ion_log  = np.log10(np.max(id_abs[valid]))
    Ioff_log = np.log10(ioff)
    mid_log  = (Ion_log + Ioff_log) / 2
    V0_init  = vgs_f[np.argmin(np.abs(log_ids - mid_log))]

    popt, _ = curve_fit(
        logistic, vgs_f, log_ids,
        p0=[Ion_log - Ioff_log, 5.0, V0_init, Ioff_log],
        bounds=([1, 0.5, -3, -20], [20, 30, 3, -5]),
        maxfev=30000,
    )
    L, K, Vth, B = popt
    print(f"  Fit: L={L:.4f}, K={K:.4f}, Vth={Vth:.4f} V, B={B:.4f}")
    return dict(L=L, K=K, Vth=Vth, B=B)


def calc_ids(vgs_arr: np.ndarray, params: Dict, vds: float = 1.7, vds_meas: float = 1.0) -> np.ndarray:
    L, K, Vth, B = params["L"], params["K"], params["Vth"], params["B"]
    ids = 10 ** (B + L / (1 + np.exp(-K * (vgs_arr - Vth))))
    VoD = vgs_arr - Vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > vds_meas, Vo * vds_meas - vds_meas ** 2 / 2, Vo ** 2 / 2)
    It = np.where(Vo > vds,      Vo * vds      - vds      ** 2 / 2, Vo ** 2 / 2)
    factor[on] = np.where(Im > 0, It / Im, 1.0)
    return ids * factor


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_linear_combined(params: Dict) -> None:
    Vth_orig = params["Vth"]
    Vth_vals = [Vth_orig + s for s in SHIFTS]
    labels_C = [f"No comp  ΔVth = {s:+.1f} V" for s in SHIFTS]
    label_A0 = "Vth Comp (all shifts overlap)"

    V2_A = np.arange(-1.0, 1.0 + 1e-9, 0.05)
    x_A  = 2 * V2_A
    V2_C = np.arange(-2.0, 2.0 + 1e-9, 0.05)
    x_C  = V2_C

    fig, ax = plt.subplots(figsize=(5.6, 4.3))

    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    # Option A — 원본 Vth 기준 (루프 외부)
    p0 = dict(params, Vth=Vth_orig)
    VGS_A = np.abs(x_A) + Vth_orig
    ids_A = calc_ids(VGS_A, p0, vds=VDS) * 1e6
    ax.plot(x_A, ids_A, color="#2196F3", lw=LW_CURVE, ls="-", label=label_A0)

    for i, (vth, lbl_c, col) in enumerate(zip(Vth_vals, labels_C, COLORS)):
        p = dict(params, Vth=vth)

        # Option C — 전체 표시
        VGS_C = np.abs(x_C)
        ids_C = calc_ids(VGS_C, p, vds=VDS) * 1e6
        ax.plot(x_C, ids_C, color=col, lw=LW_CURVE, ls=LSTYLES[i],
                label=lbl_c)

    ax.set_xlabel("V1-V2  [V]")
    ax.set_ylabel("IDS  [uA]")
    ax.set_xlim(-2.2, 2.2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.legend(fontsize=FONT_BASE - 2, loc="upper center", ncol=1,
              title=r"$\mathrm{V_{th,M0} = V_{th,M3}}$", title_fontsize=FONT_BASE - 1)
    _style_ax(ax)

    fig.tight_layout()
    save_figure(fig, "plot_linear_AC_combined")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_matplotlib()
    print(f"Loading {CSV_FILE.name} ...")
    params = fit_params(CSV_FILE)
    print("Plotting ...")
    plot_linear_combined(params)
    print(f"Done. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
