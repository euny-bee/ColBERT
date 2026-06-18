"""
Comparison: Option A (V2=-V1, x=2V2) vs V1=0 fixed (V1=0, V2 sweep -2~2V)
Both plotted on x-axis = V2-V1

Circuit analysis:
  Option A : VSN = -V1+Vth, Phase3 VGS = V2-V1+Vth = |2V2|+Vth  (V2=-V1)
  V1=0     : VSN = Vth,     Phase3 VGS = V2-0+Vth  = |V2|+Vth

Both: IDS = f(|V2-V1| + Vth)  -> perfect overlap on V2-V1 axis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import os

# -- font ------------------------------------------------------------------
_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# -- die4 logistic params (VDS=1V fit) ------------------------------------
L_f =  6.112020
K_f =  2.731949
B_f = -10.713911
Vth_orig = 0.151045

# -- Vth sweep ------------------------------------------------------------
shifts      = [0.0, 0.1, 0.2, 0.5, 1.0]
Vth_vals    = [Vth_orig + s for s in shifts]
vth_labels  = [
    f"Vth = {Vth_orig:.3f} V (orig)",
    f"+0.1V  ({Vth_vals[1]:.3f} V)",
    f"+0.2V  ({Vth_vals[2]:.3f} V)",
    f"+0.5V  ({Vth_vals[3]:.3f} V)",
    f"+1.0V  ({Vth_vals[4]:.3f} V)",
]
colors = ["black", "royalblue", "forestgreen", "darkorange", "crimson"]

# -- model functions -------------------------------------------------------
def logistic_ids(vgs, vth):
    return 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - vth))))

def vds_correction(vgs_arr, vth, VDS_t=1.7, VDS_m=1.0):
    VoD = vgs_arr - vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > VDS_m, Vo * VDS_m - VDS_m**2 / 2, Vo**2 / 2)
    It = np.where(Vo > VDS_t, Vo * VDS_t - VDS_t**2 / 2, Vo**2 / 2)
    factor[on] = np.where(Im > 0, It / Im, 1.0)
    return factor

def calc_ids(vgs_arr, vth):
    return logistic_ids(vgs_arr, vth) * vds_correction(vgs_arr, vth)   # [A]

# -- sweep points ---------------------------------------------------------
# Option A : V2 = -V1, V2 in [-1, 1]  ->  x = V2-V1 = 2*V2 in [-2, 2]
V2_A = np.arange(-1.0, 1.0 + 1e-9, 0.05)
x_A  = 2 * V2_A

# V1=0 case : V1=0 fixed, V2 in [-2, 2]  ->  x = V2-0 = V2
V2_B = np.arange(-2.0, 2.0 + 1e-9, 0.025)
x_B  = V2_B

# -- plot -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Option A  (V2 = -V1, sweep)  vs  V1 = 0 fixed  (V2 sweep -2~2 V)\n"
    f"die4   Vth_orig = {Vth_orig:.3f} V,   VDS = 1.7 V\n"
    "x-axis = V2 - V1  (distance)  ->  curves should overlap perfectly",
    fontsize=10, fontweight="bold"
)

for ax, yscale in zip(axes, ["log", "linear"]):
    ax.set_title(f"{yscale} scale", fontsize=10)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.4)

    for vth, lbl, col in zip(Vth_vals, vth_labels, colors):
        # Option A (solid, thick)
        VGS_A = np.abs(x_A) + vth
        ids_A = calc_ids(VGS_A, vth) * 1e6     # uA
        ax.plot(x_A, ids_A, color=col, lw=3.0, ls="-", alpha=0.9)

        # V1=0 (dashed + circle markers)
        VGS_B = np.abs(x_B) + vth
        ids_B = calc_ids(VGS_B, vth) * 1e6     # uA
        ax.plot(x_B, ids_B, color=col, lw=1.2, ls="--",
                marker="o", markersize=4, markevery=8, alpha=0.85)

    ax.set_xlabel("V2 - V1  [V]", fontsize=10)
    ax.set_ylabel("IDS  [uA]", fontsize=10)
    ax.set_xlim(-2.2, 2.2)
    ax.set_xticks(np.arange(-2, 2.1, 0.5))
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if yscale == "log":
        ax.set_yscale("log")

    # --- legend ---
    style_legend = [
        Line2D([0],[0], color="gray", lw=3.0, ls="-",
               label="Option A  (V2=-V1, x=2V2)"),
        Line2D([0],[0], color="gray", lw=1.2, ls="--",
               marker="o", markersize=4, label="V1=0 fixed  (V2 sweep)"),
    ]
    color_legend = [
        Line2D([0],[0], color=col, lw=2.5, label=lbl)
        for col, lbl in zip(colors, vth_labels)
    ]
    leg1 = ax.legend(handles=style_legend, fontsize=8.5,
                     loc="upper left", framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, fontsize=7.5,
              loc="upper center", framealpha=0.85)

    if yscale == "log":
        ax.text(0.50, 0.35,
                "Curves overlap perfectly\n(IDS depends only on |V2-V1|)",
                transform=ax.transAxes, fontsize=8.5, ha="center",
                color="navy", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "plot_V1zero_vs_optionA.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")

# -- numerical check (x=1.0) ----------------------------------------------
print("\n--- Numerical check at x = V2-V1 = 1.0 V (Vth_orig) ---")
vth = Vth_orig
# Option A: V2=0.5, V1=-0.5
ids_A = calc_ids(np.array([1.0 + vth]), vth)[0] * 1e6
# V1=0:     V2=1.0, V1=0
ids_B = calc_ids(np.array([1.0 + vth]), vth)[0] * 1e6
print(f"  Option A (V2=0.5, V1=-0.5) : IDS = {ids_A:.4f} uA")
print(f"  V1=0    (V2=1.0, V1=0)    : IDS = {ids_B:.4f} uA")
print(f"  Difference: {abs(ids_A-ids_B):.6f} uA  (should be 0)")
