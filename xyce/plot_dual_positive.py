"""
Dual 3T1C  —  new circuit: V4>0, V5>0
  M0: drain = -V5,  BSN = V4 - V5 + Vth  →  Vgs_M0 = V4 + Vth  (Vth cancelled)
  M3: drain = -V4,  BSN = V5 - V4 + Vth  →  Vgs_M3 = V5 + Vth

Reuses existing CSVs via coordinate transform:
  M0: x_new = V4_new - V5_new = V4_old + V5_old  (V5_old = -V5_new)
  M3: x_new = V4_new - V5_new = -(V5_old + V4_old)  (V4_old = -V4_new)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7

def I_single(vgs, v0=V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

clip = 1e-23

# ── M0 data: existing CSVs (V5_old<0) ────────────────────────────────────────
# New circuit: V5_new = -V5_old > 0, V4_new = V4_old (unchanged sweep)
# x = V4_new - V5_new = V4_old + V5_old  (same column values, just relabelled)
v5_new_list = [1.0, 0.5]
v5_old_list = [-1.0, -0.5]
colors_left = ["steelblue", "cornflowerblue"]

left_data = {}
for v5_new, v5_old in zip(v5_new_list, v5_old_list):
    tag = f"{v5_old:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    df = pd.read_csv(f"tmp_v4v5_{tag}.cir.csv")
    df.columns = ["v4", "vsn", "vmid", "iv5"]
    df["im_A"] = -df["iv5"]
    df["x"] = df["v4"] + v5_old  # = V4_new - V5_new
    left_data[v5_new] = df
    print(f"M0  V5_new={v5_new:+.1f}V loaded, x range [{df['x'].min():.1f}, {df['x'].max():.1f}]")

# ── M3 data: existing right CSVs (V4_old<0) ──────────────────────────────────
# New circuit: V4_new = -V4_old > 0, V5_new = -V5_old (sweep sign flipped)
# x = V4_new - V5_new = (-V4_old) - (-V5_old) = -(V4_old + V5_old) = -(df["v5"] + v4_old)
v4_new_list = [1.0, 0.5]
v4_old_list = [-1.0, -0.5]
colors_right = ["tomato", "salmon"]

right_data = {}
for v4_new, v4_old in zip(v4_new_list, v4_old_list):
    tag = f"{v4_old:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    fname = f"tmp_right_{tag}.cir.csv"
    df = pd.read_csv(fname)
    df.columns = ["v5", "vsn", "vmid", "iv4"]
    df["im_A"] = -df["iv4"]
    df["x"] = -(df["v5"] + v4_old)  # = V4_new - V5_new
    right_data[v4_new] = df
    print(f"M3  V4_new={v4_new:+.1f}V loaded, x range [{df['x'].min():.1f}, {df['x'].max():.1f}]")

# ── Python model ──────────────────────────────────────────────────────────────
# Vgs_M0 = V4_new + V0  →  I_M0 = I_single(V4_new + V0) * 1e-9
#   V4_new = x + V5_new  →  vgs = x + V5_new + V0
# Vgs_M3 = V5_new + V0  →  I_M3 = I_single(V5_new + V0) * 1e-9
#   V5_new = V4_new - x  →  vgs = V4_new_fixed - x + V0

x_arr = np.linspace(-2, 2, 500)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    "Dual 3T1C  (V4>0, V5>0)  —  |I| vs (V4−V5)\n"
    "M0: drain=−V5  |  M3: drain=−V4  |  Vth cancelled  |  output always ≥0",
    fontsize=10)

# M0 curves
for (v5_new, df), color in zip(left_data.items(), colors_left):
    # Xyce
    ax.plot(df["x"], np.abs(df["im_A"]).clip(lower=clip),
            color=color, lw=2, ls="-", marker="o", markevery=10, markersize=4,
            zorder=3, label=f"M0  V5={v5_new:+.1f}V (Xyce)")
    # Python unidirectional: Vgs = V4_new + V0 = (x + V5_new) + V0
    i_py = I_single(x_arr + v5_new + V0) * 1e-9
    ax.plot(x_arr, np.clip(i_py, clip, None),
            color=color, lw=1.2, ls="--", alpha=0.65, zorder=2)
    # kink position: Vgs = VSAT → x + V5_new + V0 = VSAT → x = VSAT - V5_new - V0
    xk = VSAT - v5_new - V0
    if -2 <= xk <= 2:
        ax.axvline(xk, color=color, lw=0.7, ls=":", alpha=0.4, zorder=1)

# M3 curves
for (v4_new, df), color in zip(right_data.items(), colors_right):
    # Xyce
    ax.plot(df["x"], np.abs(df["im_A"]).clip(lower=clip),
            color=color, lw=2, ls="-", marker="s", markevery=10, markersize=4,
            zorder=3, label=f"M3  V4={v4_new:+.1f}V (Xyce)")
    # Python: Vgs_M3 = V5_new + V0 = (V4_new - x) + V0
    i_py = I_single(v4_new - x_arr + V0) * 1e-9
    ax.plot(x_arr, np.clip(i_py, clip, None),
            color=color, lw=1.2, ls="--", alpha=0.65, zorder=2)
    # kink: V4_new - x + V0 = VSAT → x = V4_new + V0 - VSAT
    xk = v4_new + V0 - VSAT
    if -2 <= xk <= 2:
        ax.axvline(xk, color=color, lw=0.7, ls=":", alpha=0.4, zorder=1)

ax.axvline(0, color="gray", lw=1.2, ls=":", alpha=0.6, label="V4 = V5")
ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-5)
ax.set_xlim(-2, 2)
ax.set_xlabel("V4 − V5  [V]", fontsize=11)
ax.set_ylabel("|I|  [A]", fontsize=11)
ax.set_title("Log |I|  [A]  —  Xyce solid+marker  |  Python unidirectional dashed", fontsize=9)
ax.grid(True, ls="--", alpha=0.4)

legend_handles = [
    Line2D([0],[0], color="k", lw=2, ls="-", marker="o", markersize=4, label="Xyce (circle)"),
    Line2D([0],[0], color="k", lw=2, ls="-", marker="s", markersize=4, label="Xyce (square)"),
    Line2D([0],[0], color="k", lw=1.2, ls="--", label="Python unidirectional"),
    Line2D([0],[0], color="k", lw=0.7, ls=":", label="kink (Vgs=VSAT)"),
] + [
    Line2D([0],[0], color=c, lw=2, marker="o", markersize=4, label=f"M0  V5={v:+.1f}V")
    for v, c in zip(v5_new_list, colors_left)
] + [
    Line2D([0],[0], color=c, lw=2, marker="s", markersize=4, label=f"M3  V4={v:+.1f}V")
    for v, c in zip(v4_new_list, colors_right)
]
ax.legend(handles=legend_handles, fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig("dual_3t1c_positive_log.png", dpi=150)
print("Saved: dual_3t1c_positive_log.png")
