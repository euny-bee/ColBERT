"""
Combined M0 + M3 Phase 2 plot
  M0: I ∝ (V4-V5)*VDD  — turns on when V4 > V5
  M3: I ∝ (V5-V4)*VDD  — turns on when V5 > V4
  Both share V1. x-axis = V4, V5=1V fixed.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7
V5 = 1.0; VDD = 1.0
clip = 1e-23

def I_single(vgs, v0=V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

# ── Load Xyce data ────────────────────────────────────────────────────────────
m0 = pd.read_csv("phase2_m0.cir.csv")
m0.columns = ["v4", "vsn", "vmid", "i_raw"]
m0["I_A"] = -m0["i_raw"]   # physical current (positive)

m3 = pd.read_csv("phase2_m3.cir.csv")
m3.columns = ["v4", "vsn", "vmid", "i_raw"]
m3["I_A"] = -m3["i_raw"]

# Align on same V4 grid (both already have same sweep)
v4 = m0["v4"].values
I_m0 = m0["I_A"].values
I_m3 = m3["I_A"].values
I_sum = I_m0 + I_m3

# ── Python model ──────────────────────────────────────────────────────────────
x = np.linspace(-1, 2, 400)

# M0: Vgd_eff = V4+Vth-V5 = x+V0-V5,  Vgs_eff = x+V0-V5-VDD
vgd_m0 = x + V0 - V5        # = V4 - 0.89  (wrt 0V terminal)
vgs_m0 = x + V0 - V5 - VDD  # = V4 - 1.89  (wrt VDD terminal)
i_m0_py = np.maximum(I_single(vgd_m0) - I_single(vgs_m0), 0) * 1e-9

# M3: Vgd_eff = V5+Vth-V4 = V5+V0-x,  Vgs_eff = V5+V0-x-VDD
vgd_m3 = V5 + V0 - x
vgs_m3 = V5 + V0 - x - VDD
i_m3_py = np.maximum(I_single(vgd_m3) - I_single(vgs_m3), 0) * 1e-9

i_sum_py = i_m0_py + i_m3_py

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle(
    "Dual 3T1C  Phase 2  (V5=1V fixed, V4 swept)\n"
    r"M0: $I \propto (V_4-V_5)\cdot VDD$   |   M3: $I \propto (V_5-V_4)\cdot VDD$   |   both → V1",
    fontsize=10)

# Xyce M0
ax.plot(v4, np.abs(I_m0).clip(min=clip),
        color="steelblue", lw=2, ls="-", marker="o", markevery=5, markersize=4,
        label="M0  Xyce (V4>V5 active)")

# Xyce M3
ax.plot(v4, np.abs(I_m3).clip(min=clip),
        color="tomato", lw=2, ls="-", marker="s", markevery=5, markersize=4,
        label="M3  Xyce (V5>V4 active)")

# Xyce SUM
ax.plot(v4, np.abs(I_sum).clip(min=clip),
        color="purple", lw=2.5, ls="-",
        label="I_M0 + I_M3  (total to V1)")

# Python M0
ax.plot(x, np.clip(i_m0_py, clip, None),
        color="steelblue", lw=1, ls="--", alpha=0.55)

# Python M3
ax.plot(x, np.clip(i_m3_py, clip, None),
        color="tomato", lw=1, ls="--", alpha=0.55)

# Python SUM
ax.plot(x, np.clip(i_sum_py, clip, None),
        color="purple", lw=1, ls="--", alpha=0.55)

ax.axvline(V5, color="gray", lw=1.2, ls=":", alpha=0.7, label=f"V4 = V5 = {V5}V")

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-5)
ax.set_xlim(-1, 2)
ax.set_xlabel("V4  [V]", fontsize=11)
ax.set_ylabel("|I|  [A]", fontsize=11)
ax.set_title("Log |I|  [A]  —  Xyce solid  |  Python dashed", fontsize=9)
ax.grid(True, ls="--", alpha=0.4)

legend_handles = [
    Line2D([0],[0], color="steelblue", lw=2, marker="o", markersize=4, label="M0 (active when V4>V5)"),
    Line2D([0],[0], color="tomato",    lw=2, marker="s", markersize=4, label="M3 (active when V5>V4)"),
    Line2D([0],[0], color="purple",    lw=2.5,            label="I_M0 + I_M3  total (-> V1)"),
    Line2D([0],[0], color="k",         lw=1, ls="--",     label="Python bidirectional"),
    Line2D([0],[0], color="gray",      lw=1, ls=":",      label=f"V4 = V5 = {V5}V"),
]
ax.legend(handles=legend_handles, fontsize=9)

plt.tight_layout()
plt.savefig("dual_phase2_log.png", dpi=150)
print("Saved: dual_phase2_log.png")
