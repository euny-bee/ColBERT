"""
Python-model versions of the two dual Phase 2 graphs.
Outputs:
  dual_phase2_log_pyt.png          — M0, M3, total  (Vth=0.11V)
  dual_phase2_vthshift_log_pyt.png — original vs +0.5V Vth shift comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 6.9526; K = 5.0889; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7
V0_orig  = 0.1102
V0_shift = 0.6102
V5 = 1.0; VDD = 1.0
clip = 1e-23

def I_single(vgs, v0=V0_orig):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L/(1+np.exp(-K*(v-v0)))) + np.maximum(0, vgs-VSAT)*SLOPE) * 1e9  # nA

def I_M0(v4_arr, v0):
    """M0 Phase 2: source=VDD(1V), drain=0V, gate=V4+(v0-V5)"""
    vgd = v4_arr + v0 - V5        # wrt 0V (mid) terminal  = V4 + v0 - 1
    vgs = v4_arr + v0 - V5 - VDD  # wrt VDD(1V) terminal   = V4 + v0 - 2
    return np.maximum(I_single(vgd, v0) - I_single(vgs, v0), 0) * 1e-9  # A

def I_M3(v4_arr, v0):
    """M3 Phase 2: source=VDD(1V), drain=0V, gate=V5+(v0-V4)"""
    vgd = V5 + v0 - v4_arr        # wrt 0V terminal  = 1 + v0 - V4
    vgs = V5 + v0 - v4_arr - VDD  # wrt VDD terminal = v0 - V4
    return np.maximum(I_single(vgd, v0) - I_single(vgs, v0), 0) * 1e-9  # A

v4 = np.linspace(-1, 2, 300)

im0o = I_M0(v4, V0_orig);   im3o = I_M3(v4, V0_orig);   isuo = im0o + im3o
im0s = I_M0(v4, V0_shift);  im3s = I_M3(v4, V0_shift);  isus = im0s + im3s

# ══════════════════════════════════════════════════════════════════════════════
# Graph 1: dual_phase2_log_pyt.png  (same layout as dual_phase2_log.png)
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(9, 5))
fig1.suptitle(
    "Dual 3T1C  Phase 2  (V5=1V fixed, V4 swept)  —  Python model\n"
    r"M0: $I \propto (V_4-V_5)\cdot VDD$   |   M3: $I \propto (V_5-V_4)\cdot VDD$   |   both → V1",
    fontsize=10)

ax1.plot(v4, np.clip(im0o, clip, None),
         color="steelblue", lw=2, ls="-", marker="o", markevery=15, markersize=4,
         label="M0  (active when V4>V5)")
ax1.plot(v4, np.clip(im3o, clip, None),
         color="tomato", lw=2, ls="-", marker="s", markevery=15, markersize=4,
         label="M3  (active when V5>V4)")
ax1.plot(v4, np.clip(isuo, clip, None),
         color="purple", lw=2.5, ls="-",
         label="I_M0 + I_M3  total (-> V1)")

ax1.axvline(V5, color="gray", lw=1.2, ls=":", alpha=0.7, label=f"V4 = V5 = {V5}V")

ax1.set_yscale("log"); ax1.set_ylim(1e-14, 1e-5); ax1.set_xlim(-1, 2)
ax1.set_xlabel("V4  [V]", fontsize=11); ax1.set_ylabel("|I|  [A]", fontsize=11)
ax1.set_title("Log |I|  [A]  —  Python bidirectional model", fontsize=9)
ax1.grid(True, ls="--", alpha=0.4); ax1.legend(fontsize=9)
plt.tight_layout()
fig1.savefig("dual_phase2_log_pyt.png", dpi=150)
print("Saved: dual_phase2_log_pyt.png")

# ══════════════════════════════════════════════════════════════════════════════
# Graph 2: dual_phase2_vthshift_log_pyt.png  (same layout as vthshift graph)
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(9, 5))
fig2.suptitle(
    "Dual 3T1C  Phase 2  (V5=1V fixed, V4 swept)  —  Python model\n"
    f"Dashed: Vth={V0_orig:.4f}V (original)   |   "
    f"Solid+marker: Vth={V0_shift:.4f}V (+0.5V shift)",
    fontsize=10)

# Original (dashed)
ax2.plot(v4, np.clip(im0o, clip, None), color="steelblue", lw=1.5, ls="--", alpha=0.7,
         label=f"M0  Vth={V0_orig:.2f}V")
ax2.plot(v4, np.clip(im3o, clip, None), color="tomato",    lw=1.5, ls="--", alpha=0.7,
         label=f"M3  Vth={V0_orig:.2f}V")
ax2.plot(v4, np.clip(isuo, clip, None), color="purple",    lw=1.5, ls="--", alpha=0.7,
         label=f"Total  Vth={V0_orig:.2f}V")

# Shifted (solid + markers)
ax2.plot(v4, np.clip(im0s, clip, None), color="steelblue", lw=2.5, ls="-",
         marker="o", markevery=15, markersize=4,
         label=f"M0  Vth={V0_shift:.2f}V (+0.5V)")
ax2.plot(v4, np.clip(im3s, clip, None), color="tomato",    lw=2.5, ls="-",
         marker="s", markevery=15, markersize=4,
         label=f"M3  Vth={V0_shift:.2f}V (+0.5V)")
ax2.plot(v4, np.clip(isus, clip, None), color="purple",    lw=3,   ls="-",
         label=f"Total  Vth={V0_shift:.2f}V (+0.5V)")

# kink lines (shifted only)
v4_kink_m0s = VSAT + V5 - V0_shift   # = 1.677V
v4_kink_m3s = V5 + V0_shift - VSAT   # = 0.323V
ax2.axvline(v4_kink_m0s, color="steelblue", lw=0.9, ls=":", alpha=0.6,
            label=f"M0 kink V4={v4_kink_m0s:.3f}V")
ax2.axvline(v4_kink_m3s, color="tomato",    lw=0.9, ls=":", alpha=0.6,
            label=f"M3 kink V4={v4_kink_m3s:.3f}V")
ax2.axvline(V5, color="gray", lw=1.2, ls=":", alpha=0.7,
            label=f"V4=V5={V5}V (turn-on)")

ax2.set_yscale("log"); ax2.set_ylim(1e-14, 1e-5); ax2.set_xlim(-1, 2)
ax2.set_xlabel("V4  [V]", fontsize=11); ax2.set_ylabel("|I|  [A]", fontsize=11)
ax2.set_title("Dashed: original Vth   |   Solid+marker: Vth +0.5V shift", fontsize=9)
ax2.grid(True, ls="--", alpha=0.4); ax2.legend(fontsize=8, ncol=2)
plt.tight_layout()
fig2.savefig("dual_phase2_vthshift_log_pyt.png", dpi=150)
print("Saved: dual_phase2_vthshift_log_pyt.png")
