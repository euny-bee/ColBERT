"""
Dual 3T1C Phase 3 (L2 search) -- Xyce vs Python overlay
die4_ovl20_R0 (260505 process)

Loads:
  circuit_phase3_m0_die4.cir.csv  -- M0 Xyce result
  circuit_phase3_m3_die4.cir.csv  -- M3 Xyce result

Outputs:
  dual_phase3_xyce_die4.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

_available = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _available:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

DIR = os.path.dirname(os.path.abspath(__file__))

# ── die4 parameters ───────────────────────────────────────────────────────────
L = 6.1120; K = 2.7319; V0 = 0.1510; B = -10.7139; VSAT = 2.88; SLOPE = 4.0332e-5
V2 = -1.0; neg_V2 = -V2; VDD = 1.0
clip = 1e-23

def I_single(vgs):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE

def I_M0(v1):
    return np.maximum(I_single(v1 + V0 - neg_V2) - I_single(v1 + V0 - neg_V2 - VDD), 0)

def I_M3(v1):
    return np.maximum(I_single(neg_V2 + V0 - v1) - I_single(V0 - v1), 0)

# ── Load Xyce CSV ─────────────────────────────────────────────────────────────
csv_m0 = os.path.join(DIR, "circuit_phase3_m0_die4.cir.csv")
csv_m3 = os.path.join(DIR, "circuit_phase3_m3_die4.cir.csv")

xyce_available = os.path.isfile(csv_m0) and os.path.isfile(csv_m3)

if xyce_available:
    df_m0 = pd.read_csv(csv_m0, comment="*", header=0)
    df_m3 = pd.read_csv(csv_m3, comment="*", header=0)
    df_m0.columns = ["v1", "vsn0", "vml", "i_vdd"]
    df_m3.columns = ["v1", "vsn3", "vml", "i_vdd"]
    v1_xyce  = df_m0["v1"].values
    im0_xyce = -df_m0["i_vdd"].values   # I(VVDD) is negative when sourcing
    im3_xyce = -df_m3["i_vdd"].values
    isum_xyce = im0_xyce + im3_xyce

# ── Python model ─────────────────────────────────────────────────────────────
v1_py = np.linspace(-1, 2, 300)
im0_py   = I_M0(v1_py)
im3_py   = I_M3(v1_py)
isum_py  = im0_py + im3_py

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    f"Dual 3T1C  Phase 3 (L2 search)  --  die4_ovl20_R0  (260505)\n"
    f"V2={V2}V stored  (-V2={neg_V2}V Phase2 source),  V1 swept  |  Vth={V0}V",
    fontsize=10)

# Python model (dashed)
ax.plot(v1_py, np.clip(im0_py,   clip, None), color="steelblue", lw=1.2, ls="--", alpha=0.6)
ax.plot(v1_py, np.clip(im3_py,   clip, None), color="tomato",    lw=1.2, ls="--", alpha=0.6)
ax.plot(v1_py, np.clip(isum_py,  clip, None), color="purple",    lw=1.5, ls="--", alpha=0.6)

# Xyce (solid, if available)
if xyce_available:
    ax.plot(v1_xyce, np.clip(im0_xyce,  clip, None), color="steelblue", lw=2, ls="-",
            marker="o", markevery=15, markersize=4,
            label=f"M0  Xyce  (V1 > -V2={neg_V2}V)")
    ax.plot(v1_xyce, np.clip(im3_xyce,  clip, None), color="tomato",    lw=2, ls="-",
            marker="s", markevery=15, markersize=4,
            label=f"M3  Xyce  (-V2 > V1)")
    ax.plot(v1_xyce, np.clip(isum_xyce, clip, None), color="purple",    lw=2.5, ls="-",
            label="I_M0 + I_M3  Xyce  total")
else:
    print("Xyce CSV not found -- Python model only")

ax.axvline(neg_V2, color="gray", lw=1.2, ls=":", alpha=0.7,
           label=f"V1 = -V2 = {neg_V2}V  (crossover)")

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-3)
ax.set_xlim(-1, 2)
ax.set_xlabel("V1  [V]", fontsize=11)
ax.set_ylabel("|I|  [A]", fontsize=11)
ax.set_title(
    f"Log |I| [A]  vs  V1  --  Xyce solid / Python dashed\n"
    f"M0 gate = V1 - V2 + Vth = V1 + {-V2:.1f} + {V0}  |  "
    f"M3 gate = V2 - V1 + Vth = -{-V2:.1f} - V1 + {V0}",
    fontsize=8.5)
ax.grid(True, ls="--", alpha=0.4)

legend_handles = [
    Line2D([0],[0], color="steelblue", lw=2, marker="o", markersize=4,
           label=f"M0  (V1 > -V2={neg_V2}V 일때 ON)"),
    Line2D([0],[0], color="tomato",    lw=2, marker="s", markersize=4,
           label=f"M3  (-V2 > V1 일때 ON)"),
    Line2D([0],[0], color="purple",    lw=2.5,
           label="I_M0 + I_M3  total  |V1 - V2|"),
    Line2D([0],[0], color="k",         lw=1.2, ls="--", alpha=0.6,
           label="Python bidirectional model"),
    Line2D([0],[0], color="gray",      lw=1.2, ls=":",
           label=f"V1 = -V2 = {neg_V2}V"),
]
ax.legend(handles=legend_handles, fontsize=8.5)

plt.tight_layout()
out = os.path.join(DIR, "dual_phase3_xyce_die4.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")
if not xyce_available:
    print(f"  (Xyce CSV not found; Python-only plot)")
    print(f"  Run Xyce: Xyce circuit_phase3_m0_die4.cir && Xyce circuit_phase3_m3_die4.cir")
