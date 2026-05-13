"""
Phase 2 M0 with Vth +0.5V shift
  V0_orig  = 0.1102V  -> BSN = V4 + (0.1102 - 1.0) = V4 - 0.8898
  V0_shift = 0.6102V  -> BSN = V4 + (0.6102 - 1.0) = V4 - 0.3898
  Vth cancellation still holds: turn-on at V4 = V5 = 1V for both
  Difference: kink position (Vgs=VSAT) shifts -> different saturation onset
"""
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 6.9526; K = 5.0889; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7
V0_orig  = 0.1102
V0_shift = V0_orig + 0.5   # = 0.6102
V5 = 1.0; VDD = 1.0
clip = 1e-23

def I_single(vgs, v0=V0_orig):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

# ── Run Xyce: Vth-shifted M0 ──────────────────────────────────────────────────
offset_shift = V0_shift - V5   # = 0.6102 - 1.0 = -0.3898

cir = f"""* Phase 2 M0  Vth+0.5V shift: V0={V0_shift:.4f}V
* BSN = V4 + (V0_shift - V5) = V4 {offset_shift:+.4f}  (Vth-cancelled turn-on still at V4=V5=1V)
.INCLUDE igzo_smooth_k5.sub
VV1  v1n  0  DC 0.0
VV2  v2n  0  DC 3.0
VV3  v3n  0  DC -3.0
VV4  v4n  0  DC 0.0
V5B  v5bn 0  DC {VDD}
BSN  sn   0  V={{V(v4n) + {offset_shift:.4f}}}
XM2  v1n  v2n  mid  IGZO_SMOOTH_K5
XM1  mid  v3n  sn   IGZO_SMOOTH_K5
XM0  mid  sn   v5bn IGZO_SMOOTH_K5 PARAMS: V0={V0_shift:.4f}
RMID mid  0  10T
.DC VV4 -1 2 0.02
.PRINT DC FORMAT=CSV V(v4n) V(sn) V(mid) I(V5B)
.END
"""

cir_file = "phase2_m0_vthshift.cir"
csv_file  = cir_file + ".csv"
with open(cir_file, "w") as f:
    f.write(cir)

ret = subprocess.run(["Xyce", cir_file], capture_output=True, text=True)
if ret.returncode != 0:
    print("[ERROR]", ret.stderr[-400:])
    exit(1)

df_shift = pd.read_csv(csv_file)
df_shift.columns = ["v4", "vsn", "vmid", "i_raw"]
df_shift["I_A"] = -df_shift["i_raw"]

# ── Load original (from previous run) ────────────────────────────────────────
df_orig = pd.read_csv("phase2_m0.cir.csv")
df_orig.columns = ["v4", "vsn", "vmid", "i_raw"]
df_orig["I_A"] = -df_orig["i_raw"]

# ── Python model comparison ───────────────────────────────────────────────────
x = np.linspace(-1, 2, 400)

def py_m0(v4_arr, v0):
    vgd = v4_arr + v0 - V5       # = V4 + V0 - V5  (wrt 0V terminal)
    vgs = v4_arr + v0 - V5 - VDD # = V4 + V0 - V5 - 1  (wrt VDD terminal)
    return np.maximum(I_single(vgd, v0) - I_single(vgs, v0), 0) * 1e-9

i_orig_py  = py_m0(x, V0_orig)
i_shift_py = py_m0(x, V0_shift)

# kink positions: Vgd_eff = VSAT -> V4 = VSAT + V5 - V0
v4_kink_orig  = VSAT + V5 - V0_orig   # = 2.177V (outside sweep)
v4_kink_shift = VSAT + V5 - V0_shift  # = 1.677V (inside sweep!)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle(
    "Single M0  Phase 2  —  Vth shift comparison\n"
    f"V5={V5}V fixed, V4 swept  |  V0_orig={V0_orig:.4f}V  vs  V0_shift={V0_shift:.4f}V  (+0.5V)",
    fontsize=10)

# Original
ax.plot(df_orig["v4"], np.abs(df_orig["I_A"]).clip(lower=clip),
        color="steelblue", lw=2, ls="-", marker="o", markevery=5, markersize=4,
        label=f"Xyce  Vth={V0_orig:.4f}V (original)")
ax.plot(x, np.clip(i_orig_py, clip, None),
        color="steelblue", lw=1, ls="--", alpha=0.6)

# Shifted
ax.plot(df_shift["v4"], np.abs(df_shift["I_A"]).clip(lower=clip),
        color="tomato", lw=2, ls="-", marker="s", markevery=5, markersize=4,
        label=f"Xyce  Vth={V0_shift:.4f}V (+0.5V shift)")
ax.plot(x, np.clip(i_shift_py, clip, None),
        color="tomato", lw=1, ls="--", alpha=0.6)

# kink lines
if -1 <= v4_kink_shift <= 2:
    ax.axvline(v4_kink_shift, color="tomato", lw=1, ls=":", alpha=0.7,
               label=f"kink (shifted) V4={v4_kink_shift:.3f}V")
if -1 <= v4_kink_orig <= 2:
    ax.axvline(v4_kink_orig, color="steelblue", lw=1, ls=":", alpha=0.7,
               label=f"kink (orig) V4={v4_kink_orig:.3f}V")

ax.axvline(V5, color="gray", lw=1.2, ls=":", alpha=0.7, label=f"V4=V5={V5}V (turn-on, both)")

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-5)
ax.set_xlim(-1, 2)
ax.set_xlabel("V4  [V]", fontsize=11)
ax.set_ylabel("|I_M0|  [A]", fontsize=11)
ax.set_title("Log |I_M0|  —  Xyce solid  |  Python dashed", fontsize=9)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("phase2_m0_vthshift_log.png", dpi=150)
print("Saved: phase2_m0_vthshift_log.png")
