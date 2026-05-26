"""
Dual 3T1C Phase 2  —  Vth shift comparison
  Original : V0 = 0.1102V
  Shifted  : V0 = 0.6102V (+0.5V)
  V5=1V fixed, V4 swept -1~2V
  Plots M0, M3, total for both cases
"""
import subprocess
import pandas as pd
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
    return (10**(B + L/(1+np.exp(-K*(v-v0)))) + np.maximum(0, vgs-VSAT)*SLOPE) * 1e9

def run_xyce(cir_file, cir_txt):
    with open(cir_file, "w") as f:
        f.write(cir_txt)
    ret = subprocess.run(["Xyce", cir_file], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[ERROR] {cir_file}:", ret.stderr[-300:])
        return None
    df = pd.read_csv(cir_file + ".csv")
    return df

# ── M3 original ───────────────────────────────────────────────────────────────
bsn_m3_orig = V5 + V0_orig  # = 1.1102
cir_m3_orig = f"""* M3 Phase2 Vth={V0_orig}
.INCLUDE igzo_smooth_k5.sub
VV1  v1n 0 DC 0.0
VV2  v2n 0 DC 3.0
VV3  v3n 0 DC -3.0
VV4  v4n 0 DC 0.0
V4B  v4bn 0 DC {VDD}
BSN  sn  0 V={{{bsn_m3_orig:.4f} - V(v4n)}}
XM5  v1n v2n mid IGZO_SMOOTH_K5
XM4  mid v3n sn  IGZO_SMOOTH_K5
XM3  mid sn  v4bn IGZO_SMOOTH_K5
RMID mid 0 10T
.DC VV4 -1 2 0.02
.PRINT DC FORMAT=CSV V(v4n) V(sn) V(mid) I(V4B)
.END
"""

# ── M3 shifted ────────────────────────────────────────────────────────────────
bsn_m3_shift = V5 + V0_shift  # = 1.6102
cir_m3_shift = f"""* M3 Phase2 Vth={V0_shift}
.INCLUDE igzo_smooth_k5.sub
VV1  v1n 0 DC 0.0
VV2  v2n 0 DC 3.0
VV3  v3n 0 DC -3.0
VV4  v4n 0 DC 0.0
V4B  v4bn 0 DC {VDD}
BSN  sn  0 V={{{bsn_m3_shift:.4f} - V(v4n)}}
XM5  v1n v2n mid IGZO_SMOOTH_K5
XM4  mid v3n sn  IGZO_SMOOTH_K5
XM3  mid sn  v4bn IGZO_SMOOTH_K5 PARAMS: V0={V0_shift:.4f}
RMID mid 0 10T
.DC VV4 -1 2 0.02
.PRINT DC FORMAT=CSV V(v4n) V(sn) V(mid) I(V4B)
.END
"""

print("Running M3 original...")
df_m3o = run_xyce("phase2_m3_orig.cir", cir_m3_orig)
print("Running M3 shifted...")
df_m3s = run_xyce("phase2_m3_shift.cir", cir_m3_shift)

for df in [df_m3o, df_m3s]:
    df.columns = ["v4", "vsn", "vmid", "i_raw"]
    df["I_A"] = -df["i_raw"]

# M0: load from existing files
df_m0o = pd.read_csv("phase2_m0.cir.csv")
df_m0o.columns = ["v4", "vsn", "vmid", "i_raw"]
df_m0o["I_A"] = -df_m0o["i_raw"]

df_m0s = pd.read_csv("phase2_m0_vthshift.cir.csv")
df_m0s.columns = ["v4", "vsn", "vmid", "i_raw"]
df_m0s["I_A"] = -df_m0s["i_raw"]

v4 = df_m0o["v4"].values

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    "Dual 3T1C  Phase 2  (V5=1V fixed, V4 swept)\n"
    f"Blue/Red: Vth={V0_orig:.4f}V (original)   |   "
    f"Orange/Crimson: Vth={V0_shift:.4f}V (+0.5V shift)",
    fontsize=10)

# ── Original (solid, darker) ──────────────────────────────────────────────────
I_m0o = df_m0o["I_A"].values
I_m3o = df_m3o["I_A"].values
I_sumo = I_m0o + I_m3o

ax.plot(v4, np.abs(I_m0o).clip(clip),
        color="steelblue", lw=1.5, ls="--", alpha=0.7,
        label=f"M0  Vth={V0_orig:.2f}V")
ax.plot(v4, np.abs(I_m3o).clip(clip),
        color="tomato", lw=1.5, ls="--", alpha=0.7,
        label=f"M3  Vth={V0_orig:.2f}V")
ax.plot(v4, np.abs(I_sumo).clip(clip),
        color="purple", lw=1.5, ls="--", alpha=0.7,
        label=f"Total  Vth={V0_orig:.2f}V")

# ── Shifted (solid + markers, prominent) ─────────────────────────────────────
I_m0s = df_m0s["I_A"].values
I_m3s = df_m3s["I_A"].values
I_sums = I_m0s + I_m3s

ax.plot(v4, np.abs(I_m0s).clip(clip),
        color="steelblue", lw=2.5, ls="-", marker="o", markevery=5, markersize=4,
        label=f"M0  Vth={V0_shift:.2f}V (+0.5V)")
ax.plot(v4, np.abs(I_m3s).clip(clip),
        color="tomato", lw=2.5, ls="-", marker="s", markevery=5, markersize=4,
        label=f"M3  Vth={V0_shift:.2f}V (+0.5V)")
ax.plot(v4, np.abs(I_sums).clip(clip),
        color="purple", lw=3, ls="-",
        label=f"Total  Vth={V0_shift:.2f}V (+0.5V)")

# kink lines for shifted
v4_kink_m0s = VSAT + V5 - V0_shift  # = 1.677V
v4_kink_m3s = V5 + V0_shift - VSAT  # = 0.323V
ax.axvline(v4_kink_m0s, color="steelblue", lw=0.9, ls=":", alpha=0.6,
           label=f"M0 kink V4={v4_kink_m0s:.3f}V")
ax.axvline(v4_kink_m3s, color="tomato",    lw=0.9, ls=":", alpha=0.6,
           label=f"M3 kink V4={v4_kink_m3s:.3f}V")
ax.axvline(V5, color="gray", lw=1.2, ls=":", alpha=0.7,
           label=f"V4=V5={V5}V (turn-on)")

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-5)
ax.set_xlim(-1, 2)
ax.set_xlabel("V4  [V]", fontsize=11)
ax.set_ylabel("|I|  [A]", fontsize=11)
ax.set_title("Dashed: original Vth   |   Solid+marker: Vth +0.5V shift", fontsize=9)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig("dual_phase2_vthshift_log.png", dpi=150)
print("Saved: dual_phase2_vthshift_log.png")
