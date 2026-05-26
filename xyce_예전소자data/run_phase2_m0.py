"""
Phase 2 single M0 simulation
  Phase 1 stored: VSN = Vth - V5 = 0.11 - 1.0 = -0.89V
  Phase 2: V4 bootstraps -> VSN_final = V4 - 0.89
           V̄5 node = +1V (was -1V in Phase 1)
           V1 = 0V
  Circuit: V̄5(1V) -> M0 -> mid -> M2 -> V1(0V)
"""
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7
V5 = 1.0   # fixed V5

def I_single(vgs, v0=V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9  # nA

offset = V0 - V5  # = 0.1102 - 1.0 = -0.8898  (BSN = V4 + offset)

cir = f"""* Phase 2 - single M0: V5={V5}V, V̄5=+{V5}V, V4 swept -1~1V
* VSN_final = V4 + (Vth - V5) = V4 {offset:+.4f}
.INCLUDE igzo_smooth_k5.sub
VV1  v1n  0  DC 0.0
VV2  v2n  0  DC 3.0
VV3  v3n  0  DC -3.0
VV4  v4n  0  DC 0.0
V5B  v5bn 0  DC {V5}       ; V̄5 = +{V5}V  (Phase 2, was -{V5}V in Phase 1)
BSN  sn   0  V={{V(v4n) + {offset:.4f}}}
XM2  v1n  v2n  mid  IGZO_SMOOTH_K5
XM1  mid  v3n  sn   IGZO_SMOOTH_K5
XM0  mid  sn   v5bn IGZO_SMOOTH_K5
RMID mid  0  10T
.DC VV4 -1 2 0.02
.PRINT DC FORMAT=CSV V(v4n) V(sn) V(mid) I(V5B)
.END
"""

cir_file = "phase2_m0.cir"
csv_file = cir_file + ".csv"

with open(cir_file, "w") as f:
    f.write(cir)

ret = subprocess.run(["Xyce", cir_file], capture_output=True, text=True)
if ret.returncode != 0:
    print("[ERROR]", ret.stderr[-500:])
    exit(1)

df = pd.read_csv(csv_file)
df.columns = ["v4", "vsn", "vmid", "i_v5b"]
# current through M0: flows from V̄5(1V) → M0 → mid, so I(V5B) is positive when current leaves 1V source
# I_BIGZO is drain→source; source=v5bn(1V), drain=mid(0V) → I_BIGZO < 0
# Physical current flows v5bn→mid (1V→0V), so I_M0 = -I(V5B)
df["im0_A"] = -df["i_v5b"]

print("V4    VSN      Vmid     I_M0(A)")
print("-" * 45)
for _, r in df.iloc[::10].iterrows():
    print(f"{r['v4']:5.2f}  {r['vsn']:7.4f}  {r['vmid']:7.4f}  {r['im0_A']:+.4e}")

# Python model: mid ≈ 0V (M2 clamps), V̄5 = 1V
# Bidirectional: Vgs = VSN - mid ≈ VSN = V4 + offset
#                Vgd = VSN - V̄5 = V4 + offset - 1
x_arr = np.linspace(-1, 2, 300)
vgs_py = x_arr + offset          # = V4 - 0.89 (= V4 + Vth - V5)
vgd_py = x_arr + offset - V5     # = V4 - 0.89 - 1.0 = V4 - 1.89
i_bidir_nA = I_single(vgs_py) - I_single(vgd_py)
i_bidir_A  = i_bidir_nA * 1e-9

clip = 1e-23

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(
    f"Single M0 — Phase 2\n"
    f"V5={V5}V  |  VSN = V4 {offset:+.4f}  |  V̄5 = +{V5}V, V1 = 0V",
    fontsize=10)

# Xyce
ax.plot(df["v4"], np.abs(df["im0_A"]).clip(lower=clip),
        color="steelblue", lw=2, ls="-", marker="o", markevery=5, markersize=5,
        label="Xyce (Phase 2)")

# Python bidirectional
ax.plot(x_arr, np.abs(i_bidir_A).clip(clip),
        color="steelblue", lw=1.2, ls="--", alpha=0.7,
        label="Python bidirectional")

# kink: Vgs = VSAT → V4 + offset = VSAT → V4 = VSAT - offset
v4_kink = VSAT - offset
if -1 <= v4_kink <= 1:
    ax.axvline(v4_kink, color="steelblue", lw=0.8, ls=":", alpha=0.6,
               label=f"kink V4={v4_kink:.2f}V")

ax.axvline(V5, color="gray", lw=1, ls=":", alpha=0.6, label=f"V4=V5={V5}V")

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-5)
ax.set_xlim(-1, 2)
ax.set_xlabel("V4  [V]", fontsize=11)
ax.set_ylabel("|I_M0|  [A]", fontsize=11)
ax.set_title("Log |I_M0|  [A]  vs  V4", fontsize=10)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("phase2_m0_log.png", dpi=150)
print("Saved: phase2_m0_log.png")
