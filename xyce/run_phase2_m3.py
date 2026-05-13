"""
Phase 2 single M3 simulation  (symmetric to M0)
  Phase 1 stored: VSN = Vth - V4
  Phase 2: V5 bootstraps -> VSN_final = V5 + (Vth - V4) = (V5-V4) + Vth
           V̄4 node = +1V (VDD, was -V4 in Phase 1)
           V1 = 0V
  Current turns on when V5 > V4  (mirror image of M0)
"""
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7
V5 = 1.0   # fixed
VDD = 1.0  # V̄4 in Phase 2

def I_single(vgs, v0=V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9  # nA

# BSN = V5 + Vth - V4 = 1.1102 - V(v4n)
bsn_const = V5 + V0  # = 1.1102

cir = f"""* Phase 2 - single M3: V5={V5}V fixed, V4 swept -1~2V
* VSN_final = (V5-V4) + Vth = {bsn_const:.4f} - V4
.INCLUDE igzo_smooth_k5.sub
VV1  v1n  0  DC 0.0
VV2  v2n  0  DC 3.0
VV3  v3n  0  DC -3.0
VV4  v4n  0  DC 0.0
V4B  v4bn 0  DC {VDD}       ; V̄4 = +{VDD}V  (Phase 2 VDD)
BSN  sn   0  V={{{bsn_const:.4f} - V(v4n)}}
XM5  v1n  v2n  mid  IGZO_SMOOTH_K5
XM4  mid  v3n  sn   IGZO_SMOOTH_K5
XM3  mid  sn   v4bn IGZO_SMOOTH_K5
RMID mid  0  10T
.DC VV4 -1 2 0.02
.PRINT DC FORMAT=CSV V(v4n) V(sn) V(mid) I(V4B)
.END
"""

cir_file = "phase2_m3.cir"
csv_file = cir_file + ".csv"

with open(cir_file, "w") as f:
    f.write(cir)

ret = subprocess.run(["Xyce", cir_file], capture_output=True, text=True)
if ret.returncode != 0:
    print("[ERROR]", ret.stderr[-500:])
    exit(1)

df = pd.read_csv(csv_file)
df.columns = ["v4", "vsn", "vmid", "i_v4b"]
# 부호: M0와 동일 규칙 — I(V4B) < 0 → I_M3 = -I(V4B)
df["im3_A"] = -df["i_v4b"]

print("V4    VSN      Vmid     I_M3(A)")
print("-" * 45)
for _, r in df.iloc[::10].iterrows():
    print(f"{r['v4']:5.2f}  {r['vsn']:7.4f}  {r['vmid']:7.4f}  {r['im3_A']:+.4e}")

# Python 모델: mid≈0V, v4bn=1V
# Vgd_eff = BSN - mid  = (V5+Vth-V4) - 0 = Vth + (V5-V4)   ← 유효 on-driving 쪽
# Vgs_eff = BSN - v4bn = (V5+Vth-V4) - 1 = Vth - V4
# I_M3 = [I_single(Vgd_eff) - I_single(Vgs_eff)] * 1e-9  A
x_arr = np.linspace(-1, 2, 400)
vgd_py = V5 + V0 - x_arr   # = Vth + (V5-V4)
vgs_py = V0 - x_arr         # = Vth - V4
i_bidir_nA = I_single(vgd_py) - I_single(vgs_py)
i_bidir_A  = i_bidir_nA * 1e-9

clip = 1e-23

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(
    f"Single M3 — Phase 2\n"
    f"V5={V5}V fixed  |  VSN = {bsn_const:.4f} − V4  |  V̄4 = +{VDD}V, V1 = 0V",
    fontsize=10)

# Xyce
ax.plot(df["v4"], np.abs(df["im3_A"]).clip(lower=clip),
        color="tomato", lw=2, ls="-", marker="o", markevery=5, markersize=5,
        label="Xyce (Phase 2)")

# Python
ax.plot(x_arr, np.abs(i_bidir_A).clip(clip),
        color="tomato", lw=1.2, ls="--", alpha=0.7,
        label="Python bidirectional")

# V4=V5 위치 (threshold crossing)
ax.axvline(V5, color="gray", lw=1, ls=":", alpha=0.7, label=f"V4=V5={V5}V  (threshold)")

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-5)
ax.set_xlim(-1, 2)
ax.set_xlabel("V4  [V]", fontsize=11)
ax.set_ylabel("|I_M3|  [A]", fontsize=11)
ax.set_title("Log |I_M3|  [A]  vs  V4  (M3 turns ON when V4 < V5)", fontsize=10)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("phase2_m3_log.png", dpi=150)
print("Saved: phase2_m3_log.png")
