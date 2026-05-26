import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("circuit_3t1c_k5_v4sweep_v2.cir.csv")
df.columns = ["v4", "vsn", "vmid", "iv5"]
df["im0_nA"] = -df["iv5"] * 1e9

# Unidirectional: Vgs_M0 = V(SN) - V5 = V4 - 1V
L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7

def I_single(vgs):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

v4_arr = np.linspace(-1.5, 1.5, 500)
vgs_m0 = v4_arr - 1.0          # Vgs_M0 = V(SN) - V5 = V4 - 1
i_unidir = I_single(vgs_m0)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(
    "3T1C: V1=0V, V2=3V, V3=-3V, V5=1V, VSN_init=0V\n"
    "V(SN)=V4,  Vgs$_{M0}$=V4−1V,  Vth=+0.11V",
    fontsize=11
)

# -- Left: I(M0) linear --
ax = axes[0]
ax.plot(df["v4"], df["im0_nA"], color="steelblue", linewidth=2, label="Bidirectional (Xyce)")
ax.plot(v4_arr, i_unidir, color="tomato", linewidth=2, linestyle="--", label="Unidirectional I_single(V4−1)")
ax.set_xlabel("V4 [V]", fontsize=11)
ax.set_ylabel("I$_{M0}$ [nA]", fontsize=11)
ax.set_title("I(M0)  Linear scale", fontsize=10)
ax.set_xlim(-1.5, 1.5)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=9)

# -- Middle: I(M0) log --
ax = axes[1]
ax.plot(df["v4"], df["im0_nA"].clip(lower=1e-5), color="steelblue", linewidth=2, label="Bidirectional (Xyce)")
ax.plot(v4_arr, np.clip(i_unidir, 1e-5, None), color="tomato", linewidth=2, linestyle="--", label="Unidirectional")
ax.set_xlabel("V4 [V]", fontsize=11)
ax.set_ylabel("I$_{M0}$ [nA]", fontsize=11)
ax.set_title("I(M0)  Log scale", fontsize=10)
ax.set_yscale("log")
ax.set_ylim(1e-5, 1e4)
ax.set_xlim(-1.5, 1.5)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=9)

# -- Right: V(mid) --
ax = axes[2]
ax.plot(df["v4"], df["vmid"], color="darkorange", linewidth=2, label="V(mid) [Xyce]")
ax.axhline(1.0, color="steelblue", linestyle=":", linewidth=1, alpha=0.7, label="V5=1V")
ax.axhline(0.0, color="black", linestyle=":", linewidth=1, alpha=0.5, label="V1=0V")
ax.set_xlabel("V4 [V]", fontsize=11)
ax.set_ylabel("V(mid) [V]", fontsize=11)
ax.set_title("V(mid) vs V4", fontsize=10)
ax.set_xlim(-1.5, 1.5)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=9)

# -- 4th: Vgd vs Vgs (physical convention: source=mid(low), drain=V5(high)) --
ax = axes[3]
vgd_phys = df["v4"] - 1.0          # Vgd = V(SN) - V5(drain) = V4 - 1
vgs_phys = df["v4"] - df["vmid"]   # Vgs = V(SN) - V(mid)(source) = V4 - V(mid)
ax.plot(vgd_phys, vgs_phys, color="purple", linewidth=2)
ax.axhline(V0, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"Vth={V0}V")
ax.axvline(V0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax.plot([-2.5, 2.5], [-2.5, 2.5], color="black", linestyle="--", linewidth=1, alpha=0.4, label="Vgs=Vgd (VDS=0)")
ax.set_xlabel("Vgd = V4 − 1V  [V]", fontsize=11)
ax.set_ylabel("Vgs = V4 − V(mid)  [V]", fontsize=11)
ax.set_title("M0: Vgd vs Vgs", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("3t1c_k5_v4sweep_v2.png", dpi=150)
print("Saved: 3t1c_k5_v4sweep_v2.png")

print(f"\n[Summary]  V5=1V, V(SN)=V4, Vgs_M0=V4-1V, Vth=+0.11V")
for v4 in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.11, 1.5]:
    row = df.iloc[(df["v4"] - v4).abs().idxmin()]
    print(f"  V4={v4:+.2f}V  Vgs={v4-1:+.2f}V  I(M0)={row['im0_nA']:+.4f} nA  V(mid)={row['vmid']:.4f}V")
