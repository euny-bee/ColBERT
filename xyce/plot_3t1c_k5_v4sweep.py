import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Xyce bidirectional result
df = pd.read_csv("circuit_3t1c_k5_v4sweep.cir.csv")
df.columns = ["v4", "vsn", "vmid", "iv5"]
df["im0_nA"] = -df["iv5"] * 1e9

# Python unidirectional: I_single(Vgs_M0) = I_single(V4)
L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7

def I_single(vgs):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9  # nA

v4_arr = np.linspace(-3, 3, 500)
i_unidir = I_single(v4_arr)   # Vgs_M0 = V4

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("3T1C: M0 current vs V4  (V5=3V, V(SN)=V4+3V, Vgs$_{M0}$=V4)\n"
             "Bidirectional (Xyce) vs Unidirectional (Python)  |  Vth=+0.11V, K=5.09", fontsize=11)

for ax, yscale in zip(axes, ["linear", "log"]):
    ax.plot(df["v4"], df["im0_nA"], color="steelblue", linewidth=2,
            label="Bidirectional (Xyce)")
    ax.plot(v4_arr, i_unidir, color="tomato", linewidth=2, linestyle="--",
            label="Unidirectional I_single(V4)")
    ax.axvline(0.11, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Vth=0.11V")

    ax.set_xlabel("V4 [V]   (= Vgs$_{M0}$)", fontsize=11)
    ax.set_ylabel("I$_{M0}$ [nA]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-3, 3)
    ax.legend(fontsize=9)

    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1e4)
        ax.set_title("Log scale", fontsize=10)
    else:
        ax.set_title("Linear scale", fontsize=10)

plt.tight_layout()
plt.savefig("3t1c_k5_v4sweep_im0.png", dpi=150)
print("Saved: 3t1c_k5_v4sweep_im0.png")

print(f"\n[Summary]  V5=3V, V(SN)=V4+3V, Vgs_M0=V4, Vth=+0.11V")
for v4 in [-3, -2, -1, 0, 1, 2, 3]:
    row = df.iloc[(df["v4"] - v4).abs().idxmin()]
    print(f"  V4={v4:+.0f}V  Vgs={v4:+.2f}V  →  I(M0)={row['im0_nA']:+.3f} nA  V(mid)={row['vmid']:.3f}V")
