import subprocess, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7

def I_single(vgs):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

v5_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
results = {}

for v5 in v5_list:
    tag = f"{v5:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    cir_file = f"tmp_v4v5_{tag}.cir"
    csv_file = cir_file + ".csv"

    # VSN = V4 + V5 + Vth  →  BSN offset = V5 + Vth
    offset = v5 + V0

    cir = f"""* 3T1C  VSN=Vth+V4+V5  V5={v5}V
.INCLUDE igzo_smooth_k5.sub
VV1  v1n  0  DC 0.0
VV2  v2n  0  DC 3.0
VV3  v3n  0  DC -3.0
VV4  v4n  0  DC 0.0
VV5  v5n  0  DC {v5}
BSN  sn  0  V={{V(v4n) + {offset:.4f}}}
XM2  v1n  v2n  mid  IGZO_SMOOTH_K5
XM1  mid  v3n  sn   IGZO_SMOOTH_K5
XM0  mid  sn   v5n  IGZO_SMOOTH_K5
RMID    mid  0  10T
.DC VV4 -1 1 0.02
.PRINT DC FORMAT=CSV  V(v4n)  V(sn)  V(mid)  I(VV5)
.END
"""
    with open(cir_file, "w") as f:
        f.write(cir)

    ret = subprocess.run(["Xyce", cir_file], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[ERROR] V5={v5}: {ret.stderr[-300:]}")
        continue

    df = pd.read_csv(csv_file)
    df.columns = ["v4", "vsn", "vmid", "iv5"]
    df["im0_nA"] = -df["iv5"] * 1e9
    results[v5] = df
    print(f"V5={v5:+.1f}V done  |  V4=0: I={df.loc[(df.v4-0).abs().idxmin(),'im0_nA']:+.3f} nA"
          f"  Vmid={df.loc[(df.v4-0).abs().idxmin(),'vmid']:.3f}V")

# Unidirectional: Vgs_M0 = Vth + V4+V5 - V5 = Vth + (V4+V5) - (V5) ... x=(V4+V5), I=I_single(Vth+(V4+V5)-V5)
# Since Vgs = Vth + V4, unidirectional is independent of V5 — single reference curve
x_uni = np.linspace(-2, 2, 500)          # x = V4+V5

# Save Python I_single results to CSV (one per V5)
for v5 in v5_list:
    tag = f"{v5:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    vgs_arr = x_uni - v5 + V0       # Vgs = Vth + V4 = Vth + (x - V5)
    i_arr   = I_single(vgs_arr)
    df_uni = pd.DataFrame({
        "v4_plus_v5": x_uni,
        "vgs_m0":     vgs_arr,
        "im0_nA":     i_arr,
    })
    csv_path = f"python_unidirectional_{tag}.csv"
    df_uni.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

colors = plt.cm.coolwarm(np.linspace(0, 1, len(v5_list)))

from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("3T1C  Id(M0) vs (V4+V5)  (V5 sweep)\n"
             "VSN = Vth + (V4+V5),   x축 = V4+V5",
             fontsize=11)

# Panel 1: Linear scale
ax = axes[0]
for (v5, df), color in zip(results.items(), colors):
    x = df["v4"] + v5
    ax.plot(x, df["im0_nA"], color=color, linewidth=1, linestyle="--", alpha=0.5)
    i_uni = I_single(V0 + x_uni - v5)
    ax.plot(x_uni, i_uni, color=color, linewidth=2, label=f"V5={v5:+.1f}V")
ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="V4+V5=0")
ax.set_xlabel("V4 + V5 [V]", fontsize=11)
ax.set_ylabel("I$_{M0}$ [nA]", fontsize=11)
ax.set_title("Linear scale", fontsize=10)
ax.set_xlim(-2, 2)
ax.grid(True, linestyle="--", alpha=0.4)
# V5 color legend
v5_handles = [Line2D([0],[0], color=c, linewidth=2, label=f"V5={v5:+.1f}V")
              for (v5, _), c in zip(results.items(), colors)]
style_handles = [Line2D([0],[0], color="k", linewidth=2, linestyle="-",  label="Python I_single"),
                 Line2D([0],[0], color="k", linewidth=1, linestyle="--", label="Xyce simulation")]
ax.legend(handles=v5_handles + style_handles, fontsize=8)

# Panel 2: Log scale
ax = axes[1]
for (v5, df), color in zip(results.items(), colors):
    x = df["v4"] + v5
    ax.plot(x, np.abs(df["im0_nA"]).clip(lower=1e-5), color=color, linewidth=1, linestyle="--", alpha=0.5)
    i_uni = I_single(V0 + x_uni - v5)
    ax.plot(x_uni, np.clip(i_uni, 1e-5, None), color=color, linewidth=2, label=f"V5={v5:+.1f}V")
ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel("V4 + V5 [V]", fontsize=11)
ax.set_ylabel("|I$_{M0}$| [nA]", fontsize=11)
ax.set_title("Log scale", fontsize=10)
ax.set_yscale("log")
ax.set_ylim(1e-5, 1e4)
ax.set_xlim(-2, 2)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(handles=v5_handles + style_handles, fontsize=8)


plt.tight_layout()
plt.savefig("3t1c_k5_v4v5sweep.png", dpi=150)
print("Saved: 3t1c_k5_v4v5sweep.png")
