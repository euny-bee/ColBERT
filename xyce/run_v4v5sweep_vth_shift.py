import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7
V0_shift = V0 + 0.5   # M0 Vth shifted +0.5V

def I_single(vgs, v0=V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

v5_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
results = {}

for v5 in v5_list:
    tag = f"{v5:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    cir_file = f"tmp_vthshift_{tag}.cir"
    csv_file = cir_file + ".csv"
    offset = v5 + V0_shift   # VSN = V4 + V5 + Vth_shifted

    cir = f"""* 3T1C  VSN=Vth_shift+V4+V5  M0 Vth=+{V0_shift:.4f}V  V5={v5}V
.INCLUDE igzo_smooth_k5.sub
VV1  v1n  0  DC 0.0
VV2  v2n  0  DC 3.0
VV3  v3n  0  DC -3.0
VV4  v4n  0  DC 0.0
VV5  v5n  0  DC {v5}
BSN  sn  0  V={{V(v4n) + {offset:.4f}}}
XM2  v1n  v2n  mid  IGZO_SMOOTH_K5
XM1  mid  v3n  sn   IGZO_SMOOTH_K5
XM0  mid  sn   v5n  IGZO_SMOOTH_K5  PARAMS: V0={V0_shift:.4f}
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
    print(f"V5={v5:+.1f}V done  |  V4=0: I={df.loc[(df.v4-0).abs().idxmin(),'im0_nA']:+.3f} nA")

# Python CSV 저장
x_uni = np.linspace(-2, 2, 500)
for v5 in v5_list:
    tag = f"{v5:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    vgs_arr = x_uni - v5 + V0_shift
    i_arr   = I_single(vgs_arr, v0=V0_shift)
    pd.DataFrame({"v4_plus_v5": x_uni, "vgs_m0": vgs_arr, "im0_nA": i_arr}
                 ).to_csv(f"python_unidirectional_vthshift_{tag}.csv", index=False)

# Plot
colors = plt.cm.coolwarm(np.linspace(0, 1, len(v5_list)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"3T1C  Id(M0) vs (V4+V5)  —  M0 Vth = +{V0_shift:.4f}V  (+0.5V shift)\n"
             f"VSN = Vth_shift + (V4+V5),   Vgs_M0 = Vth_shift + V4",
             fontsize=11)

v5_handles = [Line2D([0],[0], color=c, linewidth=2, label=f"V5={v5:+.1f}V")
              for (v5, _), c in zip(results.items(), colors)]
style_handles = [Line2D([0],[0], color="k", linewidth=2, linestyle="-",  label="Python I_single"),
                 Line2D([0],[0], color="k", linewidth=1, linestyle="--", label="Xyce simulation")]

for ax, yscale in zip(axes, ["linear", "log"]):
    for (v5, df), color in zip(results.items(), colors):
        x = df["v4"] + v5
        y_xyce = df["im0_nA"] if yscale == "linear" else np.abs(df["im0_nA"]).clip(lower=1e-5)
        ax.plot(x, y_xyce, color=color, linewidth=1, linestyle="--", alpha=0.5)

        vgs_arr = x_uni - v5 + V0_shift
        i_uni = I_single(vgs_arr, v0=V0_shift)
        y_uni = i_uni if yscale == "linear" else np.clip(i_uni, 1e-5, None)
        ax.plot(x_uni, y_uni, color=color, linewidth=2)

    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("V4 + V5 [V]", fontsize=11)
    ax.set_ylabel("I$_{M0}$ [nA]", fontsize=11)
    ax.set_xlim(-2, 2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(handles=v5_handles + style_handles, fontsize=8)

    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1e4)
        ax.set_title("Log scale", fontsize=10)
    else:
        ax.set_title("Linear scale", fontsize=10)

plt.tight_layout()
plt.savefig("3t1c_k5_v4v5sweep_vthshift.png", dpi=150)
print("Saved: 3t1c_k5_v4v5sweep_vthshift.png")
