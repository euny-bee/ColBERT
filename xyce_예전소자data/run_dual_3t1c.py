import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7

def I_single(vgs, v0=V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

# ── Left circuit (M0): V5<0 fixed, V4 swept ─────────────────────────────────
v5_list = [-1.0, -0.5]
left_data = {}
for v5 in v5_list:
    tag = f"{v5:+.1f}".replace("+","p").replace("-","m").replace(".","d")
    df = pd.read_csv(f"tmp_v4v5_{tag}.cir.csv")
    df.columns = ["v4", "vsn", "vmid", "iv5"]
    df["im_A"] = -df["iv5"]
    left_data[v5] = df
    print(f"Left  V5={v5:+.1f}V loaded  |  V4=0: I={df.loc[(df.v4-0).abs().idxmin(),'im_A']*1e9:+.3f} nA")

# ── Right circuit (M3): V4<0 fixed, V5 swept ────────────────────────────────
v4_list = [-1.0, -0.5]
right_data = {}
for v4 in v4_list:
    tag = f"{v4:+.1f}".replace("+","p").replace("-","m").replace(".","d")
    cir_file = f"tmp_right_{tag}.cir"
    csv_file  = cir_file + ".csv"
    offset = v4 + V0   # BSN = V5 + (V4_fixed + Vth)  ->  VSN = Vth + V4 + V5

    cir = f"""* Right circuit (M3): V4={v4}V fixed, V5 swept
* VSN = Vth + V4_fixed + V5,  Vgs_M3 = VSN - V4 = Vth + V5
.INCLUDE igzo_smooth_k5.sub
VV1  v1n  0  DC 0.0
VV2  v2n  0  DC 3.0
VV3  v3n  0  DC -3.0
VV4  v4n  0  DC {v4}
VV5  v5n  0  DC 0.0
BSN  sn  0  V={{V(v5n) + {offset:.4f}}}
XM5  v1n  v2n  mid  IGZO_SMOOTH_K5
XM4  mid  v3n  sn   IGZO_SMOOTH_K5
XM3  mid  sn   v4n  IGZO_SMOOTH_K5
RMID mid  0  10T
.DC VV5 -1 1 0.02
.PRINT DC FORMAT=CSV V(v5n) V(sn) V(mid) I(VV4)
.END
"""
    with open(cir_file, "w") as f:
        f.write(cir)

    ret = subprocess.run(["Xyce", cir_file], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[ERROR] V4={v4}: {ret.stderr[-300:]}")
        continue

    df = pd.read_csv(csv_file)
    df.columns = ["v5", "vsn", "vmid", "iv4"]
    df["im_A"] = -df["iv4"]
    right_data[v4] = df
    print(f"Right V4={v4:+.1f}V done  |  V5=0: I={df.loc[(df.v5-0).abs().idxmin(),'im_A']*1e9:+.3f} nA")

# ── Helper ───────────────────────────────────────────────────────────────────
clip = 1e-23
colors_left  = ["steelblue",  "cornflowerblue"]
colors_right = ["tomato",     "salmon"]

def I_bidir_A(vgs_arr, vgd_arr, v0=V0):
    return (I_single(np.asarray(vgs_arr), v0) - I_single(np.asarray(vgd_arr), v0)) * 1e-9

def build_legend(axes_ref):
    return (
        [Line2D([0],[0], color="k", lw=2, ls="-",  marker="o", markersize=4, label="M0: Xyce solid+marker"),
         Line2D([0],[0], color="k", lw=2, ls="--",              label="M3: Xyce dashed  (same curve, symmetric)"),
         Line2D([0],[0], color="gray", lw=1.2, ls="--",         label="Python bidirectional (dashed)")]
        + [Line2D([0],[0], color=c, lw=2, marker="o", markersize=4, label=f"M0 V5={v:+.1f}V") for v,c in zip(v5_list,colors_left)]
        + [Line2D([0],[0], color=c, lw=2, ls="--", label=f"M3 V4={v:+.1f}V") for v,c in zip(v4_list,colors_right)]
    )

def draw_axes(ax, xyce_ls, xyce_lw, xyce_alpha, py_ls, py_lw, py_alpha):
    # M0: xyce_ls 스타일 + 마커 (파랑), Python: py_ls + 마커
    for (v5, df), color in zip(left_data.items(), colors_left):
        x = df["v4"] + v5
        ax.plot(x, np.abs(df["im_A"]).clip(lower=clip),
                color=color, lw=xyce_lw, ls=xyce_ls, alpha=xyce_alpha,
                marker="o", markevery=10, markersize=4, zorder=3)
        vgs = df["vsn"] - v5
        vgd = df["vsn"] - df["vmid"]
        i_py = np.abs(I_bidir_A(vgs, vgd)).clip(clip)
        ax.plot(x, i_py, color=color, lw=py_lw, ls=py_ls, alpha=py_alpha,
                marker="o", markevery=10, markersize=4, zorder=3,
                label=f"M0  V5={v5:+.1f}V")
        xk = (VSAT - V0) + v5
        if -2 <= xk <= 2:
            ax.axvline(xk, color=color, lw=0.7, ls=":", alpha=0.4, zorder=1)

    # M3: xyce_ls 스타일 + 점선 덮어그리기 (빨강)  → 마커 사이 빈 공간에서 보임
    for (v4, df), color in zip(right_data.items(), colors_right):
        x = df["v5"] + v4
        ax.plot(x, np.abs(df["im_A"]).clip(lower=clip),
                color=color, lw=xyce_lw+0.5, ls="--", alpha=0.85, zorder=2)
        vgs = df["vsn"] - v4
        vgd = df["vsn"] - df["vmid"]
        i_py = np.abs(I_bidir_A(vgs, vgd)).clip(clip)
        ax.plot(x, i_py, color=color, lw=py_lw+0.5, ls="--", alpha=0.85, zorder=2,
                label=f"M3  V4={v4:+.1f}V")
        xk = (VSAT - V0) + v4
        if -2 <= xk <= 2:
            ax.axvline(xk, color=color, lw=0.7, ls=":", alpha=0.4, zorder=1)

    ax.axvline(0, color="gray", lw=1, ls=":", alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 1e-5)
    ax.set_xlim(-2, 2)
    ax.set_xlabel("V4 + V5  [V]", fontsize=11)
    ax.set_ylabel("|I|  [A]", fontsize=11)
    ax.grid(True, ls="--", alpha=0.4)

# ── Figure 1: Xyce = solid (V-shape visible), Python = dashed ─────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig1.suptitle("Dual 3T1C:  I_M0 (V5<0)  vs  I_M3 (V4<0)\n"
              "Solid: Xyce (V-shape)  |  Dashed: Python bidirectional",
              fontsize=10)
draw_axes(ax1, xyce_ls="-", xyce_lw=2, xyce_alpha=0.9,
               py_ls="--", py_lw=1.2, py_alpha=0.6)
ax1.set_title("Log |I|  [A]  —  Xyce solid", fontsize=9)
ax1.legend(handles=build_legend(ax1), fontsize=8, ncol=2)
plt.tight_layout()
fig1.savefig("dual_3t1c_v_log.png", dpi=150)
print("Saved: dual_3t1c_v_log.png")

# ── Figure 2: Python = solid, Xyce = dashed (original convention) ──────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.suptitle("Dual 3T1C:  I_M0 (V5<0)  vs  I_M3 (V4<0)\n"
              "Solid: Python bidirectional  |  Dashed: Xyce simulation",
              fontsize=10)
draw_axes(ax2, xyce_ls="--", xyce_lw=1.2, xyce_alpha=0.7,
               py_ls="-",  py_lw=2,   py_alpha=1.0)
ax2.set_title("Log |I|  [A]  —  Python solid", fontsize=9)
style2 = [Line2D([0],[0], color="k", lw=2,   ls="-",  label="Python bidirectional"),
          Line2D([0],[0], color="k", lw=1.2, ls="--", label="Xyce simulation")]
ax2.legend(handles=style2
           + [Line2D([0],[0], color=c, lw=2, label=f"M0 V5={v:+.1f}V") for v,c in zip(v5_list,colors_left)]
           + [Line2D([0],[0], color=c, lw=2, label=f"M3 V4={v:+.1f}V") for v,c in zip(v4_list,colors_right)],
           fontsize=8, ncol=2)
plt.tight_layout()
fig2.savefig("dual_3t1c_log.png", dpi=150)
print("Saved: dual_3t1c_log.png")

plt.tight_layout()
plt.savefig("dual_3t1c_log.png", dpi=150)
print("Saved: dual_3t1c_log.png")
