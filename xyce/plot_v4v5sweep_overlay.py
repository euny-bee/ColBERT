import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7
V0_shift = V0 + 0.5

def I_single(vgs, v0=V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

v5_list  = [-1.0, -0.5, 0.0, 0.5, 1.0]
colors   = plt.cm.coolwarm(np.linspace(0, 1, len(v5_list)))
v4_range = np.linspace(-1, 1, 300)   # V4 -1~1V only

# Load Xyce CSVs
def load_xyce(tag_fmt, v5_list):
    data = {}
    for v5 in v5_list:
        tag = f"{v5:+.1f}".replace("+","p").replace("-","m").replace(".","d")
        fname = tag_fmt.format(tag=tag)
        df = pd.read_csv(fname)
        df.columns = ["v4","vsn","vmid","iv5"]
        df["im0_nA"] = -df["iv5"] * 1e9
        data[v5] = df
    return data

orig_xyce  = load_xyce("tmp_v4v5_{tag}.cir.csv",    v5_list)
shift_xyce = load_xyce("tmp_vthshift_{tag}.cir.csv", v5_list)

fig, ax_single = plt.subplots(1, 1, figsize=(7, 5))
axes = [ax_single]
fig.suptitle("3T1C  Id(M0) vs (V4+V5)  — Vth comparison  [Log scale]\n"
             "Solid: Python I_single  |  Dashed: Xyce bidirectional\n"
             "Dark: Vth=+0.11V (original)  |  Light+marker: Vth=+0.61V (+0.5V shift)",
             fontsize=10)

unit_info = {
    "linear": {"scale": 1.0,   "unit": "nA", "ylim": None,          "clip": None},
    "log":    {"scale": 1e-9,  "unit": "A",  "ylim": (1e-14, 1e-5), "clip": 1e-23},
}

for ax, yscale in zip(axes, ["log"]):
    sc   = unit_info[yscale]["scale"]
    unit = unit_info[yscale]["unit"]

    for (v5, df_orig), (_, df_shift), color in zip(
            orig_xyce.items(), shift_xyce.items(), colors):
        x_orig  = df_orig["v4"]  + v5
        x_shift = df_shift["v4"] + v5

        # --- Xyce (점선) ---
        yo = np.abs(df_orig["im0_nA"])  * sc
        ys = np.abs(df_shift["im0_nA"]) * sc
        if unit_info[yscale]["clip"]:
            yo = yo.clip(lower=unit_info[yscale]["clip"])
            ys = ys.clip(lower=unit_info[yscale]["clip"])
        ax.plot(x_orig,  yo, color=color, lw=1,   ls="--", alpha=0.6)
        ax.plot(x_shift, ys, color=color, lw=1,   ls="--", alpha=0.3)

        # --- Python I_single (실선, V4 -1~1V 범위만) ---
        x_py   = v4_range + v5                        # x = V4 + V5
        i_orig  = I_single(v4_range + V0,      v0=V0)       * sc
        i_shift = I_single(v4_range + V0_shift, v0=V0_shift) * sc
        if unit_info[yscale]["clip"]:
            i_orig  = np.clip(i_orig,  unit_info[yscale]["clip"], None)
            i_shift = np.clip(i_shift, unit_info[yscale]["clip"], None)
        ax.plot(x_py, i_orig,  color=color, lw=2,   ls="-", alpha=1.0,
                label=f"V5={v5:+.1f}V" if ax is axes[0] else "")
        ax.plot(x_py, i_shift, color=color, lw=1.5, ls="-", alpha=0.5,
                marker="o", markevery=40, markersize=3)

        # kink 위치 표시
        for xk, alp in [(VSAT-V0+v5, 0.5), (VSAT-V0_shift+v5, 0.3)]:
            if -2 <= xk <= 2:
                ax.axvline(xk, color=color, lw=0.7, ls=":", alpha=alp)

    ax.axvline(0, color="gray", lw=1, ls=":", alpha=0.5)
    ax.set_xlabel("V4 + V5  [V]", fontsize=11)
    ax.set_ylabel(f"I$_{{M0}}$ [{unit}]", fontsize=11)
    ax.set_xlim(-2, 2)
    ax.grid(True, ls="--", alpha=0.4)

    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(*unit_info["log"]["ylim"])
        ax.set_title("Log scale  [A]", fontsize=10)
    else:
        ax.set_title("Linear scale  [nA]", fontsize=10)

# 범례
v5_handles = [Line2D([0],[0], color=c, lw=2, label=f"V5={v5:+.1f}V")
              for v5, c in zip(v5_list, colors)]
style_handles = [
    Line2D([0],[0], color="k", lw=2,   ls="-",  label="Python I_single"),
    Line2D([0],[0], color="k", lw=1,   ls="--", label="Xyce bidirectional"),
    Line2D([0],[0], color="k", lw=2,   ls="-",  alpha=1.0, label="Vth=+0.11V (original)"),
    Line2D([0],[0], color="k", lw=1.5, ls="-",  alpha=0.5,
           marker="o", markersize=4,            label="Vth=+0.61V (+0.5V shift)"),
    Line2D([0],[0], color="k", lw=0.7, ls=":",  label="kink position"),
]
axes[0].legend(handles=v5_handles + style_handles, fontsize=7.5, ncol=2)

plt.tight_layout()
plt.savefig("3t1c_k5_v4v5sweep_overlay_log.png", dpi=150)
print("Saved: 3t1c_k5_v4v5sweep_overlay_log.png")
