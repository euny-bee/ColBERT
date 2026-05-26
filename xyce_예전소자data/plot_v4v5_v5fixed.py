import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 6.9526; K = 5.0889; V0 = 0.1102; B = -13.1232; VSAT = 1.287; SLOPE = 2.307e-7
V0_shift = V0 + 0.5
V5 = 1.0

def I_single(vgs, v0=V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return (10**(B + L / (1 + np.exp(-K * (v - v0)))) + np.maximum(0, vgs - VSAT) * SLOPE) * 1e9

# Load Xyce CSVs (V5=+1.0V only)
def load_csv(fname):
    df = pd.read_csv(fname)
    df.columns = ["v4", "vsn", "vmid", "iv5"]
    df["im0_A"] = -df["iv5"]   # A 단위 (iv5 already in A)
    return df

df_orig  = load_csv("tmp_v4v5_p1d0.cir.csv")
df_shift = load_csv("tmp_vthshift_p1d0.cir.csv")

# Python I_single (V4 -1~1V)
v4_range = np.linspace(-1, 1, 300)
x_py     = v4_range + V5                          # x = V4 + V5

i_orig_A  = I_single(v4_range + V0,      v0=V0)       * 1e-9   # nA → A
i_shift_A = I_single(v4_range + V0_shift, v0=V0_shift) * 1e-9

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f"3T1C  Id(M0) vs (V4+V5)  —  V5 = +{V5}V fixed\n"
             "Solid: Python I_single  |  Dashed: Xyce bidirectional",
             fontsize=10)

clip = 1e-23

# Vth original
x_xyce = df_orig["v4"] + V5
ax.plot(x_xyce, np.abs(df_orig["im0_A"]).clip(lower=clip),
        color="steelblue", lw=1.2, ls="--", alpha=0.7, label="Xyce  Vth=+0.11V")
ax.plot(x_py, np.clip(i_orig_A, clip, None),
        color="steelblue", lw=2, ls="-", label="Python  Vth=+0.11V")

# Vth +0.5V shift
x_xyce_s = df_shift["v4"] + V5
ax.plot(x_xyce_s, np.abs(df_shift["im0_A"]).clip(lower=clip),
        color="tomato", lw=1.2, ls="--", alpha=0.7, label="Xyce  Vth=+0.61V (+0.5V)")
ax.plot(x_py, np.clip(i_shift_A, clip, None),
        color="tomato", lw=2, ls="-", label="Python  Vth=+0.61V (+0.5V)")

# kink 위치
for xk, color, label in [
    (VSAT - V0      + V5, "steelblue", f"kink Vth=+0.11V  (x={VSAT-V0+V5:.2f}V)"),
    (VSAT - V0_shift + V5, "tomato",   f"kink Vth=+0.61V  (x={VSAT-V0_shift+V5:.2f}V)"),
]:
    ax.axvline(xk, color=color, lw=0.8, ls=":", alpha=0.6, label=label)

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-5)
ax.set_xlim(-2, 2)
ax.set_xlabel("V4 + V5  [V]", fontsize=11)
ax.set_ylabel("I$_{M0}$  [A]", fontsize=11)
ax.set_title("Log scale  (V5=+1V fixed, V4: -1~1V)", fontsize=10)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("3t1c_k5_v5fixed_log.png", dpi=150)
print("Saved: 3t1c_k5_v5fixed_log.png")
