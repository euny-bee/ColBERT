import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(fname, cols):
    df = pd.read_csv(fname)
    df.columns = cols
    return df[df["time"] >= 0].reset_index(drop=True)

cases = [
    ("circuit_2t1c_sn3_v5m1_m0p5V.cir.csv", "Vth = -0.42V  (shift -0.5V)", "tomato"),
    ("circuit_2t1c_sn3_v5m1.cir.csv",        "Vth =  +0.08V (original)",     "steelblue"),
    ("circuit_2t1c_sn3_v5m1_p0p5V.cir.csv",  "Vth = +0.61V  (shift +0.5V)", "forestgreen"),
]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.suptitle("2T1C: V(SN)$_{IC}$=3V, V3=3V, V4=0V, V5=−1V\n"
             "Effect of Vth shift on discharge behavior", fontsize=12)

for fname, label, color in cases:
    df = load(fname, ["t1","time","vsn","vmid"])
    t_us = df["time"] * 1e6
    vsn  = df["vsn"]
    ss   = vsn.iloc[-1]
    ax.plot(t_us, vsn, color=color, linewidth=2,
            label=f"{label}  →  {ss:.3f} V")

ax.axhline(-1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6,
           label="V5 = -1V")
ax.axhline(3.0,  color="gray", linestyle=":",  linewidth=0.8, alpha=0.4)
ax.set_ylabel("V(SN) [V]", fontsize=11)
ax.set_xlabel("Time [us]", fontsize=11)
ax.set_ylim(-1.3, 3.3)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_title("Storage Node Voltage  —  steady-state depends on Vth", fontsize=10)

plt.tight_layout()
plt.savefig("vth_shift_sim_comparison_vsn.png", dpi=150)
print("Saved: vth_shift_sim_comparison_vsn.png")

print("\n[Steady-state V(SN) vs Vth]")
print("  Theory: V(SN)_ss = Vth + V5 = Vth + (-1)")
for fname, label, _ in cases:
    df = load(fname, ["t1","time","vsn","vmid"])
    ss = df["vsn"].iloc[-1]
    print(f"  {label.split('(')[0].strip():30s}  V(SN) = {ss:.4f} V")
