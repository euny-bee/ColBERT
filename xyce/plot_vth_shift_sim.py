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

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("2T1C: V(SN)$_{IC}$=3V, V3=3V, V4=0V, V5=−1V\n"
             "Effect of Vth shift on discharge behavior", fontsize=12)

for fname, label, color in cases:
    df = load(fname, ["t1","time","vsn","vmid"])
    t_us = df["time"] * 1e6
    vsn  = df["vsn"]
    vmid = df["vmid"]
    ss   = vsn.iloc[-1]

    axes[0].plot(t_us, vsn,  color=color, linewidth=2,
                 label=f"{label}  →  {ss:.3f} V")
    axes[1].plot(t_us, vmid, color=color, linewidth=2, label=label)

axes[0].axhline(-1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                label="V5 = -1V")
axes[0].axhline(3.0,  color="gray", linestyle=":",  linewidth=0.8, alpha=0.4)
axes[0].set_ylabel("V(SN) [V]", fontsize=11)
axes[0].set_ylim(-1.3, 3.3)
axes[0].legend(fontsize=9, loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_title("Storage Node Voltage  —  steady-state depends on Vth", fontsize=10)

axes[1].axhline(-1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
axes[1].set_ylabel("V(mid) [V]", fontsize=11)
axes[1].set_xlabel("Time [us]", fontsize=11)
axes[1].legend(fontsize=9, loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].set_title("Mid Node Voltage", fontsize=10)

plt.tight_layout()
plt.savefig("vth_shift_sim_comparison.png", dpi=150)
print("Saved: vth_shift_sim_comparison.png")

print("\n[Steady-state V(SN) vs Vth]")
print("  Theory: V(SN)_ss = Vth + V5 = Vth + (-1)")
for fname, label, _ in cases:
    df = load(fname, ["t1","time","vsn","vmid"])
    ss = df["vsn"].iloc[-1]
    print(f"  {label.split('(')[0].strip():30s}  V(SN) = {ss:.4f} V")
