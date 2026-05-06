import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(fname):
    df = pd.read_csv(fname)
    df.columns = ["t1","time","vsn","vmid"]
    return df[df["time"] >= 0].reset_index(drop=True)

cases = [
    ("circuit_2t1c_sn3_v5p1_m0p5V.cir.csv",
     "Vth = -0.42V  (shift -0.5V)", "tomato",     -0.3898, 0.5),
    ("circuit_2t1c_sn3_v5p1_orig.cir.csv",
     "Vth =  +0.08V (original)",    "steelblue",   0.0802,  0.5),
    ("circuit_2t1c_sn3_v5p1_p0p5V.cir.csv",
     "Vth = +0.61V  (shift +0.5V)", "forestgreen", 0.6102,  0.5),
]
V5 = 1.0

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("2T1C: V(SN)$_{IC}$=3V, V3=3V, V4=0V, V5=+1V\n"
             "Effect of Vth shift  (discharge from 3V)", fontsize=12)

for fname, label, color, vth, _ in cases:
    df = load(fname)
    t_us = df["time"] * 1e6
    vsn  = df["vsn"]
    vmid = df["vmid"]
    theory = max(vth + V5, V5)   # clamped at V5 if Vth+V5 < V5

    axes[0].plot(t_us, vsn,  color=color, linewidth=2,
                 label=f"{label}")
    axes[1].plot(t_us, vmid, color=color, linewidth=2, label=label)

    # Theoretical steady state line
    axes[0].axhline(theory, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].text(51, theory + 0.04,
                 f"Vth+V5={'%.2f'%(vth+V5)}V" + (" → clamped" if vth+V5 < V5 else ""),
                 color=color, fontsize=8, va="bottom")

axes[0].axhline(V5, color="gray", linestyle=":", linewidth=1.0, alpha=0.5,
                label=f"V5 = {V5}V  (self-clamp floor)")
axes[0].axhline(3.0, color="gray", linestyle=":",  linewidth=0.8, alpha=0.3)
axes[0].set_ylabel("V(SN) [V]", fontsize=11)
axes[0].set_ylim(0.5, 3.3)
axes[0].legend(fontsize=9, loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_title("Storage Node Voltage  (dashed = theoretical V(SN)$_{ss}$ = max(Vth+V5, V5))", fontsize=10)

axes[1].axhline(V5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
axes[1].set_ylabel("V(mid) [V]", fontsize=11)
axes[1].set_xlabel("Time [us]", fontsize=11)
axes[1].legend(fontsize=9, loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].set_title("Mid Node Voltage", fontsize=10)

plt.xlim(0, 50)
plt.tight_layout()
plt.savefig("vth_shift_v5p1_comparison.png", dpi=150)
print("Saved: vth_shift_v5p1_comparison.png")

print(f"\n[Summary]  V3=3V, V4=0V, V5=+1V,  V(SN)_IC=3V")
print(f"  Theory: V(SN)_ss = max(Vth + V5,  V5) = max(Vth + 1,  1)")
for fname, label, color, vth, _ in cases:
    df = load(fname)
    theory = max(vth + V5, V5)
    sim    = df["vsn"].iloc[-1]
    print(f"  {label.split('(')[0].strip():28s}  theory={theory:.3f}V  sim@50us={sim:.3f}V")
