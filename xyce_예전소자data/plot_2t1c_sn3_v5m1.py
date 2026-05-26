import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# New model result (logistic + linear extension)
df = pd.read_csv("circuit_2t1c_sn3_v5m1.cir.csv")
df.columns = ["t1", "time", "vsn", "vmid"]
df = df[df["time"] >= 0].reset_index(drop=True)

t_us  = df["time"] * 1e6
vsn   = df["vsn"]
vmid  = df["vmid"]

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
fig.suptitle("2T1C: V(SN)$_{IC}$=3V,  V3=3V, V4=0V, V5=−1V\n"
             "Model: Logistic + Linear Extension (fitted from data)", fontsize=11)

# ── V(SN) ──────────────────────────────────────────────────────────────
axes[0].plot(t_us, vsn, color="steelblue", linewidth=2, label="V(SN)  [new model]")
axes[0].axhline(-1.0, color="tomato", linestyle="--", linewidth=1.2,
                label="V5 = −1V  (theoretical floor)")
axes[0].axhline(0.66, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
                label="Old model stuck at 0.66V (= V3−VSAT)")
axes[0].axhline(3.0, color="lightgray", linestyle=":", linewidth=0.8)
axes[0].set_ylabel("V(SN) [V]", fontsize=11)
axes[0].set_ylim(-1.3, 3.3)
axes[0].legend(fontsize=9, loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_title("Storage Node Voltage", fontsize=10)

ss = vsn.iloc[-1]
axes[0].text(0.97, 0.12, f"Steady: {ss:.3f} V\n(≈ V5=−1V, limited by RBLEED)",
             transform=axes[0].transAxes, ha="right", fontsize=9, color="steelblue",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# ── V(mid) ─────────────────────────────────────────────────────────────
axes[1].plot(t_us, vmid, color="darkorange", linewidth=2, label="V(mid)")
axes[1].axhline(-1.0, color="tomato", linestyle="--", linewidth=1.0, alpha=0.7,
                label="V5 = −1V")
axes[1].set_ylabel("V(mid) [V]", fontsize=11)
axes[1].set_xlabel("Time [us]", fontsize=11)
axes[1].legend(fontsize=9, loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].set_title("Mid Node Voltage", fontsize=10)

plt.tight_layout()
plt.savefig("2t1c_sn3_v5m1_result.png", dpi=150)
print("Saved: 2t1c_sn3_v5m1_result.png")
print(f"\n[V5=-1V, new model]")
print(f"  V(SN) steady = {vsn.iloc[-1]:.4f} V  (old model was stuck at +0.66V)")
print(f"  V(mid) steady = {vmid.iloc[-1]:.4f} V")
print(f"  Theory: V(SN) -> V5 = -1V  (RBLEED causes ~0.08V offset)")
