import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("circuit_2t1c_sn3.cir.csv")
df.columns = ["t1", "time", "vsn", "vmid"]
df = df[df["time"] >= 0].reset_index(drop=True)

t_us = df["time"] * 1e6   # µs
vsn  = df["vsn"]
vmid = df["vmid"]

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
fig.suptitle("2T1C:  M2 removed,  V(SN)$_{IC}$=3V,  V3=3V, V4=0V, V5=1V\n"
             "M1: gate=V3, source=SN   |   M0: gate=SN, source=V5",
             fontsize=11)

# ── V(SN) ──────────────────────────────────────────────────────────────
axes[0].plot(t_us, vsn, color="steelblue", linewidth=2, label="V(SN)")
axes[0].axhline(1.0, color="tomato", linestyle="--", linewidth=1.2,
                label="V5 = 1V  (M0 threshold: Vgs→0 when SN→V5)")
axes[0].axhline(3.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
                label="V(SN) initial = 3V")
axes[0].set_ylabel("V(SN) [V]", fontsize=11)
axes[0].set_ylim(0.5, 3.5)
axes[0].legend(fontsize=9, loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_title("Storage Node Voltage  —  discharge via M0  (Vgs_M0 = V(SN) − 1V)", fontsize=10)

# ── V(mid) ─────────────────────────────────────────────────────────────
axes[1].plot(t_us, vmid, color="darkorange", linewidth=2, label="V(mid)")
axes[1].axhline(1.0, color="tomato", linestyle="--", linewidth=1.0, alpha=0.7,
                label="V5 = 1V")
axes[1].set_ylabel("V(mid) [V]", fontsize=11)
axes[1].set_xlabel("Time [us]", fontsize=11)
axes[1].legend(fontsize=9, loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].set_title("Mid Node Voltage  —  shared drain node (M1 & M0)", fontsize=10)

# Annotation: steady-state approach
ss_vsn = vsn.iloc[-1]
axes[0].text(0.97, 0.12,
             f"t=10µs: V(SN)={ss_vsn:.3f}V\n(slowly → 1V as Vgs_M0→0)",
             transform=axes[0].transAxes, ha="right", fontsize=9, color="steelblue",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("2t1c_sn3_result.png", dpi=150)
print("Saved: 2t1c_sn3_result.png")
print(f"\n[V(SN)=3V IC 결과]")
print(f"  t=0      : V(SN) = 3.000 V,  Vgs_M0 = 2.000 V  (M0 강하게 ON)")
print(f"  t=10 µs  : V(SN) = {ss_vsn:.3f} V,  Vgs_M0 = {ss_vsn-1:.3f} V")
print(f"  t→∞     : V(SN) → 1V  (M0 꺼지면서 방전 정지)")
print(f"\n  V(mid) 최종: {vmid.iloc[-1]:.3f} V  (≈ V(SN), M1 통해 equalize)")
