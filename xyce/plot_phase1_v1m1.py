import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("circuit_phase1_v1m1.cir.csv")
df.columns = ["t1", "time", "vsn", "vmid", "im0s"]
df = df[df["time"] >= 0].reset_index(drop=True)

t_ns   = df["time"] * 1e9
vsn    = df["vsn"]
vmid   = df["vmid"]
im0_nA = df["im0s"].abs() * 1e9

fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
fig.suptitle("Phase 1:  V1=−1V, V2=3V, V3=3V, V4=0V, V5=−3V\n"
             "Data store  (C=100fF, bidirectional IGZO model)", fontsize=11)

# ── V(SN) ──────────────────────────────────────────────────────────────
axes[0].plot(t_ns, vsn, color="steelblue", linewidth=2)
axes[0].set_ylabel("V(SN) [V]", fontsize=11)
axes[0].set_ylim(-0.5, 0.5)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                label="0 V reference")
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_title("Storage Node Voltage  V(SN) — gate of M0", fontsize=10)

# ── V(mid) ─────────────────────────────────────────────────────────────
axes[1].plot(t_ns, vmid, color="darkorange", linewidth=2)
axes[1].set_ylabel("V(mid) [V]", fontsize=11)
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].axhline(-1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                label="V1 = −1 V")
axes[1].legend(fontsize=9)
axes[1].set_title("Mid Node Voltage  V(mid)", fontsize=10)

# ── I(M0) ──────────────────────────────────────────────────────────────
axes[2].plot(t_ns, im0_nA, color="tomato", linewidth=2)
axes[2].set_ylabel("|I$_{M0}$| [nA]", fontsize=11)
axes[2].set_xlabel("Time [ns]", fontsize=11)
axes[2].grid(True, linestyle="--", alpha=0.4)
axes[2].set_title("M0 Drive Current", fontsize=10)

# Steady-state annotations
ss_vsn  = vsn.iloc[-1]
ss_vmid = vmid.iloc[-1]
ss_im0  = im0_nA.iloc[-1]

axes[0].text(0.97, 0.15, f"Steady: {ss_vsn*1e3:.2f} mV",
             transform=axes[0].transAxes, ha="right", fontsize=9, color="steelblue",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[1].text(0.97, 0.15, f"Steady: {ss_vmid:.3f} V",
             transform=axes[1].transAxes, ha="right", fontsize=9, color="darkorange",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[2].text(0.97, 0.85, f"Steady: {ss_im0:.2f} nA",
             transform=axes[2].transAxes, ha="right", fontsize=9, color="tomato",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("phase1_v1m1_result.png", dpi=150)
print("Saved: phase1_v1m1_result.png")
print(f"\n[Phase 1, V1=−1V  steady state]")
print(f"  V(SN)  = {ss_vsn*1e3:.3f} mV  (≈ 0V: capacitor retained initial condition)")
print(f"  V(mid) = {ss_vmid:.3f} V  (M0 self-limiting: −V5+VSAT clamp)")
print(f"  I(M0)  = {ss_im0:.3f} nA")
