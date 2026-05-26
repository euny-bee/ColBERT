import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("circuit_phase1.cir.csv")
df.columns = ["t1", "time", "vsn", "vmid", "im0s"]
df = df[df["time"] >= 0].reset_index(drop=True)

t_ns  = df["time"] * 1e9
vsn   = df["vsn"]
vmid  = df["vmid"]
im0_nA = df["im0s"].abs() * 1e9   # |I(M0)| in nA

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
fig.suptitle("Phase 1:  V1=V2=V3=3V,  V4=0V,  V5=−3V\n"
             "Data store & compensation  (C=100fF)", fontsize=11)

# ── V(SN) ──────────────────────────────────────────────────
axes[0].plot(t_ns, vsn, color="steelblue", linewidth=2)
axes[0].set_ylabel("V(SN) [V]", fontsize=11)
axes[0].set_ylim(-0.5, 0.5)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                label="0 V reference")
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_title("Storage Node Voltage  (≈ 0 V,  numerical noise ~10⁻²⁶ V)", fontsize=10)

# ── I(M0) ──────────────────────────────────────────────────
axes[1].plot(t_ns, im0_nA, color="tomato", linewidth=2)
axes[1].set_ylabel("|I$_{M0}$| [nA]", fontsize=11)
axes[1].set_xlabel("Time [ns]", fontsize=11)
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].set_title("M0 Drive Current", fontsize=10)

# Annotate steady state
ss_vsn = vsn.iloc[-1] * 1e3
ss_im0 = im0_nA.iloc[-1]
axes[0].text(0.97, 0.15, f"Steady: {ss_vsn:.2f} mV",
             transform=axes[0].transAxes, ha="right", fontsize=9,
             color="steelblue",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[1].text(0.97, 0.15, f"Steady: {ss_im0:.1f} nA",
             transform=axes[1].transAxes, ha="right", fontsize=9,
             color="tomato",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("phase1_result.png", dpi=150)
print("Saved: phase1_result.png")
print(f"\n[Phase 1 steady state]")
print(f"  V(SN)  = {ss_vsn:.3f} mV  (→ 0V: threshold compensation 완료)")
print(f"  I(M0)  = {ss_im0:.1f} nA  (Vgs_M0 = V(SN)-V5 = 0-(-3) = 3V)")
print(f"  V(mid) = {df['vmid'].iloc[-1]*1e3:.3f} mV")
