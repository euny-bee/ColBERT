import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("circuit_3t1c.cir.csv")
df.columns = ["time1", "time", "vsn", "vmid", "ivdd"]
df = df[df["time"] >= 0].reset_index(drop=True)

t_ns = df["time"] * 1e9       # ns
vsn  = df["vsn"]
vmid = df["vmid"]
ivdd = df["ivdd"].abs() * 1e9  # nA

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

# ── Region shading ─────────────────────────────────────────────────────────
regions = [
    (0,   100, "Write\n(Vcomp=5V)",  "skyblue",   0.18),
    (100, 200, "Hold\n(both=0V)",     "lightyellow", 0.25),
    (200, 400, "Read\n(Vread=5V)",   "lightgreen",  0.18),
]

for ax in axes:
    for x0, x1, label, color, alpha in regions:
        ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0)

# ── Panel 1: V(SN) ─────────────────────────────────────────────────────────
axes[0].plot(t_ns, vsn, color="steelblue", linewidth=2, label="V(SN) — gate of M0")
axes[0].axhline(2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                label="V1 = 2 V (input)")
axes[0].set_ylabel("V(SN) [V]", fontsize=11)
axes[0].legend(fontsize=9, loc="upper left")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_title("3T1C IGZO TFT — Write / Hold / Read Transient  (V$_{DD}$=5V, C=100fF, V1=2V)",
                  fontsize=11)

# ── Panel 2: V(mid) ────────────────────────────────────────────────────────
axes[1].plot(t_ns, vmid, color="tomato", linewidth=2, label="V(mid) — output node")
axes[1].set_ylabel("V(mid) [V]", fontsize=11)
axes[1].legend(fontsize=9, loc="upper left")
axes[1].grid(True, linestyle="--", alpha=0.4)

# ── Panel 3: Supply current |I(VDD)| ───────────────────────────────────────
axes[2].semilogy(t_ns, ivdd.clip(lower=1e-6), color="purple", linewidth=2,
                 label="|I(V$_{DD}$)| — supply current")
axes[2].set_ylabel("|I$_{DD}$| [nA]", fontsize=11)
axes[2].set_xlabel("Time [ns]", fontsize=11)
axes[2].legend(fontsize=9, loc="upper left")
axes[2].grid(True, which="both", linestyle="--", alpha=0.4)

# Region labels on top panel
for x0, x1, label, _, _ in regions:
    axes[0].text((x0 + x1) / 2, axes[0].get_ylim()[1] * 0.92, label,
                 ha="center", va="top", fontsize=8.5, style="italic",
                 color="dimgray")

plt.xlim(0, 400)
plt.tight_layout()
plt.savefig("3t1c_transient.png", dpi=150)
print("Saved: 3t1c_transient.png")

# ── Summary ────────────────────────────────────────────────────────────────
read = df[df["time"] > 2e-7]
print(f"\n[Read phase steady state]")
print(f"  V(SN)  = {read['vsn'].iloc[-1]:.3f} V  (stored gate voltage)")
print(f"  V(mid) = {read['vmid'].iloc[-1]:.3f} V  (output node)")
print(f"  I(VDD) = {read['ivdd'].abs().iloc[-1]*1e9:.1f} nA  (drive current)")
