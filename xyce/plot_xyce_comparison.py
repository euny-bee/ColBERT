import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw = pd.read_csv("igzo_measure.csv", comment="#", names=["vgs","vds","ids"])
raw = raw.dropna()
raw = raw[raw["ids"] > 0]
raw = raw.sort_values("vgs").reset_index(drop=True)

sm  = pd.read_csv("test_smooth.cir.csv")
ex  = pd.read_csv("test_exact.cir.csv")

sm.columns  = ["vgs", "ids"]
ex.columns  = ["vgs", "ids"]

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(raw["vgs"], raw["ids"], "o", color="gray",
            markersize=4, alpha=0.5, label="Measured data")
ax.semilogy(sm["vgs"], sm["ids"].abs(), color="crimson", linewidth=2.2,
            label="Xyce: smooth (logistic)")
ax.semilogy(ex["vgs"], ex["ids"].abs(), color="teal",   linewidth=2.0,
            linestyle="--", label="Xyce: exact (4-region)")

ax.set_xlabel("V$_{GS}$ (V)", fontsize=13)
ax.set_ylabel("|I$_{DS}$| (A)", fontsize=13)
ax.set_xlim(-3, 5)
ax.set_ylim(1e-14, 1e-5)
ax.set_title("IGZO TFT — Xyce Simulation vs Measured Data  (V$_{DS}$=0.5V)", fontsize=12)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("xyce_comparison.png", dpi=150)
print("Saved: xyce_comparison.png")
