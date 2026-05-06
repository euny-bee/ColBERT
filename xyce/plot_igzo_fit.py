import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Measured data
df = pd.read_csv("igzo_measure.csv", comment="#", header=None, names=["vgs","vds","ids"])
df = df.dropna()
vgs_meas = df["vgs"].values
ids_meas = df["ids"].values

# Model parameters
L    = 6.9509
K    = 3.3919
V0   = 0.0802
B    = -13.0068
VSAT = 2.34

def model(vgs):
    vgs_c = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (vgs_c - V0))))

vgs_fit = np.linspace(-3, 3, 500)
ids_fit = model(vgs_fit)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("IGZO TFT Transfer Curve — Measurement vs Smooth Logistic Fit\n"
             f"L={L}, K={K}, V0={V0}, B={B}, VSAT={VSAT}V", fontsize=11)

for ax, yscale in zip(axes, ["log", "linear"]):
    ax.plot(vgs_fit, ids_fit, color="tomato", linewidth=2,
            label="Model (smooth logistic)")
    ax.scatter(vgs_meas, ids_meas, color="steelblue", s=20, zorder=5,
               label="Measured (Vds=0.5V)")
    ax.axvline(VSAT, color="gray", linestyle="--", linewidth=1,
               label=f"VSAT = {VSAT}V (clamp)")
    ax.set_xlabel("Vgs [V]", fontsize=11)
    ax.set_ylabel("Ids [A]", fontsize=11)
    ax.set_xlim(-3, 3)
    ax.set_yscale(yscale)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    title = "Log scale" if yscale == "log" else "Linear scale"
    ax.set_title(title, fontsize=10)

plt.tight_layout()
plt.savefig("igzo_fit_comparison.png", dpi=150)
print("Saved: igzo_fit_comparison.png")
