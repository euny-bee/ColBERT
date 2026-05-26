import pandas as pd
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib.pyplot as plt

df = pd.read_csv("igzo_measure.csv", comment="#", header=None, names=["vgs","vds","ids"])
df = df.dropna()
vgs = df["vgs"].values
ids = df["ids"].values

# Linear region: Vgs >= 2.34V
mask_lin = vgs >= 2.34
vgs_lin = vgs[mask_lin]
ids_lin = ids[mask_lin]

VSAT = 2.34
coeffs = np.polyfit(vgs_lin, ids_lin, 1)
slope, intercept = coeffs[0], coeffs[1]
I_at_VSAT = slope * VSAT + intercept
ids_pred = slope * vgs_lin + intercept
ss_res = np.sum((ids_lin - ids_pred)**2)
ss_tot = np.sum((ids_lin - ids_lin.mean())**2)
r2 = 1 - ss_res/ss_tot

print(f"Linear fit (Vgs >= {VSAT}V):")
print(f"  Slope  = {slope:.6e} A/V  ({slope*1e9:.2f} nA/V)")
print(f"  I(VSAT)= {I_at_VSAT:.4e} A  (fitted line at Vgs=2.34V)")
print(f"  R²     = {r2:.6f}")

# Logistic model params
L, K, V0, B = 6.9509, 3.3919, 0.0802, -13.0068

def logistic(vg):
    vg_c = np.minimum(vg, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (vg_c - V0))))

def model_extended(vg):
    I_log = logistic(np.minimum(vg, VSAT))
    I_lin = np.maximum(0, vg - VSAT) * slope
    return I_log + I_lin

vgs_plot = np.linspace(-3, 5, 1000)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("IGZO Model: Logistic + Linear Extension (data-fitted)\n"
             f"Linear slope = {slope*1e9:.1f} nA/V  (fitted from Vgs=2.34~3V data)", fontsize=11)

for ax, yscale in zip(axes, ["log", "linear"]):
    ax.scatter(vgs, ids, color="steelblue", s=20, zorder=5, label="Measured data")
    ax.plot(vgs_plot, logistic(vgs_plot), color="gray", linewidth=1.5,
            linestyle="--", label="Old model (clamped at VSAT)")
    ax.plot(vgs_plot, model_extended(vgs_plot), color="tomato", linewidth=2,
            label="New model (logistic + linear extension)")
    ax.axvline(VSAT, color="orange", linestyle="--", linewidth=1,
               label=f"VSAT = {VSAT}V")
    ax.set_xlabel("Vgs [V]", fontsize=11)
    ax.set_ylabel("Ids [A]", fontsize=11)
    ax.set_xlim(-3, 5)
    ax.set_yscale(yscale)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_title("Log scale" if yscale == "log" else "Linear scale", fontsize=10)

plt.tight_layout()
plt.savefig("igzo_fit_extended.png", dpi=150)
print("\nSaved: igzo_fit_extended.png")
print(f"\n→ SLOPE parameter for Xyce: {slope:.6e}")
