import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("igzo_measure.csv", comment="#", names=["vgs", "vds", "ids"])
df = df.dropna()
df = df[df["ids"] > 0]
df = df.sort_values("vgs").reset_index(drop=True)

vgs = df["vgs"].values
ids = df["ids"].values

# --- Region split: Vgs = -0.1V 기준 ---
V_BOUNDARY = -0.1

mask_off = vgs < V_BOUNDARY
mask_on  = vgs >= V_BOUNDARY

vgs_off, ids_off = vgs[mask_off], ids[mask_off]
vgs_on,  ids_on  = vgs[mask_on],  ids[mask_on]

# --- OFF: 상수 (log 도메인 평균) ---
log_ids_off  = np.log10(ids_off)
ioff_const   = np.mean(log_ids_off)       # log10 평균
ioff_val     = 10 ** ioff_const           # 실제 전류값

# --- ON fit: log10(Ids) = a*Vgs^3 + ... (log 도메인 3차) ---
log_ids_on = np.log10(ids_on)
coeffs_on  = np.polyfit(vgs_on, log_ids_on, 3)
poly_on    = np.poly1d(coeffs_on)

r2_on = 1 - np.sum((log_ids_on - poly_on(vgs_on))**2) / \
            np.sum((log_ids_on - log_ids_on.mean())**2)


# --- Fit 선 생성 ---
vgs_off_fit = np.linspace(vgs.min(), V_BOUNDARY, 300)
ids_off_fit = np.full_like(vgs_off_fit, ioff_val)

vgs_on_fit  = np.linspace(V_BOUNDARY, vgs.max(), 300)
ids_on_fit  = 10 ** poly_on(vgs_on_fit)

# --- Print results ---
print(f"OFF/ON 경계: Vgs = {V_BOUNDARY} V")
print(f"\n[OFF]  Ids = {ioff_val:.3e} A  (상수, log 평균 = {ioff_const:.4f})")
print(f"\n[ON]   log10(Ids) = {coeffs_on[0]:.4f}·Vgs³ + {coeffs_on[1]:.4f}·Vgs²"
      f" + {coeffs_on[2]:.4f}·Vgs + ({coeffs_on[3]:.4f})")
print(f"       R² = {r2_on:.4f}")
print(f"\nON fit at boundary: {10**poly_on(V_BOUNDARY):.3e} A  (OFF constant: {ioff_val:.3e} A)")

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))

ax.semilogy(vgs, ids, "o", color="gray", markersize=4, alpha=0.6, label="Data")
ax.semilogy(vgs_off_fit, ids_off_fit, color="steelblue", linewidth=2.5,
            label=f"OFF = {ioff_val:.2e} A (constant)")
ax.semilogy(vgs_on_fit, ids_on_fit, color="tomato", linewidth=2.5,
            label=f"ON fit (3rd)  R²={r2_on:.3f}")

ax.axvline(V_BOUNDARY, color="green", linestyle="--", linewidth=1.2,
           label=f"Boundary = {V_BOUNDARY} V")

ax.set_xlabel("V$_{GS}$ (V)", fontsize=13)
ax.set_ylabel("|I$_{DS}$| (A)", fontsize=13)
ax.set_xlim(vgs.min(), vgs.max())
ax.set_ylim(1e-14, 1e-5)
ax.set_title("IGZO TFT Transfer Curve Fitting  (V$_{DS}$=0.5V)", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

off_eq = (f"OFF ($V_{{GS}} < {V_BOUNDARY}$ V):\n"
          f"$I_{{DS}} = {ioff_val:.3e}$ A  (constant)")
on_eq  = (f"ON ($V_{{GS}} \\geq {V_BOUNDARY}$ V):\n"
          f"$\\log_{{10}}(I_{{DS}}) = {coeffs_on[0]:.3f}\\cdot V_{{GS}}^3"
          f" {coeffs_on[1]:+.3f}\\cdot V_{{GS}}^2"
          f" {coeffs_on[2]:+.3f}\\cdot V_{{GS}} ({coeffs_on[3]:+.3f})$\n"
          f"$R^2 = {r2_on:.3f}$")

ax.text(0.02, 0.98, off_eq, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", color="steelblue",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.text(0.02, 0.84, on_eq, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", color="tomato",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("transfer_curve_const.png", dpi=150)
plt.show()
print("\nSaved: transfer_curve_const.png")
