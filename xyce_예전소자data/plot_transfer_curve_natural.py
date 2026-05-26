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

# --- 각 region 원래 경계(Vgs=-0.1V) 기준으로 fit ---
V_SPLIT = -0.1

mask_off = vgs < V_SPLIT
mask_on  = vgs >= V_SPLIT

vgs_off, ids_off = vgs[mask_off], ids[mask_off]
vgs_on,  ids_on  = vgs[mask_on],  ids[mask_on]

# OFF: 상수 (log 도메인 평균)
log_ids_off = np.log10(ids_off)
ioff_const  = np.mean(log_ids_off)
ioff_val    = 10 ** ioff_const

# ON: log 도메인 3차 다항식
log_ids_on = np.log10(ids_on)
coeffs_on  = np.polyfit(vgs_on, log_ids_on, 3)
poly_on    = np.poly1d(coeffs_on)

r2_on = 1 - np.sum((log_ids_on - poly_on(vgs_on))**2) / \
            np.sum((log_ids_on - log_ids_on.mean())**2)

# --- 자연 교차점: poly_on(Vgs) = ioff_const 를 만족하는 Vgs ---
# poly_on(x) - ioff_const = 0 의 근을 탐색
roots = np.roots(coeffs_on - np.array([0, 0, 0, ioff_const]))
# 실수 근 중 vgs 범위 내에 있는 것 선택
real_roots = roots[np.isreal(roots)].real
valid_roots = real_roots[(real_roots >= vgs.min()) & (real_roots <= vgs.max())]
v_natural = float(np.min(valid_roots))   # 가장 왼쪽(낮은) 교차점

print(f"자연 교차점: Vgs = {v_natural:.4f} V  (Ids = {10**poly_on(v_natural):.3e} A)")
print(f"\n[OFF]  Ids = {ioff_val:.3e} A (constant)")
print(f"\n[ON]   log10(Ids) = {coeffs_on[0]:.4f}·Vgs³ + {coeffs_on[1]:.4f}·Vgs²"
      f" + {coeffs_on[2]:.4f}·Vgs + ({coeffs_on[3]:.4f})")
print(f"       R² = {r2_on:.4f}")

# --- Fit 선: 교차점에서 연결 ---
vgs_off_fit = np.linspace(vgs.min(), v_natural, 300)
ids_off_fit = np.full_like(vgs_off_fit, ioff_val)

vgs_on_fit  = np.linspace(v_natural, vgs.max(), 300)
ids_on_fit  = 10 ** poly_on(vgs_on_fit)

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))

ax.semilogy(vgs, ids, "o", color="gray", markersize=4, alpha=0.6, label="Data")
ax.semilogy(vgs_off_fit, ids_off_fit, color="steelblue", linewidth=2.5,
            label=f"OFF = {ioff_val:.2e} A (constant)")
ax.semilogy(vgs_on_fit, ids_on_fit, color="tomato", linewidth=2.5,
            label=f"ON fit (3rd)  R²={r2_on:.3f}")

ax.axvline(v_natural, color="green", linestyle="--", linewidth=1.2,
           label=f"Natural boundary = {v_natural:.3f} V")

ax.set_xlabel("V$_{GS}$ (V)", fontsize=13)
ax.set_ylabel("|I$_{DS}$| (A)", fontsize=13)
ax.set_xlim(vgs.min(), vgs.max())
ax.set_ylim(1e-14, 1e-5)
ax.set_title("IGZO TFT Transfer Curve Fitting  (V$_{DS}$=0.5V)", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

off_eq = (f"OFF ($V_{{GS}} < {v_natural:.3f}$ V):\n"
          f"$I_{{DS}} = {ioff_val:.3e}$ A  (constant)")
on_eq  = (f"ON ($V_{{GS}} \\geq {v_natural:.3f}$ V):\n"
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
plt.savefig("transfer_curve_natural.png", dpi=150)
plt.show()
print("\nSaved: transfer_curve_natural.png")
