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

# ── 1. V_lower: 데이터가 noise floor 위로 처음 올라오는 지점 ─────────────────
NOISE_THR = 1e-12
first_rise_idx = np.where(ids >= NOISE_THR)[0][0]
V_lower   = vgs[first_rise_idx]            # ≈ -0.18V (데이터 기반, 자동 탐지)

# ── 2. 포화점: 데이터가 Ion의 90%에 처음 도달하는 Vgs ───────────────────────
Ion_data = np.max(ids)
idx_sat  = np.where(ids >= 0.9 * Ion_data)[0][0]
V_sat    = vgs[idx_sat]
Ion_fit  = None  # poly_on 계산 후 결정

print(f"V_lower      : {V_lower:.3f} V  (first data point above {NOISE_THR:.0e} A)")
print(f"Saturation   : V_sat = {V_sat:.3f} V")

# ── 3. ON fit: V_lower ~ V_sat 구간 ──────────────────────────────────────────
mask_rise = (vgs >= V_lower) & (vgs < V_sat)
vgs_rise  = vgs[mask_rise]
log_rise  = np.log10(ids[mask_rise])

coeffs_on = np.polyfit(vgs_rise, log_rise, 3)
poly_on   = np.poly1d(coeffs_on)

r2_on = 1 - np.sum((log_rise - poly_on(vgs_rise))**2) / \
            np.sum((log_rise - log_rise.mean())**2)

# OFF 레벨 = 실제 noise floor 평균
ioff_log = np.mean(np.log10(ids[ids < NOISE_THR]))
ioff_val = 10 ** ioff_log

# 포화 전류 = ON 다항식의 V_sat 지점 값 → 상단 연결점 정확히 일치
Ion_fit = 10 ** poly_on(V_sat)

print(f"OFF level    : Ioff = {ioff_val:.3e} A  (noise floor mean)")
print(f"\n[ON]  log10(Ids) = {coeffs_on[0]:.4f}·Vgs³ + {coeffs_on[1]:.4f}·Vgs²"
      f" + {coeffs_on[2]:.4f}·Vgs + ({coeffs_on[3]:.4f})")
print(f"      R² = {r2_on:.4f}")

# ── 5. Plot 선 ────────────────────────────────────────────────────────────────
vgs_off_plt  = np.linspace(vgs.min(), V_lower, 300)
ids_off_plt  = np.full_like(vgs_off_plt, ioff_val)

vgs_rise_plt = np.linspace(V_lower, V_sat, 300)
ids_rise_plt = 10 ** poly_on(vgs_rise_plt)

vgs_sat_plt  = np.linspace(V_sat, 5.0, 300)   # 5V까지 연장
ids_sat_plt  = np.full_like(vgs_sat_plt, Ion_fit)

# ── 6. Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.semilogy(vgs, ids, "o", color="gray", markersize=4, alpha=0.6, label="Data")

ax.semilogy(vgs_off_plt, ids_off_plt, color="steelblue", linewidth=2.5,
            label=f"OFF = {ioff_val:.2e} A (constant)")
ax.semilogy(vgs_rise_plt, ids_rise_plt, color="tomato", linewidth=2.5,
            label=f"ON fit (3rd)  R²={r2_on:.3f}")
ax.semilogy(vgs_sat_plt, ids_sat_plt, color="darkorange", linewidth=2.5,
            label=f"Saturation = {Ion_fit:.2e} A")

ax.axvline(V_lower, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.7,
           label=f"V_lower = {V_lower:.3f} V")
ax.axvline(V_sat,   color="darkorange", linestyle="--", linewidth=1.0, alpha=0.7,
           label=f"V_sat = {V_sat:.2f} V")

on_eq = (f"ON ($V_{{GS}}$: {V_lower:.2f} ~ {V_sat:.2f} V):\n"
         f"$\\log_{{10}}(I_{{DS}}) = {coeffs_on[0]:.3f}\\cdot V_{{GS}}^3"
         f" {coeffs_on[1]:+.3f}\\cdot V_{{GS}}^2"
         f" {coeffs_on[2]:+.3f}\\cdot V_{{GS}} ({coeffs_on[3]:+.3f})$\n"
         f"$R^2 = {r2_on:.3f}$")

ax.text(0.02, 0.98,
        f"OFF ($V_{{GS}} < {V_lower:.3f}$ V):  $I_{{DS}} = {ioff_val:.2e}$ A",
        transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
        color="steelblue",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.text(0.02, 0.88, on_eq, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", color="tomato",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.text(0.02, 0.64,
        f"SAT ($V_{{GS}} > {V_sat:.2f}$ V):  $I_{{DS}} = {Ion_fit:.2e}$ A",
        transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
        color="darkorange",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_xlabel("V$_{GS}$ (V)", fontsize=13)
ax.set_ylabel("|I$_{DS}$| (A)", fontsize=13)
ax.set_xlim(-3, 5)
ax.set_ylim(1e-14, 1e-5)
ax.set_title("IGZO TFT Transfer Curve Fitting  (V$_{DS}$=0.5V)", fontsize=12)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("transfer_curve_v2.png", dpi=150)
plt.show()
print("\nSaved: transfer_curve_v2.png")
