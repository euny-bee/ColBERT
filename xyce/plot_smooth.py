import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("igzo_measure.csv", comment="#", names=["vgs", "vds", "ids"])
df = df.dropna()
df = df[df["ids"] > 0]
df = df.sort_values("vgs").reset_index(drop=True)
vgs = df["vgs"].values
ids = df["ids"].values

# ── 단일 로지스틱 함수 fit ────────────────────────────────────────────────────
# log10(Ids) = b + L / (1 + exp(-k*(Vgs - v0)))
# 파라미터: L (전체 폭), k (가파름), v0 (변곡점), b (OFF 레벨)
def logistic(v, L, k, v0, b):
    return b + L / (1 + np.exp(-k * (v - v0)))

# 노이즈 플로어 위 데이터만 fit
mask_fit = ids > 1e-13
p0 = [7.0, 5.0, 0.5, -13.0]   # 초기값: [총높이, 가파름, 변곡점, OFF레벨]
popt, _ = curve_fit(logistic, vgs[mask_fit], np.log10(ids[mask_fit]),
                    p0=p0, maxfev=20000)
L, k, v0, b = popt

print(f"Logistic fit parameters:")
print(f"  L  = {L:.4f}  (log10 range: {b:.2f} → {b+L:.2f})")
print(f"  k  = {k:.4f}  (steepness, SS ≈ {1000/k/np.log10(np.e):.1f} mV/dec at inflection)")
print(f"  v0 = {v0:.4f} V  (inflection point)")
print(f"  b  = {b:.4f}  (OFF level: {10**b:.3e} A)")
print(f"  Ion = {10**(b+L):.3e} A")

# ── 5V까지 포화 유지: v > v_sat에서 L 값 고정 ───────────────────────────────
Ion_data = np.max(ids)
idx_sat  = np.where(ids >= 0.9 * Ion_data)[0][0]
V_sat    = vgs[idx_sat]
Ion_sat  = logistic(V_sat, *popt)   # V_sat에서의 로지스틱 값

def log_ids_model(v):
    return np.where(v < V_sat, logistic(v, L, k, v0, b), Ion_sat)

# ── Plot 데이터 ───────────────────────────────────────────────────────────────
vgs_plot = np.linspace(-3, 5, 3000)
ids_plot = 10 ** log_ids_model(vgs_plot)
gm       = np.gradient(np.log10(ids_plot), vgs_plot)

# R² 계산
log_pred = logistic(vgs[mask_fit], *popt)
log_data = np.log10(ids[mask_fit])
r2 = 1 - np.sum((log_data - log_pred)**2) / np.sum((log_data - log_data.mean())**2)
print(f"  R² = {r2:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(7, 7),
                         gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.semilogy(vgs, ids, "o", color="gray", markersize=4, alpha=0.6, label="Data")
ax.semilogy(vgs_plot, ids_plot, color="crimson", linewidth=2.5,
            label=f"Logistic fit  R²={r2:.3f}")
ax.axvline(v0,    color="purple",     linestyle="--", linewidth=1.0,
           alpha=0.8, label=f"Inflection v0={v0:.3f} V")
ax.axvline(V_sat, color="darkorange", linestyle="--", linewidth=1.0,
           alpha=0.7, label=f"V_sat={V_sat:.2f} V")

eq = (f"$\\log_{{10}}(I_{{DS}}) = {b:.3f} + \\dfrac{{{L:.3f}}}{{1 + e^{{-{k:.3f}(V_{{GS}}-{v0:.3f})}}}}$\n"
      f"$R^2 = {r2:.3f}$")
ax.text(0.02, 0.98, eq, transform=ax.transAxes, fontsize=9,
        va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

ax.set_ylabel("|I$_{DS}$| (A)", fontsize=12)
ax.set_xlim(-3, 5); ax.set_ylim(1e-14, 1e-5)
ax.set_title("Logistic Smooth Model  (V$_{DS}$=0.5V)", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

ax2 = axes[1]
ax2.plot(vgs_plot, gm, color="purple", linewidth=1.8)
ax2.set_xlabel("V$_{GS}$ (V)", fontsize=12)
ax2.set_ylabel("d(log I)/dV\n(dec/V)", fontsize=10)
ax2.set_xlim(-3, 5); ax2.set_ylim(bottom=0)
ax2.set_title("Transconductance (gm) — single smooth peak", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.axvline(v0,    color="purple",     linestyle="--", linewidth=1.0, alpha=0.7)
ax2.axvline(V_sat, color="darkorange", linestyle="--", linewidth=1.0, alpha=0.7)

plt.tight_layout()
plt.savefig("transfer_curve_smooth.png", dpi=150)
plt.show()
print("\nSaved: transfer_curve_smooth.png")
