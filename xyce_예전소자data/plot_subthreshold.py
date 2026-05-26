import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("igzo_measure.csv", comment="#", names=["vgs", "vds", "ids"])
df = df.dropna()
df = df[df["ids"] > 0]
df = df.sort_values("vgs").reset_index(drop=True)
vgs = df["vgs"].values
ids = df["ids"].values

NOISE_THR = 1e-12
SUB_UPPER = 1e-10

Ion_data = np.max(ids)
V_sat    = vgs[np.where(ids >= 0.9 * Ion_data)[0][0]]

ioff_log = np.mean(np.log10(ids[ids < NOISE_THR]))
ioff_val = 10 ** ioff_log

# ── ON polynomial: 1e-10 이상 ~ V_sat ────────────────────────────────────────
first_on_idx = np.where(ids >= SUB_UPPER)[0][0]
V_on_start   = vgs[first_on_idx]
mask_on  = (vgs >= V_on_start) & (vgs < V_sat)
coeffs_on = np.polyfit(vgs[mask_on], np.log10(ids[mask_on]), 3)
poly_on   = np.poly1d(coeffs_on)
Ion_fit   = 10 ** poly_on(V_sat)

# ── Subthreshold 지수 모델: 1e-12 ~ 1e-10 구간 선형 fit ──────────────────────
mask_sub  = (ids >= NOISE_THR) & (ids < SUB_UPPER)
coeffs_sub = np.polyfit(vgs[mask_sub], np.log10(ids[mask_sub]), 1)
poly_sub   = np.poly1d(coeffs_sub)

SS_mV = 1.0 / coeffs_sub[0] * 1000  # mV/dec

# V1: subthreshold 선 ∩ noise floor
V1 = (ioff_log - coeffs_sub[1]) / coeffs_sub[0]

# V2: subthreshold 선 ∩ ON polynomial
diff = coeffs_on.copy()
diff[-2] -= coeffs_sub[0]
diff[-1] -= coeffs_sub[1]
roots = np.roots(diff)
real_roots  = roots[np.isreal(roots)].real
valid_roots = real_roots[(real_roots > V1) & (real_roots < V_sat)]
V2 = float(np.min(valid_roots))

print(f"[Subthreshold] SS = {SS_mV:.1f} mV/dec")
print(f"  V1 (OFF→sub)  = {V1:.3f} V")
print(f"  V2 (sub→ON)   = {V2:.3f} V")
print(f"  V_sat         = {V_sat:.3f} V")
print(f"  OFF={ioff_val:.2e} A,  Ion={Ion_fit:.2e} A")

# ── Piecewise 모델 ────────────────────────────────────────────────────────────
def log_ids_sub(v):
    v = np.atleast_1d(np.array(v, dtype=float))
    result = np.empty_like(v)
    result[v <  V1]                      = ioff_log
    result[(v >= V1) & (v <  V2)]        = poly_sub(v[(v >= V1) & (v <  V2)])
    result[(v >= V2) & (v <  V_sat)]     = poly_on(v[(v >= V2)  & (v <  V_sat)])
    result[v >= V_sat]                   = np.log10(Ion_fit)
    return result

vgs_plot = np.linspace(-3, 5, 3000)
ids_plot = 10 ** log_ids_sub(vgs_plot)
gm       = np.gradient(np.log10(ids_plot), vgs_plot)

fig, axes = plt.subplots(2, 1, figsize=(7, 7),
                         gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.semilogy(vgs, ids, "o", color="gray", markersize=4, alpha=0.6, label="Data")
ax.semilogy(vgs_plot, ids_plot, color="teal", linewidth=2.5,
            label="3-region subthreshold model")
ax.axvline(V1,    color="steelblue",  linestyle="--", linewidth=1.0,
           alpha=0.8, label=f"V1 (OFF→sub) = {V1:.3f} V")
ax.axvline(V2,    color="seagreen",   linestyle="--", linewidth=1.0,
           alpha=0.8, label=f"V2 (sub→ON)  = {V2:.3f} V")
ax.axvline(V_sat, color="darkorange", linestyle="--", linewidth=1.0,
           alpha=0.8, label=f"V_sat        = {V_sat:.2f} V")

ax.set_ylabel("|I$_{DS}$| (A)", fontsize=12)
ax.set_xlim(-3, 5); ax.set_ylim(1e-14, 1e-5)
ax.set_title("3-Region Subthreshold Exponential Model  (V$_{DS}$=0.5V)", fontsize=12)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

sub_eq = (f"Sub:  $\\log_{{10}}(I_{{DS}}) = {coeffs_sub[0]:.3f}\\cdot V_{{GS}}"
          f" + ({coeffs_sub[1]:.3f})$  [SS={SS_mV:.0f} mV/dec]\n"
          f"ON:   $\\log_{{10}}(I_{{DS}}) = {coeffs_on[0]:.3f}\\cdot V_{{GS}}^3"
          f" {coeffs_on[1]:+.3f}\\cdot V_{{GS}}^2"
          f" {coeffs_on[2]:+.3f}\\cdot V_{{GS}} ({coeffs_on[3]:+.3f})$")
ax.text(0.02, 0.98, sub_eq, transform=ax.transAxes, fontsize=8.2,
        va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax2 = axes[1]
ax2.plot(vgs_plot, gm, color="purple", linewidth=1.5)
ax2.set_xlabel("V$_{GS}$ (V)", fontsize=12)
ax2.set_ylabel("d(log I)/dV\n(dec/V)", fontsize=10)
ax2.set_xlim(-3, 5)
ax2.set_title("Transconductance (gm) — 연속성 확인", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.4)
for vl, c in [(V1, "steelblue"), (V2, "seagreen"), (V_sat, "darkorange")]:
    ax2.axvline(vl, color=c, linestyle="--", linewidth=1.0, alpha=0.7)

plt.tight_layout()
plt.savefig("transfer_curve_sub.png", dpi=150)
plt.show()
print("Saved: transfer_curve_sub.png")
