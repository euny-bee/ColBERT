import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Load measured data ──────────────────────────────────────────────────
df = pd.read_csv("igzo_measure.csv", comment="#", header=None,
                 names=["vgs","vds","ids"]).dropna()
vgs_meas = df["vgs"].values.astype(float)
ids_meas = df["ids"].values.astype(float)

# ── Logistic model (in log10 space) ────────────────────────────────────
def logistic_log(vgs, L, K, V0, B):
    return B + L / (1 + np.exp(-K * (vgs - V0)))

def fit_model(vgs_data, ids_data, label):
    """Fit logistic + find VSAT + fit linear slope above VSAT."""
    log_ids = np.log10(ids_data)

    # Initial guess from original fit
    p0 = [6.9509, 3.3919, np.median(vgs_data[ids_data > 1e-9]), -13.0068]
    bounds = ([0, 0, -10, -20], [15, 20, 10, 0])

    popt, _ = curve_fit(logistic_log, vgs_data, log_ids, p0=p0,
                        bounds=bounds, maxfev=10000)
    L, K, V0, B = popt

    # Predicted log10(I) on fine grid
    vgs_fine = np.linspace(vgs_data.min(), vgs_data.max(), 5000)
    log_pred = logistic_log(vgs_fine, *popt)

    # VSAT: where d(log I)/dVgs drops below 1% of its max
    dlogI = np.gradient(log_pred, vgs_fine)
    peak  = dlogI.max()
    after_peak = np.where(vgs_fine > vgs_fine[dlogI.argmax()])[0]
    sat_idx = after_peak[np.where(dlogI[after_peak] < 0.01 * peak)[0]]
    VSAT = float(vgs_fine[sat_idx[0]]) if len(sat_idx) > 0 else vgs_data.max()

    # Linear fit above VSAT (in linear Ids, not log)
    mask_lin = vgs_data >= VSAT
    slope = 0.0
    if mask_lin.sum() >= 2:
        c = np.polyfit(vgs_data[mask_lin], ids_data[mask_lin], 1)
        slope = c[0]

    # R² on log scale
    log_obs = np.log10(ids_data)
    log_fit = logistic_log(vgs_data, *popt)
    ss_res  = np.sum((log_obs - log_fit)**2)
    ss_tot  = np.sum((log_obs - log_obs.mean())**2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n[{label}]")
    print(f"  L={L:.4f}, K={K:.4f}, V0={V0:.4f} V, B={B:.4f}")
    print(f"  VSAT={VSAT:.3f} V,  SLOPE={slope:.4e} A/V")
    print(f"  R² (log scale) = {r2:.4f}")

    return dict(L=L, K=K, V0=V0, B=B, VSAT=VSAT, SLOPE=slope, r2=r2)

def eval_model(vgs, p):
    vgs_c = np.minimum(vgs, p['VSAT'])
    I_log = 10 ** (p['B'] + p['L'] / (1 + np.exp(-p['K'] * (vgs_c - p['V0']))))
    I_lin = np.maximum(0, vgs - p['VSAT']) * p['SLOPE']
    return I_log + I_lin

# ── Three datasets: original, -0.5V shift, +0.5V shift ─────────────────
datasets = [
    ("Original  (Vth ≈ +0.08V)",  vgs_meas,        ids_meas, "steelblue",  "o"),
    ("Vth − 0.5V (Vth ≈ −0.42V)", vgs_meas - 0.5,  ids_meas, "tomato",     "s"),
    ("Vth + 0.5V (Vth ≈ +0.58V)", vgs_meas + 0.5,  ids_meas, "forestgreen","^"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("IGZO Transfer Curve — Vth Shift ±0.5V with Logistic Fitting", fontsize=12)

vgs_plot = np.linspace(-4, 4, 2000)
params_all = []

for label, vgs_d, ids_d, color, marker in datasets:
    p = fit_model(vgs_d, ids_d, label)
    params_all.append((label, p, color, marker))

    I_fit = eval_model(vgs_plot, p)

    for ax, yscale in zip(axes, ["log", "linear"]):
        ax.scatter(vgs_d, ids_d, color=color, s=15, alpha=0.5,
                   marker=marker, zorder=4)
        ax.plot(vgs_plot, I_fit, color=color, linewidth=2,
                label=f"{label}  (V0={p['V0']:.3f}V, R²={p['r2']:.4f})")
        ax.axvline(p['V0'],   color=color, linestyle=":",  linewidth=0.8, alpha=0.5)
        ax.axvline(p['VSAT'], color=color, linestyle="-.", linewidth=0.8, alpha=0.5)

for ax, yscale in zip(axes, ["log", "linear"]):
    ax.set_yscale(yscale)
    ax.set_xlabel("Vgs [V]", fontsize=11)
    ax.set_ylabel("Ids [A]", fontsize=11)
    ax.set_xlim(-4, 4)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_title("Log scale" if yscale=="log" else "Linear scale", fontsize=10)

plt.tight_layout()
plt.savefig("igzo_vth_shift_fit.png", dpi=150)
print("\nSaved: igzo_vth_shift_fit.png")
