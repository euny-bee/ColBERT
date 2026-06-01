import numpy as np
from scipy.optimize import curve_fit
import os

def parse_b1500_csv(filepath):
    gate, id_ = [], []
    in_data = False
    sweep_done = False
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("DataName"):
                if in_data:
                    sweep_done = True
                in_data = True
                continue
            if sweep_done:
                break
            if not in_data or not line.startswith("DataValue"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                gate.append(float(parts[1]))
                id_.append(float(parts[3]))
            except ValueError:
                continue
    return np.array(gate), np.array(id_)

def logistic(vgs, L, K, V0, B):
    return B + L / (1 + np.exp(-K * (vgs - V0)))

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
fname = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "die4" in f and f.startswith("IGZO_TR")][0]
print("File:", fname)
gate, id_ = parse_b1500_csv(os.path.join(DATA_DIR, fname))
id_abs = np.abs(id_)
valid = id_abs > 0
vgs_fit = gate[valid]
ids_fit = id_abs[valid]
log_ids = np.log10(ids_fit)
noise_mask = gate <= -2.0
ioff_med = np.median(id_abs[noise_mask])
Ion_log = np.log10(np.max(ids_fit))
Ioff_log = np.log10(ioff_med)
L_init = Ion_log - Ioff_log
mid_log = (Ion_log + Ioff_log) / 2
V0_init = vgs_fit[np.argmin(np.abs(log_ids - mid_log))]
popt, _ = curve_fit(logistic, vgs_fit, log_ids, p0=[L_init, 5.0, V0_init, Ioff_log],
    bounds=([1, 0.5, -3, -20], [20, 30, 3, -5]), maxfev=30000)
L_f, K_f, V0_f, B_f = popt
Ion_data = np.max(id_abs)
sat_mask = id_abs >= 0.9 * Ion_data
VSAT_f = gate[sat_mask][0]
print(f"L={L_f:.6f}")
print(f"K={K_f:.6f}")
print(f"V0(Vth)={V0_f:.6f}")
print(f"B={B_f:.6f}")
print(f"VSAT={VSAT_f:.4f}")
print(f"ioff_med={ioff_med:.4e}")
