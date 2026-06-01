"""
260505 공정 IGZO TFT — 9개 die 전체 logistic + linear extension fitting
  log10(Ids) = B + L / (1 + exp(-K*(Vgs - V0)))   [Vgs < VSAT]
  Ids = logistic(VSAT) + (Vgs - VSAT) * SLOPE       [Vgs >= VSAT]

VDS = 1V (drain constant), sweep: gate -3V ~ 3V

출력:
  fit_logistic_all_260505.png  — 3x3 subplot (data + fit)
  콘솔: 각 die의 L, K, V0(=Vth), B, VSAT, SLOPE, R^2
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit

_available = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _available:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 파싱 (첫 번째 sweep만) ────────────────────────────────────────────────────
def parse_b1500_csv(filepath):
    gate, id_ = [], []
    in_data = False
    sweep_done = False
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("DataName"):
                if in_data:
                    sweep_done = True   # 두 번째 DataName → 첫 sweep 끝
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

def die_label(filename):
    m = re.search(r"(die\d+_ovl\d+_R\d+)", filename)
    return m.group(1) if m else filename

# ── logistic 모델 ─────────────────────────────────────────────────────────────
def logistic(vgs, L, K, V0, B):
    return B + L / (1 + np.exp(-K * (vgs - V0)))

# ── 파일 수집 ─────────────────────────────────────────────────────────────────
all_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f.startswith("IGZO_TR")],
    key=lambda f: (int(re.search(r"die(\d+)", f).group(1)),
                   int(re.search(r"_R(\d+)", f).group(1)))
)
files = [f for f in all_files if "die8_ovl20_R0" not in f]

# ── 피팅 + 플롯 ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle("IGZO TFT Logistic Fitting — 260505 공정  (VDS = 1 V)", fontsize=13)

results = []

print(f"\n{'Die':<22} {'L':>7} {'K':>7} {'V0(Vth)':>9} {'B':>9} {'VSAT':>7} {'SLOPE':>12} {'R2':>7}")
print("─" * 90)

for idx, fname in enumerate(files):
    ax = axes.flatten()[idx]
    fpath = os.path.join(DATA_DIR, fname)
    gate, id_ = parse_b1500_csv(fpath)
    label = die_label(fname)

    id_abs = np.abs(id_)

    # ── OFF 구간 중앙값 (B 초기값) ───────────────────────────────────────────
    noise_mask = gate <= -2.0
    ioff_med = np.median(id_abs[noise_mask]) if np.any(noise_mask) else 1e-11
    ioff_med = max(ioff_med, 1e-14)

    # ── 전체 구간 fit (ON + OFF 모두) ─────────────────────────────────────────
    # 0 또는 음수 값 제거
    valid = id_abs > 0
    vgs_fit = gate[valid]
    ids_fit = id_abs[valid]
    log_ids = np.log10(ids_fit)

    Ion_log  = np.log10(np.max(ids_fit))
    Ioff_log = np.log10(ioff_med)
    L_init   = Ion_log - Ioff_log
    # V0 초기값: log10(Ids) = 중간값이 되는 Vgs
    mid_log  = (Ion_log + Ioff_log) / 2
    V0_init  = vgs_fit[np.argmin(np.abs(log_ids - mid_log))]

    try:
        popt, _ = curve_fit(
            logistic, vgs_fit, log_ids,
            p0=[L_init, 5.0, V0_init, Ioff_log],
            bounds=([1, 0.5, -3, -20], [20, 30, 3, -5]),
            maxfev=30000
        )
        L_f, K_f, V0_f, B_f = popt
    except RuntimeError:
        print(f"  {label}: fit 실패 — 초기값으로 대체")
        L_f, K_f, V0_f, B_f = L_init, 5.0, V0_init, Ioff_log

    # ── VSAT 찾기 (Ion의 90% 도달 지점) ─────────────────────────────────────
    Ion_data = np.max(id_abs)
    sat_mask = id_abs >= 0.9 * Ion_data
    VSAT_f = gate[sat_mask][0] if np.any(sat_mask) else gate[-1]

    # ── linear extension slope (Vgs >= VSAT) ─────────────────────────────────
    lin_mask = gate >= VSAT_f
    if np.sum(lin_mask) >= 2:
        c = np.polyfit(gate[lin_mask], id_abs[lin_mask], 1)
        SLOPE_f = max(c[0], 0)
    else:
        SLOPE_f = 0.0

    # ── R² 계산 (전체 구간) ──────────────────────────────────────────────────
    log_pred = logistic(vgs_fit, L_f, K_f, V0_f, B_f)
    ss_res = np.sum((log_ids - log_pred) ** 2)
    ss_tot = np.sum((log_ids - log_ids.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    results.append(dict(label=label, L=L_f, K=K_f, V0=V0_f, B=B_f,
                        VSAT=VSAT_f, SLOPE=SLOPE_f, R2=r2,
                        ioff=ioff_med))

    print(f"{label:<22} {L_f:>7.4f} {K_f:>7.4f} {V0_f:>9.4f} {B_f:>9.4f} "
          f"{VSAT_f:>7.3f} {SLOPE_f:>12.4e} {r2:>7.4f}")

    # ── 플롯용 모델 곡선 ─────────────────────────────────────────────────────
    vgs_plot = np.linspace(-3, 3, 1000)

    def model_ids(vg):
        log_v = logistic(np.minimum(vg, VSAT_f), L_f, K_f, V0_f, B_f)
        I_log = 10 ** log_v
        I_lin = np.maximum(0, vg - VSAT_f) * SLOPE_f
        return (I_log + I_lin).clip(1e-14)

    ids_model = model_ids(vgs_plot)

    # ── subplot 그리기 ────────────────────────────────────────────────────────
    ax.semilogy(gate, id_abs.clip(1e-14), "o",
                color="steelblue", markersize=2.5, alpha=0.6, label="Data")
    ax.semilogy(vgs_plot, ids_model,
                color="tomato", linewidth=2, label=f"Fit  R²={r2:.3f}")
    ax.axvline(V0_f, color="purple", linewidth=1.0, linestyle="--", alpha=0.8,
               label=f"Vth={V0_f:.3f}V")
    ax.axvline(VSAT_f, color="orange", linewidth=0.9, linestyle=":", alpha=0.7,
               label=f"VSAT={VSAT_f:.2f}V")

    ax.set_title(label, fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(1e-13, 1e-3)
    ax.set_xlabel("VGS (V)", fontsize=9)
    ax.set_ylabel("|ID| (A)", fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper left")

    ax.text(0.97, 0.05,
            f"K={K_f:.2f}\nV0={V0_f:.3f}V\nB={B_f:.2f}",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
out = os.path.join(DATA_DIR, "fit_logistic_all_260505.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"\nSaved: {out}")

# ── Vth 요약 ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*35}")
print(f"  Vth (V0) 요약")
print(f"{'─'*35}")
vths = [r["V0"] for r in results]
for r in results:
    print(f"  {r['label']:<22}  Vth = {r['V0']:+.4f} V")
print(f"{'─'*35}")
print(f"  평균  Vth = {np.mean(vths):+.4f} V")
print(f"  최소  Vth = {np.min(vths):+.4f} V  ({results[np.argmin(vths)]['label']})")
print(f"  최대  Vth = {np.max(vths):+.4f} V  ({results[np.argmax(vths)]['label']})")
print(f"  표준편차 = {np.std(vths):.4f} V")
