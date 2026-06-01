"""
원래 첫 번째 fig3_combined 버전 재생성 스크립트
On 모델:
  die4 만  → ID = ITH + (Ion−ITH)·(1−exp(−α·(VGS−Vth)^β))  (포화 내장)
  나머지 8개 → ID = A·(VGS−Vth)^n + ITH                    (원래 power law)
출력: fig3_original_260505.png
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import linregress
from scipy.optimize import curve_fit

_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ITH = 1e-10

def parse_b1500_csv(fp):
    gate, id_ = [], []
    in_data = sweep_done = False
    with open(fp, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("DataName"):
                if in_data: sweep_done = True
                in_data = True; continue
            if sweep_done: break
            if not in_data or not line.startswith("DataValue"): continue
            p = line.strip().split(",")
            if len(p) < 4: continue
            try:
                gate.append(float(p[1]))
                id_.append(float(p[3]))
            except ValueError:
                continue
    return np.array(gate), np.array(id_)

def die_label(fn):
    m = re.search(r"(die\d+_ovl\d+_R\d+)", fn)
    return m.group(1) if m else fn

# ── Off 피팅 ─────────────────────────────────────────────────────────────────
def fit_off(vgs, ids_abs, ith=ITH):
    id_min = ids_abs.min()
    id_floor = id_min
    idx_min = np.argmin(ids_abs)
    lo = max(1.5 * id_min, 5e-12)
    hi = 1e-8
    mask = (ids_abs >= lo) & (ids_abs <= hi) & (vgs >= vgs[idx_min] - 0.12)
    if mask.sum() < 4:
        mask = (ids_abs >= lo) & (ids_abs <= hi)
    if mask.sum() < 2:
        return None, None, None, None, id_floor
    sl, ic, _, _, _ = linregress(vgs[mask], np.log10(ids_abs[mask]))
    if sl <= 0:
        return None, None, sl, ic, id_floor
    vth = (np.log10(ith) - ic) / sl
    ss  = 1000.0 / sl
    return vth, ss, sl, ic, id_floor

def off_curve(vgs_arr, sl, ic, id_floor):
    return np.maximum(10.0 ** (sl * vgs_arr + ic), id_floor)

# ── On 피팅: 원래 power law (die4 제외 8개) ──────────────────────────────────
def fit_on_pow(vgs, ids_abs, vth, ith=ITH):
    ion  = ids_abs.max()
    mask = (vgs > vth + 0.05) & (ids_abs > 10 * ith) & (ids_abs < 0.95 * ion)
    if mask.sum() < 4:
        mask = (vgs > vth + 0.01) & (ids_abs > 5 * ith) & (ids_abs < 0.99 * ion)
    if mask.sum() < 2:
        return None, None, 0.0, mask
    delta = vgs[mask] - vth
    y = np.log10(np.maximum(ids_abs[mask] - ith, 1e-20))
    x = np.log10(delta)
    n, logA, _, _, _ = linregress(x, y)
    yp = n * x + logA
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return n, 10.0 ** logA, r2, mask

def pow_curve(vgs_arr, n, A, vth, ith=ITH):
    return A * np.maximum(vgs_arr - vth, 0.0) ** n + ith

# ── On 피팅: 포화 모델 (die4 전용) ───────────────────────────────────────────
def fit_on_sat(vgs, ids_abs, vth, ith=ITH):
    """
    ID = ith + (Ion−ith)·(1−exp(−α·(VGS−Vth)^β))
    log10(ID) 스케일 curve_fit → S-curve 형태 직접 최적화
    """
    ion  = ids_abs.max()
    mask = (vgs > vth + 0.02) & (ids_abs > 5 * ith)
    if mask.sum() < 4:
        mask = (vgs > vth) & (ids_abs > 2 * ith)
    if mask.sum() < 2:
        return None, None, 0.0, mask

    vgs_m   = vgs[mask]
    ids_m   = ids_abs[mask]
    log_ids = np.log10(ids_m)

    def model_log(vg, alpha, beta):
        delta = np.maximum(vg - vth, 1e-8)
        val   = ith + (ion - ith) * (1.0 - np.exp(-alpha * delta ** beta))
        return np.log10(np.maximum(val, 1e-20))

    # 초기값: log-log 회귀로 β 추정 후 α 보정
    delta_i = np.maximum(vgs_m - vth, 1e-6)
    y_i     = np.log10(np.maximum(ids_m - ith, 1e-20))
    n0, lA0, _, _, _ = linregress(np.log10(delta_i), y_i)
    beta0   = max(min(n0, 7.0), 1.0)
    # α 초기값: 중간 전류(√(ITH·Ion))에 도달하는 VGS 기준
    mid_ids = np.sqrt(ith * ion)
    mid_idx = np.argmin(np.abs(ids_m - mid_ids))
    delta_mid = max(vgs_m[mid_idx] - vth, 0.1)
    alpha0  = 1.0 / (delta_mid ** beta0)

    try:
        popt, _ = curve_fit(
            model_log, vgs_m, log_ids,
            p0=[alpha0, beta0],
            bounds=([1e-6, 0.5], [1e3, 8.0]),
            maxfev=20000
        )
        alpha, beta = popt
    except RuntimeError:
        alpha, beta = alpha0, beta0

    yp = model_log(vgs_m, alpha, beta)
    ss_res = np.sum((log_ids - yp) ** 2)
    ss_tot = np.sum((log_ids - log_ids.mean()) ** 2)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return alpha, beta, r2, mask

def sat_curve(vgs_arr, alpha, beta, vth, ith, ion):
    delta = np.maximum(vgs_arr - vth, 0.0)
    return ith + (ion - ith) * (1.0 - np.exp(-alpha * delta ** beta))

# ── 파일 수집 ─────────────────────────────────────────────────────────────────
all_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f.startswith("IGZO_TR")],
    key=lambda f: (int(re.search(r"die(\d+)", f).group(1)),
                   int(re.search(r"_R(\d+)", f).group(1)))
)
files = [f for f in all_files if "die8_ovl20_R0" not in f]

# ── 분석 ─────────────────────────────────────────────────────────────────────
die_data = []
for fname in files:
    vgs, id_ = parse_b1500_csv(os.path.join(DATA_DIR, fname))
    ids_abs  = np.abs(id_)
    label    = die_label(fname)

    vth, ss, sl_off, ic_off, id_floor = fit_off(vgs, ids_abs)
    if vth is None:
        vth, ss = 0.0, float("nan")

    ion = ids_abs.max()

    if "die4" in label:
        # die4: 포화 모델 (S-curve 맞춤)
        alpha_on, beta_on, r2_on, _ = fit_on_sat(vgs, ids_abs, vth)
        n_on, A_on = None, None
        print(f"{label:<22}  [SAT]  α={alpha_on:.4e}  β={beta_on:.3f}  R²={r2_on:.4f}")
    else:
        # 나머지: 원래 power law
        n_on, A_on, r2_on, _ = fit_on_pow(vgs, ids_abs, vth)
        alpha_on, beta_on = None, None
        print(f"{label:<22}  [POW]  n={n_on:.3f}  R²={r2_on:.4f}")

    die_data.append(dict(
        label=label, vgs=vgs, ids_abs=ids_abs,
        vth=vth, ss=ss,
        sl_off=sl_off, ic_off=ic_off, id_floor=id_floor,
        n_on=n_on, A_on=A_on,
        alpha_on=alpha_on, beta_on=beta_on,
        r2_on=r2_on, ion=ion, ioff=ids_abs.min(),
    ))

# ── 플롯 ─────────────────────────────────────────────────────────────────────
VGS_FULL = np.linspace(-3.2, 3.2, 4000)

fig3, ax3 = plt.subplots(3, 3, figsize=(14, 12))
fig3.suptitle(
    "IGZO TFT — Off + On 피팅   260505 공정  VDS = 1 V\n"
    "(die4: 포화 모델 / 나머지: Power Law)",
    fontsize=12)

for i, d in enumerate(die_data):
    ax  = ax3.flatten()[i]
    vth = d['vth']

    ax.semilogy(d['vgs'], d['ids_abs'].clip(1e-15),
                "o", color="steelblue", ms=2.5, alpha=0.55, label="Data")

    # Off fit
    if d['sl_off'] is not None:
        vf = VGS_FULL[(VGS_FULL >= -3.0) & (VGS_FULL <= vth + 0.02)]
        yf = off_curve(vf, d['sl_off'], d['ic_off'], d['id_floor'])
        ax.semilogy(vf, yf.clip(1e-15), "-", color="tomato",
                    lw=2, alpha=0.9, label="Off fit")

    # On fit
    vf_on = VGS_FULL[(VGS_FULL >= vth - 0.02) & (VGS_FULL <= 3.0)]
    if d['n_on'] is not None:
        yf = pow_curve(vf_on, d['n_on'], d['A_on'], vth)
        ok = ~np.isnan(yf) & (yf > 1e-15)
        ax.semilogy(vf_on[ok], yf[ok], "-", color="darkorange",
                    lw=2, alpha=0.9, label="On fit")
    elif d['alpha_on'] is not None:
        yf = sat_curve(vf_on, d['alpha_on'], d['beta_on'], vth, ITH, d['ion'])
        ok = ~np.isnan(yf) & (yf > 1e-15)
        ax.semilogy(vf_on[ok], yf[ok], "-", color="darkorange",
                    lw=2, alpha=0.9, label="On fit")

    ax.axhline(ITH, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.axvline(vth, color="purple", ls="--", lw=1.2,
               label=f"Vth = {vth:+.3f} V")

    ax.set_xlim(-3.0, 3.0);  ax.set_ylim(1e-13, 1e-3)
    ax.set_title(d['label'], fontsize=10)
    ax.set_xlabel("VGS (V)", fontsize=9);  ax.set_ylabel("|ID| (A)", fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper left")

    ion_ioff = d['ion'] / max(d['ioff'], 1e-15)
    info = f"Vth = {vth:+.3f} V"
    if d['ss'] == d['ss']:
        info += f"\nSS = {d['ss']:.0f} mV/dec"
    info += f"\nIon/Ioff = {ion_ioff:.1e}"
    if d['n_on'] is not None:
        info += f"\nn = {d['n_on']:.2f}"
    elif d['beta_on'] is not None:
        info += f"\nβ = {d['beta_on']:.2f}"
    ax.text(0.97, 0.05, info, transform=ax.transAxes,
            fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))

plt.tight_layout()
out = os.path.join(DATA_DIR, "fig3_original_260505.png")
fig3.savefig(out, dpi=150)
print(f"\nSaved → {out}")
plt.show()
