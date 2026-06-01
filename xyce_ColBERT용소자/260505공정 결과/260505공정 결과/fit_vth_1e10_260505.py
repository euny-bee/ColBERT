"""
260505 공정 IGZO TFT — off/on 분리 피팅 + Vth = 연결점

● Off 모델  (sub-threshold 지수)
    ID_off = max( 10^(sl·VGS + ic), ID_floor )
      - VGS << Vth : 노이즈 바닥(ID_floor)에서 flat
      - VGS ↗ Vth  : 지수 상승
    Vth : 지수선이 1e-10 A 를 외삽 통과하는 점 (음수)

● On 모델   (포화 내장, Vth 에서 연속)
    ID = ith + (Ion−ith)·(1−exp(−α·(VGS−Vth)^β))
    → VGS = Vth 에서 ID = ith (연결), VGS 증가 시 Ion 으로 포화

출력:
    fig1_off_260505.png       — off 피팅 3x3
    fig2_on_260505.png        — on 피팅 3x3
    fig3_combined_260505.png  — off + on 합체 3x3 (전체 VGS 커버)
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import linregress
from scipy.optimize import curve_fit

# ── 폰트 ──────────────────────────────────────────────────────────────────────
_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ITH  = 1e-10   # Vth 기준 전류
VDS  = 1.0     # 드레인 바이어스

# ── CSV 파싱 (DataName: gate, IG, ID → parts[3]=ID) ──────────────────────────
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

# ══════════════════════════════════════════════════════════
#  Off 피팅
# ══════════════════════════════════════════════════════════
def fit_off(vgs, ids_abs, ith=ITH):
    """
    서브스레숄드 상승 구간에서 log10(ID)=sl·VGS+ic 피팅.
    Vth = (log10(ith)-ic)/sl  (외삽)
    """
    id_min  = ids_abs.min()
    id_floor = id_min
    idx_min  = np.argmin(ids_abs)

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
    ss  = 1000.0 / sl        # mV/dec
    return vth, ss, sl, ic, id_floor

def off_curve(vgs_arr, sl, ic, id_floor):
    return np.maximum(10.0 ** (sl * vgs_arr + ic), id_floor)

# ══════════════════════════════════════════════════════════
#  On 피팅  (포화 내장 모델, Vth 에서 자동 연속)
# ══════════════════════════════════════════════════════════
def fit_on(vgs, ids_abs, vth, ith=ITH):
    """
    ID = ith + (ion−ith)·(1−exp(−α·(VGS−Vth)^β))
    • VGS = Vth  → ID = ith  (연속 보장)
    • VGS → ∞   → ID → ion  (포화 내장 → 3V 발산 없음)
    log10(ID) 스케일로 curve_fit → turn-on 저전류 ~ 고전류 균등 최적화
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

    mid_vgs = vgs_m[np.argmin(np.abs(ids_m - np.sqrt(ith * ion)))]
    alpha0  = max(1.0 / max(mid_vgs - vth, 0.1), 0.1)

    try:
        popt, _ = curve_fit(
            model_log, vgs_m, log_ids,
            p0=[alpha0, 1.5],
            bounds=([1e-3, 0.2], [1e4, 8.0]),
            maxfev=15000
        )
        alpha, beta = popt
    except RuntimeError:
        alpha, beta = alpha0, 1.5

    yp = model_log(vgs_m, alpha, beta)
    ss_res = np.sum((log_ids - yp) ** 2)
    ss_tot = np.sum((log_ids - log_ids.mean()) ** 2)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return alpha, beta, r2, mask

def on_curve(vgs_arr, alpha, beta, vth, ith, ion):
    """
    ID = ith + (ion−ith)·(1−exp(−α·(VGS−Vth)^β))
    VGS = Vth 에서 ID = ith,  VGS → ∞ 에서 ID → ion.
    """
    delta = np.maximum(vgs_arr - vth, 0.0)
    return ith + (ion - ith) * (1.0 - np.exp(-alpha * delta ** beta))

# ── 파일 수집 ─────────────────────────────────────────────────────────────────
all_files = sorted(
    [f for f in os.listdir(DATA_DIR)
     if f.endswith(".csv") and f.startswith("IGZO_TR")],
    key=lambda f: (int(re.search(r"die(\d+)", f).group(1)),
                   int(re.search(r"_R(\d+)",  f).group(1)))
)
files = [f for f in all_files if "die8_ovl20_R0" not in f]

# ── 분석 ──────────────────────────────────────────────────────────────────────
die_data = []
print(f"\n{'Die':<22} {'Vth (V)':>9} {'SS (mV/dec)':>12} "
      f"{'Ion (A)':>11} {'R²_on':>7}")
print("─" * 65)

for fname in files:
    vgs, id_ = parse_b1500_csv(os.path.join(DATA_DIR, fname))
    ids_abs  = np.abs(id_)
    label    = die_label(fname)

    vth, ss, sl_off, ic_off, id_floor = fit_off(vgs, ids_abs)
    if vth is None:
        vth, ss = 0.0, float("nan")

    alpha_on, beta_on, r2_on, on_mask = fit_on(vgs, ids_abs, vth)
    ion = ids_abs.max()

    die_data.append(dict(
        label=label, vgs=vgs, ids_abs=ids_abs,
        vth=vth, ss=ss,
        sl_off=sl_off, ic_off=ic_off, id_floor=id_floor,
        alpha_on=alpha_on, beta_on=beta_on, r2_on=r2_on,
        ion=ion, ioff=ids_abs.min(),
    ))

    s_s = f"{ss:.0f}" if ss == ss else "N/A"
    print(f"{label:<22} {vth:>+9.3f} {s_s:>12} {ion:>11.2e} {r2_on:>7.4f}")

# ── 공통 ──────────────────────────────────────────────────────────────────────
VGS_FULL = np.linspace(-3.2, 3.2, 4000)

def _fig(title):
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=13)
    return fig, axes

def _style(ax, title, xlim, ylim):
    ax.set_xlim(*xlim);  ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("VGS (V)", fontsize=9)
    ax.set_ylabel("|ID| (A)", fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.tick_params(labelsize=8)

def _vth_lines(ax, vth):
    ax.axhline(ITH, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.axvline(vth, color="purple", ls="--", lw=1.2,
               label=f"Vth = {vth:+.3f} V")

def _box(ax, text):
    ax.text(0.97, 0.05, text, transform=ax.transAxes,
            fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))

def _plot_off(ax, d, xlim):
    if d['sl_off'] is None:
        return
    vf = VGS_FULL[(VGS_FULL >= xlim[0]) & (VGS_FULL <= xlim[1])]
    yf = off_curve(vf, d['sl_off'], d['ic_off'], d['id_floor'])
    ax.semilogy(vf, yf.clip(1e-15), "-", color="tomato",
                lw=2, alpha=0.9, label="Off fit")

def _plot_on(ax, d, xlim):
    if d['alpha_on'] is None:
        return
    vf  = VGS_FULL[(VGS_FULL >= xlim[0]) & (VGS_FULL <= xlim[1])]
    yf  = on_curve(vf, d['alpha_on'], d['beta_on'], d['vth'], ITH, d['ion'])
    ok  = ~np.isnan(yf) & (yf > 1e-15)
    ax.semilogy(vf[ok], yf[ok], "-", color="darkorange",
                lw=2, alpha=0.9, label="On fit")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Off 영역 피팅
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = _fig("IGZO TFT — Off 영역 피팅   260505 공정  VDS = 1 V")

for i, d in enumerate(die_data):
    ax  = ax1.flatten()[i]
    xlr = min(d['vth'] + 1.5, 2.5)
    xlim = (-3.0, xlr)

    m = (d['vgs'] >= -3.0) & (d['vgs'] <= xlr)
    ax.semilogy(d['vgs'][m], d['ids_abs'][m].clip(1e-15),
                "o", color="steelblue", ms=3, alpha=0.7, label="Data")

    _plot_off(ax, d, xlim)
    _vth_lines(ax, d['vth'])
    _style(ax, d['label'], xlim, (1e-13, 1e-4))
    ax.legend(fontsize=7, loc="upper left")

    info = f"Vth = {d['vth']:+.3f} V"
    if d['ss'] == d['ss']:
        info += f"\nSS = {d['ss']:.0f} mV/dec"
    _box(ax, info)

plt.tight_layout()
p1 = os.path.join(DATA_DIR, "fig1_off_260505.png")
fig1.savefig(p1, dpi=150);  print(f"\nSaved → {p1}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — On 영역 피팅
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = _fig("IGZO TFT — On 영역 피팅   260505 공정  VDS = 1 V")

for i, d in enumerate(die_data):
    ax  = ax2.flatten()[i]
    xll = max(d['vth'] - 0.5, -2.0)
    xlim = (xll, 3.0)

    m = (d['vgs'] >= xll) & (d['vgs'] <= 3.0)
    ax.semilogy(d['vgs'][m], d['ids_abs'][m].clip(1e-15),
                "o", color="steelblue", ms=3, alpha=0.7, label="Data")

    _plot_on(ax, d, xlim)
    _vth_lines(ax, d['vth'])
    _style(ax, d['label'], xlim, (1e-11, 1e-3))
    ax.legend(fontsize=7, loc="upper left")

    b_str = f"\nβ = {d['beta_on']:.2f}" if d['beta_on'] is not None else ""
    _box(ax, f"Vth = {d['vth']:+.3f} V\nIon = {d['ion']:.2e} A\nR²_on = {d['r2_on']:.3f}{b_str}")

plt.tight_layout()
p2 = os.path.join(DATA_DIR, "fig2_on_260505.png")
fig2.savefig(p2, dpi=150);  print(f"Saved → {p2}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Combined: off + on 연결  (전체 VGS 커버)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = _fig(
    "IGZO TFT — Off + On 피팅 합체 (전체 VGS 커버)   260505 공정  VDS = 1 V"
)

for i, d in enumerate(die_data):
    ax   = ax3.flatten()[i]
    vth  = d['vth']

    ax.semilogy(d['vgs'], d['ids_abs'].clip(1e-15),
                "o", color="steelblue", ms=2.5, alpha=0.55, label="Data")

    _plot_off(ax, d, (-3.0, vth + 0.02))
    _plot_on(ax, d,  (vth - 0.02, 3.0))

    _vth_lines(ax, vth)
    _style(ax, d['label'], (-3.0, 3.0), (1e-13, 1e-3))
    ax.legend(fontsize=7, loc="upper left")

    ion_ioff = d['ion'] / max(d['ioff'], 1e-15)
    info = f"Vth = {vth:+.3f} V"
    if d['ss'] == d['ss']:
        info += f"\nSS = {d['ss']:.0f} mV/dec"
    info += f"\nIon/Ioff = {ion_ioff:.1e}"
    _box(ax, info)

plt.tight_layout()
p3 = os.path.join(DATA_DIR, "fig3_combined_260505.png")
fig3.savefig(p3, dpi=150);  print(f"Saved → {p3}")

plt.show()

# ── 요약 ──────────────────────────────────────────────────────────────────────
print(f"\n{'─'*52}")
print(f"  Vth 요약  (|ID| = {ITH:.0e} A 기준)")
print(f"{'─'*52}")
vths = [d['vth'] for d in die_data]
for d in die_data:
    s_s = f"{d['ss']:.0f}" if d['ss'] == d['ss'] else "N/A"
    print(f"  {d['label']:<22}  Vth = {d['vth']:+.3f} V   SS = {s_s} mV/dec")
print(f"{'─'*52}")
print(f"  평균  Vth = {np.mean(vths):+.3f} V")
print(f"  최소  Vth = {np.min(vths):+.3f} V  ({die_data[int(np.argmin(vths))]['label']})")
print(f"  최대  Vth = {np.max(vths):+.3f} V  ({die_data[int(np.argmax(vths))]['label']})")
print(f"  표준편차  = {np.std(vths):.3f} V")
