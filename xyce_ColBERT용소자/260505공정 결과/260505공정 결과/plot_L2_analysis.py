"""
L2 Search Analysis — 범용 버전
사용법:
  python plot_L2_analysis.py              # 대화형으로 die 선택
  python plot_L2_analysis.py die4         # 직접 지정
  python plot_L2_analysis.py die4 die1    # 여러 die 비교

출력 (die 당):
  plot_L2_<die>_phase3.png   — Phase 3 L2 곡선 (VDS 1V vs 1.7V)
  plot_L2_<die>_AC.png       — Option A vs C  (Vth shift 비교)
"""

import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit

# ── 폰트 ──────────────────────────────────────────────────────────────────────
_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 1. CSV 파싱 / die 목록
# ══════════════════════════════════════════════════════════════════════════════

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

def discover_dies(data_dir):
    """CSV 파일에서 die 이름 목록 반환 (die 번호 순 정렬)"""
    files = [f for f in os.listdir(data_dir)
             if f.endswith(".csv") and f.startswith("IGZO_TR")]
    die_map = {}   # die_label -> filepath
    for f in files:
        m = re.search(r"(die\d+)", f)
        if m:
            label = m.group(1)
            die_map.setdefault(label, os.path.join(data_dir, f))
    return dict(sorted(die_map.items(),
                       key=lambda x: int(re.search(r"\d+", x[0]).group())))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Logistic 피팅
# ══════════════════════════════════════════════════════════════════════════════

def logistic(vgs, L, K, V0, B):
    return B + L / (1 + np.exp(-K * (vgs - V0)))

def fit_die(filepath):
    """CSV 파일을 읽고 logistic 모델을 피팅 → 파라미터 dict 반환"""
    gate, id_ = parse_b1500_csv(filepath)
    id_abs = np.abs(id_)
    valid = id_abs > 0
    vgs_fit = gate[valid]
    ids_fit = id_abs[valid]
    log_ids = np.log10(ids_fit)

    noise_mask = gate <= -2.0
    ioff_med = max(np.median(id_abs[noise_mask]) if np.any(noise_mask) else 1e-11, 1e-14)
    Ion_log  = np.log10(np.max(ids_fit))
    Ioff_log = np.log10(ioff_med)
    L_init   = Ion_log - Ioff_log
    mid_log  = (Ion_log + Ioff_log) / 2
    V0_init  = vgs_fit[np.argmin(np.abs(log_ids - mid_log))]

    popt, _ = curve_fit(
        logistic, vgs_fit, log_ids,
        p0=[L_init, 5.0, V0_init, Ioff_log],
        bounds=([1, 0.5, -3, -20], [20, 30, 3, -5]),
        maxfev=30000
    )
    L_f, K_f, V0_f, B_f = popt

    sat_mask = id_abs >= 0.9 * np.max(id_abs)
    VSAT_f = gate[sat_mask][0] if np.any(sat_mask) else gate[-1]

    return dict(L=L_f, K=K_f, Vth=V0_f, B=B_f, VSAT=VSAT_f)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 전류 계산 함수
# ══════════════════════════════════════════════════════════════════════════════

def calc_ids(vgs_arr, params, VDS_target=1.7, VDS_meas=1.0):
    """
    logistic 모델 + VDS 보정 → IDS [A]
    params: fit_die() 반환값
    """
    L, K, Vth, B = params["L"], params["K"], params["Vth"], params["B"]

    # logistic IDS at VDS_meas
    ids = 10 ** (B + L / (1 + np.exp(-K * (vgs_arr - Vth))))

    # MOSFET linear/saturation 보정
    VoD = vgs_arr - Vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > VDS_meas, Vo*VDS_meas - VDS_meas**2/2, Vo**2/2)
    It = np.where(Vo > VDS_target, Vo*VDS_target - VDS_target**2/2, Vo**2/2)
    factor[on] = np.where(Im > 0, It / Im, 1.0)

    return ids * factor

# ══════════════════════════════════════════════════════════════════════════════
# 4. Plot 1 — Phase 3 L2 curve (VDS 1V vs 1.7V)
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase3(die_label, params, save_dir):
    Vth = params["Vth"]
    V2  = np.arange(-1.0, 1.0 + 1e-9, 0.05)
    x   = 2 * V2          # V2 - V1 = 2*V2
    VGS = np.abs(x) + Vth

    IDS_1V  = calc_ids(VGS, params, VDS_target=1.0) * 1e6
    IDS_17V = calc_ids(VGS, params, VDS_target=1.7) * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Phase 3 L2 Search  —  {die_label}  (Vth = {Vth:.3f} V),  V2 = -V1\n"
        f"M0/M3: source = 0 V,  drain = VDD",
        fontsize=11
    )

    for ax, yscale in zip(axes, ["log", "linear"]):
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.plot(x, IDS_1V,  "o-",  color="steelblue", ms=4, lw=1.8,
                label="VDS = 1.0 V  (fitting 조건)")
        ax.plot(x, IDS_17V, "s--", color="tomato",    ms=4, lw=1.8,
                label="VDS = 1.7 V  (MOSFET 보정)")
        for vds_val, c in [(1.0, "steelblue"), (1.7, "tomato")]:
            ax.axvline( vds_val, color=c, lw=0.9, ls=":", alpha=0.7)
            ax.axvline(-vds_val, color=c, lw=0.9, ls=":", alpha=0.7)
        if yscale == "log":
            ax.set_yscale("log")
            ax.set_title("Log scale")
        else:
            ax.set_title("Linear scale")
        ax.set_xlabel("V2 - V1  (= 2*V2)  [V]", fontsize=10)
        ax.set_ylabel("IDS  [uA]", fontsize=10)
        ax.set_xlim(-2.2, 2.2)
        ax.set_xticks(np.arange(-2, 2.1, 0.5))
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls="--", alpha=0.35)

    fig.text(0.01, 0.01,
             f"log10(IDS)=B+L/(1+exp(-K*(VGS-V0)))  "
             f"L={params['L']:.3f}, K={params['K']:.3f}, "
             f"V0={Vth:.4f}V, B={params['B']:.3f}",
             fontsize=7.5, color="dimgray")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(save_dir, f"plot_L2_{die_label}_phase3.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Plot 2 — Option A vs C (Vth shift 비교)
# ══════════════════════════════════════════════════════════════════════════════

def plot_AC(die_label, params, save_dir,
            shifts=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    Vth_orig = params["Vth"]
    Vth_vals = [Vth_orig + s for s in shifts]
    labels   = [f"Vth={Vth_orig:.3f}V (원본)"] + \
               [f"+{s:.1f}V  ->  {v:.3f}V" for s, v in zip(shifts[1:], Vth_vals[1:])]
    colors   = ["black", "royalblue", "forestgreen", "gold", "darkorange", "crimson"]
    lstyles  = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,1))]

    # 스윕 포인트
    V2_A = np.arange(-1.0,  1.0 + 1e-9, 0.05)
    x_A  = 2 * V2_A
    V2_C = np.arange(-2.0,  2.0 + 1e-9, 0.05)
    x_C  = V2_C

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"With Vth Compensation (A)  vs  No Compensation (C)  —  {die_label}\n"
        f"Vth_orig = {Vth_orig:.3f} V,  VDS = 1.7 V",
        fontsize=12, fontweight="bold"
    )

    for row, yscale in enumerate(["log", "linear"]):
        ax_A = axes[row, 0]
        ax_C = axes[row, 1]

        # ── Option A ─────────────────────────────────────────────────────────
        ax_A.set_title(f"Option A: With Vth Compensation  ({yscale} scale)", fontsize=10)
        ax_A.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

        for vth, lbl, col, ls in zip(Vth_vals, labels, colors, lstyles):
            p = dict(params, Vth=vth)          # Vth만 교체, 나머지 동일
            VGS = np.abs(x_A) + vth            # VGS - Vth = |2V2| 항상
            ids = calc_ids(VGS, p) * 1e6
            ax_A.plot(x_A, ids, color=col, lw=2.0, ls=ls, label=lbl)

        ax_A.set_xlabel("V2 - V1  [V]", fontsize=10)
        ax_A.set_ylabel("IDS  [uA]", fontsize=10)
        ax_A.set_xlim(-2.2, 2.2)
        ax_A.set_xticks(np.arange(-2, 2.1, 0.5))
        ax_A.legend(fontsize=8, loc="upper center")
        ax_A.grid(True, which="both", ls="--", alpha=0.3)
        if yscale == "log":
            ax_A.set_yscale("log")
            ax_A.text(0.5, 0.45, "모든 커브 완전 겹침\n(Vth 보상 완벽)",
                      transform=ax_A.transAxes, fontsize=9, ha="center",
                      color="navy", style="italic",
                      bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))

        # ── Option C ─────────────────────────────────────────────────────────
        ax_C.set_title(f"Option C: No Compensation  ({yscale} scale)", fontsize=10)
        ax_C.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

        for vth, lbl, col, ls in zip(Vth_vals, labels, colors, lstyles):
            p = dict(params, Vth=vth)
            VGS = np.abs(x_C)                  # VGS = |V2|, Vth 없음
            ids = calc_ids(VGS, p) * 1e6
            ax_C.plot(x_C, ids, color=col, lw=2.0, ls=ls, label=lbl)
            ax_C.axvline( vth, color=col, lw=0.8, ls=":", alpha=0.55)
            ax_C.axvline(-vth, color=col, lw=0.8, ls=":", alpha=0.55)

        ax_C.axvspan(-Vth_vals[-1], Vth_vals[-1],
                     alpha=0.06, color="red",
                     label=f"dead zone (max: +/-{Vth_vals[-1]:.3f}V)")
        ax_C.set_xlabel("V2  [V]", fontsize=10)
        ax_C.set_ylabel("IDS  [uA]", fontsize=10)
        ax_C.set_xlim(-2.2, 2.2)
        ax_C.set_xticks(np.arange(-2, 2.1, 0.5))
        ax_C.legend(fontsize=8, loc="upper center")
        ax_C.grid(True, which="both", ls="--", alpha=0.3)
        if yscale == "log":
            ax_C.set_yscale("log")

    plt.tight_layout()
    out = os.path.join(save_dir, f"plot_L2_{die_label}_AC.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    die_map = discover_dies(DATA_DIR)
    if not die_map:
        print("ERROR: IGZO_TR*.csv 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # 커맨드라인 인자 처리
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if not args:
        # 대화형 선택
        print("\n사용 가능한 die 목록:")
        for i, name in enumerate(die_map):
            print(f"  [{i}] {name}")
        sel = input("\n분석할 die 번호 또는 이름 (쉼표로 여러 개 가능, Enter=전체): ").strip()
        if sel == "":
            targets = list(die_map.keys())
        else:
            targets = []
            for tok in sel.replace(",", " ").split():
                if tok.isdigit():
                    key = list(die_map.keys())[int(tok)]
                else:
                    key = tok if tok in die_map else None
                if key:
                    targets.append(key)
                else:
                    print(f"  경고: '{tok}' 를 찾을 수 없어 건너뜁니다.")
    else:
        targets = [a for a in args if a in die_map]
        missing = [a for a in args if a not in die_map]
        if missing:
            print(f"경고: {missing} 를 찾을 수 없습니다. 사용 가능: {list(die_map.keys())}")

    if not targets:
        print("분석할 die가 없습니다.")
        sys.exit(1)

    print(f"\n분석 대상: {targets}\n")

    for die in targets:
        print(f"[{die}] 피팅 중...")
        params = fit_die(die_map[die])
        print(f"  L={params['L']:.4f}, K={params['K']:.4f}, "
              f"Vth={params['Vth']:.4f}V, B={params['B']:.4f}")
        plot_phase3(die, params, DATA_DIR)
        plot_AC(die, params, DATA_DIR)
        print()

    print("완료!")

if __name__ == "__main__":
    main()
