"""
Phase 3 L2 search - die4 logistic model
V2 = -V1, V2: -1V ~ 1V (step 0.05V)
X축: V2 - V1 = 2*V2 (-2V ~ 2V)
Y축: IDS (M0 or M3, active transistor)

M0: source=0V, drain=VDD
Phase 3: VSN = V2 - V1 + Vth = 2V2 + Vth  →  VGS_M0 = 2V2 + Vth
M3: symmetric,  VGS_M3 = -2V2 + Vth = |2V2| + Vth  (when V2 < 0)

VDS=1V  : logistic 모델 직접 적용 (fitting 조건)
VDS=1.7V: MOSFET linear/saturation 보정 적용
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트
_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# ── die4 logistic 파라미터 (VDS=1V fitting) ────────────────────────────────
L_f   =  6.112020
K_f   =  2.731949
Vth   =  0.151045   # = V0
B_f   = -10.713911

def logistic_ids(vgs):
    """log10(IDS) → IDS [A],  VDS=1V fitting 기준"""
    return 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - Vth))))

def vds_correction(vgs_arr, VDS_target, VDS_meas=1.0):
    """
    MOSFET linear/saturation 보정 계수 (배열 입력)
    IDS(VDS_target) = IDS(VDS_meas) * correction
    """
    VoD = vgs_arr - Vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0

    VoD_on = VoD[on]
    # measured side
    I_m = np.where(VoD_on > VDS_meas,
                   VoD_on * VDS_meas - VDS_meas**2 / 2,
                   VoD_on**2 / 2)
    # target side
    I_t = np.where(VoD_on > VDS_target,
                   VoD_on * VDS_target - VDS_target**2 / 2,
                   VoD_on**2 / 2)
    factor[on] = np.where(I_m > 0, I_t / I_m, 1.0)
    return factor

# ── 스윕 설정 ──────────────────────────────────────────────────────────────
V2   = np.arange(-1.0, 1.0 + 1e-9, 0.05)
V1   = -V2
x    = V2 - V1          # = 2*V2,  -2V ~ +2V

# VGS of active transistor at each point:
#   V2 > 0  →  M0 ON,  VGS = 2V2 + Vth
#   V2 < 0  →  M3 ON,  VGS = -2V2 + Vth = |2V2| + Vth
VGS  = np.abs(x) + Vth  # 항상 |2V2| + Vth (대칭)

# ── 전류 계산 ──────────────────────────────────────────────────────────────
IDS_1V  = logistic_ids(VGS)
cf_17   = vds_correction(VGS, VDS_target=1.7, VDS_meas=1.0)
IDS_17V = IDS_1V * cf_17

# ── 플롯 ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Phase 3 L2 Search — die4 (Vth = {Vth:.3f} V),  V2 = −V1\n"
    f"M0/M3: source = 0 V,  drain = VDD",
    fontsize=11
)

colors = {"1V": "steelblue", "1.7V": "tomato"}

for ax, yscale in zip(axes, ["log", "linear"]):
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-",  alpha=0.3)

    ax.plot(x, IDS_1V * 1e6,  "o-",
            color=colors["1V"],  markersize=4, linewidth=1.8,
            label=f"VDS = 1.0 V  (fitting 조건)")
    ax.plot(x, IDS_17V * 1e6, "s--",
            color=colors["1.7V"], markersize=4, linewidth=1.8,
            label=f"VDS = 1.7 V  (MOSFET 보정 적용)")

    # 경계선: VGS-Vth = VDS (saturation 경계)
    for vds_val, c in [(1.0, colors["1V"]), (1.7, colors["1.7V"])]:
        x_sat = vds_val  # |x| = VDS 인 지점 (VoD = VDS)
        ax.axvline( x_sat, color=c, linewidth=0.9, linestyle=":", alpha=0.7)
        ax.axvline(-x_sat, color=c, linewidth=0.9, linestyle=":", alpha=0.7)

    if yscale == "log":
        ax.set_yscale("log")
        ax.set_title("Log scale")
    else:
        ax.set_title("Linear scale")

    ax.set_xlabel("V₂ − V₁  (= 2·V₂)  [V]", fontsize=10)
    ax.set_ylabel("IDS  [μA]", fontsize=10)
    ax.set_xlim(-2.2, 2.2)
    ax.set_xticks(np.arange(-2, 2.1, 0.5))
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    # 영역 레이블
    y_top = ax.get_ylim()[1] if yscale == "linear" else None
    ax.text(-1.5, 0.93, "M3 active\n(V₂ < 0)", transform=ax.transAxes,
            fontsize=8, ha="center", color="gray",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.text( 0.75, 0.93, "M0 active\n(V₂ > 0)", transform=ax.transAxes,
            fontsize=8, ha="center", color="gray",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# 파라미터 주석
fig.text(0.01, 0.01,
         f"Logistic model: log₁₀(IDS) = B + L/(1+exp(−K·(VGS−V0)))\n"
         f"L={L_f:.3f}, K={K_f:.3f}, V0(Vth)={Vth:.4f} V, B={B_f:.3f}  |  "
         f"점선: saturation 경계 (|V₂−V₁| = VDS)",
         fontsize=7.5, color="dimgray")

plt.tight_layout(rect=[0, 0.05, 1, 1])

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_phase3_L2_die4.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")

# ── 수치 확인 ──────────────────────────────────────────────────────────────
print(f"\n{'x=V2-V1':>10}  {'V2':>6}  {'VGS':>8}  {'IDS_1V(uA)':>13}  {'IDS_1.7V(uA)':>14}  {'보정계수':>8}")
print("─" * 70)
for i, (xi, v2i, vgsi, i1, i17, cf) in enumerate(
        zip(x, V2, VGS, IDS_1V*1e6, IDS_17V*1e6, cf_17)):
    print(f"{xi:>10.2f}  {v2i:>6.2f}  {vgsi:>8.4f}  {i1:>13.4e}  {i17:>14.4e}  {cf:>8.4f}")
