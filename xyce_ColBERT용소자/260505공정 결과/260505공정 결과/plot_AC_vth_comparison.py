"""
Option A (With Vth Compensation) vs Option C (No Compensation)
Vth shift: +0.0, +0.1, +0.2, +0.5, +1.0V from original 0.151V
VDS = 1.7V (MOSFET correction applied from 1V fitting)

Option A:
  V2 = -V1, V2 in [-1, 1] step 0.05V
  x-axis: V2 - V1 = 2*V2  in [-2, 2]
  VGS = |2V2| + Vth_new,  model V0 = Vth_new
  -> VGS - V0 = |2V2|  always  => all curves overlap

Option C (No Compensation):
  VSN = V2,  V2 in [-2, 2] step 0.05V
  x-axis: V2
  VGS = |V2|,  model V0 = Vth_new
  -> dead zone: |V2| < Vth_new  (widens as Vth increases)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ── 폰트 ──────────────────────────────────────────────────────────────────────
_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# ── die4 logistic 파라미터 (VDS=1V fitting) ───────────────────────────────────
L_f =  6.112020
K_f =  2.731949
B_f = -10.713911
Vth_orig = 0.151045

# ── Vth 값 설정 ───────────────────────────────────────────────────────────────
shifts      = [0.0, 0.1, 0.2, 0.5, 1.0]
Vth_vals    = [Vth_orig + s for s in shifts]
shift_labels = [
    f"Vth = {Vth_orig:.3f}V (원본)",
    f"+0.1V  ->  {Vth_vals[1]:.3f}V",
    f"+0.2V  ->  {Vth_vals[2]:.3f}V",
    f"+0.5V  ->  {Vth_vals[3]:.3f}V",
    f"+1.0V  ->  {Vth_vals[4]:.3f}V",
]
colors    = ["black", "royalblue", "forestgreen", "darkorange", "crimson"]
linestyle = ["-",    "--",         "-.",          ":",          (0,(3,1,1,1))]

# ── 모델 함수 ─────────────────────────────────────────────────────────────────
def logistic_ids(vgs, vth):
    """IDS [A],  VDS=1V logistic model"""
    return 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - vth))))

def vds_correction(vgs_arr, vth, VDS_t=1.7, VDS_m=1.0):
    """MOSFET linear/saturation 보정: IDS(VDS_t) / IDS(VDS_m)"""
    VoD = vgs_arr - vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > VDS_m, Vo*VDS_m - VDS_m**2/2, Vo**2/2)
    It = np.where(Vo > VDS_t, Vo*VDS_t - VDS_t**2/2, Vo**2/2)
    factor[on] = np.where(Im > 0, It / Im, 1.0)
    return factor

def calc_ids(vgs_arr, vth):
    ids = logistic_ids(vgs_arr, vth)
    cf  = vds_correction(vgs_arr, vth)
    return ids * cf   # [A]

# ── 스윕 포인트 ───────────────────────────────────────────────────────────────
# Option A: V2 in [-1, 1]
V2_A = np.arange(-1.0, 1.0 + 1e-9, 0.05)
x_A  = 2 * V2_A               # V2 - V1 = 2*V2

# Option C: V2 in [-2, 2]
V2_C = np.arange(-2.0, 2.0 + 1e-9, 0.05)
x_C  = V2_C                    # x = V2 directly

# ── 플롯 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "With Vth Compensation (A)  vs  No Compensation (C)\n"
    f"die4  Vth_orig = {Vth_orig:.3f} V,  VDS = 1.7 V",
    fontsize=12, fontweight="bold"
)

for row, yscale in enumerate(["log", "linear"]):
    ax_A = axes[row, 0]
    ax_C = axes[row, 1]

    # ── Option A ──────────────────────────────────────────────────────────────
    ax_A.set_title(f"Option A: With Vth Compensation  ({yscale} scale)", fontsize=10)
    ax_A.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    for vth, lbl, col, ls in zip(Vth_vals, shift_labels, colors, linestyle):
        VGS = np.abs(x_A) + vth          # compensation: VGS - Vth = |2V2|
        ids = calc_ids(VGS, vth) * 1e6   # uA
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

    # ── Option C ──────────────────────────────────────────────────────────────
    ax_C.set_title(f"Option C: No Compensation  ({yscale} scale)", fontsize=10)
    ax_C.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    for vth, lbl, col, ls in zip(Vth_vals, shift_labels, colors, linestyle):
        VGS = np.abs(x_C)                # no compensation: VGS = |V2|
        ids = calc_ids(VGS, vth) * 1e6   # uA
        ax_C.plot(x_C, ids, color=col, lw=2.0, ls=ls, label=lbl)
        # dead zone 경계 표시
        ax_C.axvline( vth, color=col, lw=0.8, ls=":", alpha=0.55)
        ax_C.axvline(-vth, color=col, lw=0.8, ls=":", alpha=0.55)

    # dead zone 음영 (가장 큰 Vth 기준)
    ax_C.axvspan(-Vth_vals[-1], Vth_vals[-1],
                 alpha=0.06, color="red", label=f"dead zone (max: +/-{Vth_vals[-1]:.3f}V)")

    ax_C.set_xlabel("V2  [V]", fontsize=10)
    ax_C.set_ylabel("IDS  [uA]", fontsize=10)
    ax_C.set_xlim(-2.2, 2.2)
    ax_C.set_xticks(np.arange(-2, 2.1, 0.5))
    ax_C.legend(fontsize=8, loc="upper center")
    ax_C.grid(True, which="both", ls="--", alpha=0.3)
    if yscale == "log":
        ax_C.set_yscale("log")

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "plot_AC_vth_comparison.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")
