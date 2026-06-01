# Phase 3 (L2 search) - V2=0V, Vth=-0.849V (원래 0.151V에서 -1.0V shift)
# 이전 그래프(Vth=-0.349V)에서 Vth를 추가로 -0.5V shift한 경우

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

_available = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _available:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# ── die4 파라미터 (Vth만 shift) ───────────────────────────────────────────────
L     = 6.1120
K     = 2.7319
V0    = -0.8490     # Vth: 원래 0.151V에서 -1.0V shift
B     = -10.7139
VSAT  = 2.88
SLOPE = 4.0332e-5

# ── 회로 조건 ─────────────────────────────────────────────────────────────────
V2     = 0.0         # 저장된 데이터 전압 (V2=0V 경우)
neg_V2 = -V2         # = 0.0V
VDD    = 1.0
clip   = 1e-23

# ── 단방향 전류 모델 ──────────────────────────────────────────────────────────
def I_single(vgs):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v   = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE

# ── 양방향 전류 ───────────────────────────────────────────────────────────────
# M0: VSN0 = V1 + V2 + Vth = V1 + 0 + (-0.849) = V1 - 0.849
def I_M0(v1_arr):
    Vgd = v1_arr + V0 - neg_V2   # = v1 - 0.849
    Vgs = Vgd - VDD               # = v1 - 1.849
    return np.maximum(I_single(Vgd) - I_single(Vgs), 0)

# M3: VSN3 = -V2 + Vth - V1 = 0 + (-0.849) - V1 = -0.849 - V1
def I_M3(v1_arr):
    Vgd = neg_V2 + V0 - v1_arr   # = -0.849 - v1
    Vgs = Vgd - VDD               # = -1.849 - v1
    return np.maximum(I_single(Vgd) - I_single(Vgs), 0)

# ── V1 sweep ──────────────────────────────────────────────────────────────────
v1 = np.linspace(-1.5, 1.5, 300)
im0    = I_M0(v1)
im3    = I_M3(v1)
itotal = im0 + im3

# ── 플롯 ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    f"Dual 3T1C  Phase 3 (L2 search)  —  die4_ovl20_R0\n"
    f"V2={V2}V (stored, -V2={neg_V2}V),  V1 swept  |  Vth={V0}V  (-1.0V shift)",
    fontsize=10)

ax.plot(v1, np.clip(im0,    clip, None), color="steelblue", lw=2, ls="-",
        marker="o", markevery=15, markersize=4,
        label=f"M0  (V1 > -V2={neg_V2}V 일때 ON,  I $\propto$ V1+V2)")
ax.plot(v1, np.clip(im3,    clip, None), color="tomato",    lw=2, ls="-",
        marker="s", markevery=15, markersize=4,
        label=f"M3  (-V2 > V1 일때 ON,  I $\propto$ -(V1+V2))")
ax.plot(v1, np.clip(itotal, clip, None), color="purple",    lw=2.5, ls="-",
        label="I$_{M0}$ + I$_{M3}$  total  $\propto$ |V1 + V2|")

ax.axvline(neg_V2, color="gray", lw=1.2, ls=":", alpha=0.7,
           label=f"V1 = -V2 = {neg_V2}V  (crossover / minimum)")

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-3)
ax.set_xlim(-1.5, 1.5)
ax.set_xlabel("V1  [V]", fontsize=11)
ax.set_ylabel("|I|  [A]", fontsize=11)
ax.set_title(
    f"Log |I| [A]  vs  V1  —  Python bidirectional model\n"
    f"M0: VSN = V1+V2+Vth = V1+({V2:.1f})+({V0:.3f})  |  "
    f"M3: VSN = -V2+Vth-V1 = {neg_V2:.1f}+({V0:.3f})-V1",
    fontsize=8.5)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=8.5)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "dual_phase3_log_die4_v2_0V_vth_m0p85.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")
