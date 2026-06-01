# die4_ovl20_R0 - dual 3T1C Phase 3 (L2 search) 시뮬레이션
# 실제 회로 변수 기준:
#   Phase 2: M0 source line = -V2 = 1V (V2=-1V 저장)
#   Phase 3: VSN0 = V1 + V2 + Vth = V1 + (-1) + Vth  (= V1 - 0.849)
#            VSN3 = -V2 + Vth - V1 = 1.0 + Vth - V1   (= 1.151 - V1)
#   M0 ON when V1 > -V2,  M3 ON when -V2 > V1
# die4 fit params: L=6.112, K=2.732, V0=0.151, B=-10.714, VSAT=2.88, SLOPE=4.033e-5

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

# ── die4 파라미터 ─────────────────────────────────────────────────────────────
L     = 6.1120
K     = 2.7319
V0    = 0.1510      # Vth
B     = -10.7139
VSAT  = 2.88
SLOPE = 4.0332e-5

# ── 실제 회로 변수 ─────────────────────────────────────────────────────────────
V2     = -1.0        # 저장된 데이터 전압 (Phase 2에서 -V2 = 1V 인가)
neg_V2 = -V2         # Phase 2 source line = -V2 = 1.0V
VDD    = 1.0         # 전원 전압
clip   = 1e-23

# ── 단방향 전류 모델 ──────────────────────────────────────────────────────────
def I_single(vgs):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v   = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE

# ── 양방향 전류 ───────────────────────────────────────────────────────────────
# M0: Phase 3 VSN0 = V1 + V2 + Vth = V1 + V0 + V2  (= V1 - 0.849)
#     drain=VML=0V (위), source=VDD (아래)
#     Vgd wrt VML(0V): V1 + V0 + V2  (= VSN0)
#     Vgs wrt VDD:     V1 + V0 + V2 - VDD
def I_M0(v1_arr):
    Vgd = v1_arr + V0 - neg_V2
    Vgs = v1_arr + V0 - neg_V2 - VDD
    return np.maximum(I_single(Vgd) - I_single(Vgs), 0)

# M3: Phase 3 VSN3 = -V2 + Vth - V1 = neg_V2 + V0 - V1  (= 1.151 - V1)
#     drain=VML=0V (위), source=VDD (아래)
#     Vgd wrt VML(0V): neg_V2 + V0 - v1  (= VSN3)
#     Vgs wrt VDD:     V0 - v1
def I_M3(v1_arr):
    Vgd = neg_V2 + V0 - v1_arr
    Vgs = Vgd - VDD          # = neg_V2 + V0 - v1_arr - VDD
    return np.maximum(I_single(Vgd) - I_single(Vgs), 0)

# ── V1 sweep ──────────────────────────────────────────────────────────────────
v1 = np.linspace(-1, 2, 300)

im0    = I_M0(v1)
im3    = I_M3(v1)
itotal = im0 + im3

# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
print("die4_ovl20_R0 - Dual 3T1C Phase 3 (L2 search)")
print(f"  L={L}, K={K}, V0(Vth)={V0}, B={B}")
print(f"  V2={V2}V (stored data) -> -V2={neg_V2}V (Phase2 source line)")
print(f"  VDD={VDD}V, VML=0V (match line)")
print(f"\n{'V1':>6}  {'I_M0(A)':>12}  {'I_M3(A)':>12}  {'I_total(A)':>12}")
print("-" * 50)
for idx in range(0, len(v1), 30):
    print(f"{v1[idx]:>6.2f}  {im0[idx]:>12.4e}  {im3[idx]:>12.4e}  {itotal[idx]:>12.4e}")

# ── 플롯 ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    f"Dual 3T1C  Phase 3 (L2 search)  —  die4_ovl20_R0\n"
    f"V2={V2}V (stored, -V2={neg_V2}V),  V1 swept  |  Vth={V0}V",
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
           label=f"V1 = -V2 = {neg_V2}V  (crossover)")

ax.set_yscale("log")
ax.set_ylim(1e-14, 1e-3)
ax.set_xlim(-1, 2)
ax.set_xlabel("V1  [V]", fontsize=11)
ax.set_ylabel("|I|  [A]", fontsize=11)
ax.set_title(
    f"Log |I| [A]  vs  V1  —  Python bidirectional model\n"
    f"M0: VSN = V1+V2+Vth = V1+({V2:.1f})+{V0}  |  "
    f"M3: VSN = -V2+Vth-V1 = {neg_V2:.1f}+{V0}-V1",
    fontsize=8.5)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=8.5)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "dual_phase3_log_die4.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"\nSaved: {out}")
