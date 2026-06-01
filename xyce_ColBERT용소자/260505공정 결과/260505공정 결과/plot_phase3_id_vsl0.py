"""
Phase 3 (2번): Id(M0) vs V_SL0
- Phase 2에서 저장: VSN = Vth (M0 diode-connected, V_BL0=0V)
- Phase 3 cap boosting: VSN = Vth + V_SL0
- M0: gate=VSN, source=V_BL0=VDD=1.7V (fixed)
- Vgs(M0) = VSN - VDD = (Vth + V_SL0) - VDD = V_SL0 - 1.549V
- Effective drive = Vgs - Vth = V_SL0 - VDD  (Vth 소거!)
"""
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

# ── die4_ovl20_R0 파라미터 ────────────────────────────────────────────────────
L     = 6.1120
K     = 2.7319
V0    = 0.1510      # Vth
B     = -10.7139
VSAT  = 2.88
SLOPE = 4.0332e-5

# ── 회로 조건 ─────────────────────────────────────────────────────────────────
VDD    = 1.7        # M0 source (V_BL0 in Phase 3)
V_BL0_p2 = 0.0     # Phase 2 M0 source (Vth 저장 시)

# Phase 2 저장값: VSN = Vth + V_BL0_p2 = V0 + 0 = V0
VSN_stored = V0 + V_BL0_p2   # = 0.151V

# ── 단방향 전류 모델 (포화 영역 가정, Vds 충분히 큼) ─────────────────────────
def I_single(vgs):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v   = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE

# ── V_SL0 sweep ───────────────────────────────────────────────────────────────
vsl0 = np.linspace(-1.0, 1.0, 500)

# cap boosting: VSN = VSN_stored + V_SL0
VSN  = VSN_stored + vsl0           # = V0 + V_SL0

# M0 Vgs = VSN - VDD = V_SL0 + V0 - VDD
Vgs_M0 = VSN - VDD                 # = V_SL0 - 1.549V

Id = I_single(Vgs_M0)

# ── 플롯 (log + linear 2패널) ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Phase 3 (2번):  Id(M0) vs V_SL0\n"
    f"VSN = Vth + V_SL0 = {V0}+V_SL0,  "
    f"Vgs(M0) = VSN - VDD = V_SL0+{V0}-{VDD} = V_SL0-{VDD-V0:.3f}V,  "
    f"VDD = {VDD}V",
    fontsize=9.5)

# --- 왼쪽: log scale ---
ax = axes[0]
ax.semilogy(vsl0, Id, color="steelblue", lw=2.5)
ax.set_xlabel("V_SL0  [V]", fontsize=11)
ax.set_ylabel("Id  [A]  (log)", fontsize=11)
ax.set_title("Log scale", fontsize=10)
ax.set_xlim(-1, 1)
ax.set_ylim(1e-14, 1e-6)
ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.6, label="V_SL0 = 0V")
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(fontsize=9)

# 유효 drive 표시
ax2 = ax.twiny()
ax2.set_xlim(-1 - VDD, 1 - VDD)   # V_SL0 - VDD
ax2.set_xlabel("V_SL0 - VDD  [V]  (effective drive, Vth 소거)", fontsize=8.5, color="tomato")
ax2.tick_params(axis="x", labelcolor="tomato")

# --- 오른쪽: linear scale ---
ax = axes[1]
ax.plot(vsl0, Id * 1e12, color="tomato", lw=2.5)
ax.set_xlabel("V_SL0  [V]", fontsize=11)
ax.set_ylabel("Id  [pA]  (linear)", fontsize=11)
ax.set_title("Linear scale", fontsize=10)
ax.set_xlim(-1, 1)
ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.6, label="V_SL0 = 0V")
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=9)

# 주요 수치 annotation
for vsl_mark in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    vgs_mark = vsl_mark + V0 - VDD
    id_mark  = float(I_single(np.array([vgs_mark])))
    axes[0].annotate(f"{id_mark:.1e}A",
                     xy=(vsl_mark, id_mark),
                     xytext=(vsl_mark + 0.05, id_mark * 3),
                     fontsize=7, color="steelblue",
                     arrowprops=dict(arrowstyle="-", color="steelblue", lw=0.7))

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "phase3_id_vsl0.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")

# 콘솔 출력
print(f"\nPhase 3 Id(M0) vs V_SL0  (Vth={V0}V, VDD={VDD}V)")
print(f"  VSN = Vth + V_SL0,  Vgs(M0) = V_SL0 + {V0} - {VDD} = V_SL0 - {VDD-V0:.3f}")
print(f"  Effective drive (Vgs-Vth) = V_SL0 - VDD = V_SL0 - {VDD}  [Vth compensated]")
print(f"\n{'V_SL0':>7}  {'Vgs(M0)':>10}  {'Id(A)':>12}")
print("-" * 35)
for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    vg = v + V0 - VDD
    print(f"{v:>7.1f}  {vg:>10.4f}  {float(I_single(np.array([vg]))):>12.4e}")
