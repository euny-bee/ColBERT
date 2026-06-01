"""
Phase 3 L2 search: Id vs V (Vth = +0.151V)
M0 (V1 > V2=0): Vgs = V1+Vth, Vgd = Vth -> I_M0 = I_single(V1+Vth) - I_single(Vth)
M3 (V2 > V1=0): Vgs = V2+Vth, Vgd = Vth -> I_M3 = I_single(V2+Vth) - I_single(Vth)
두 식 동일 -> 같은 S-curve / V=0에서 I=0 (V1=V2 crossover)
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

L = 6.1120; K = 2.7319; B = -10.7139; VSAT = 2.88; SLOPE = 4.0332e-5
V0 = 0.151   # Vth

def I_single(vgs):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v   = np.minimum(vgs, VSAT)
    return 10**(B + L/(1 + np.exp(-K*(v - V0)))) + np.maximum(0, vgs - VSAT)*SLOPE

V = np.linspace(-1, 1, 600)

# M0 (V2=0): Vgs=V+Vth, Vgd=Vth
I_M0 = I_single(V + V0) - I_single(V0)   # V = V1 sweep

# M3 (V1=0): Vgs=V+Vth, Vgd=Vth  -> 완전 동일
I_M3 = I_single(V + V0) - I_single(V0)   # V = V2 sweep

# ── 플롯 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Phase 3 L2 search  —  Id vs V  (Vth = +{V0}V)\n"
    f"M0: V1>V2=0,  Vgs=V1+{V0},  I_M0 = I(V1+{V0}) - I({V0})\n"
    f"M3: V2>V1=0,  Vgs=V2+{V0},  I_M3 = I(V2+{V0}) - I({V0})  [same curve]",
    fontsize=9)

for ax, scale in zip(axes, ["linear", "log"]):
    # M0 curve
    ax.plot(V, I_M0 * 1e9, color="steelblue", lw=2.5, ls="-",
            label=f"M0  (V=V1, V2=0)  →  I_M0")
    # M3 curve (dashed, overlaps M0)
    ax.plot(V, I_M3 * 1e9, color="tomato",    lw=2,   ls="--",
            label=f"M3  (V=V2, V1=0)  →  I_M3  [M0와 완전 겹침]")

    ax.axvline(0,  color="gray", lw=1.2, ls=":",  alpha=0.7,
               label="V=0  (V1=V2,  I=0,  minimum)")
    ax.axhline(0,  color="gray", lw=0.8, ls="-",  alpha=0.3)

    ax.set_xlabel("V  [V]  (= V1 or V2, the nonzero one)", fontsize=10)
    ax.set_xlim(-1, 1)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=9)

    if scale == "linear":
        ax.set_ylabel("Id  [nA]", fontsize=11)
        ax.set_ylim(-500, 12000)
        ax.set_title("Linear scale", fontsize=10)
        # 주요 포인트 annotation
        for v_mark in [0.3, 0.5, 0.7, 1.0]:
            i_mark = float(I_single(np.array([v_mark + V0])) - I_single(np.array([V0])))*1e9
            ax.annotate(f"{i_mark:.0f} nA",
                        xy=(v_mark, i_mark),
                        xytext=(v_mark - 0.18, i_mark + 600),
                        fontsize=7.5, color="steelblue",
                        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8))
    else:
        ax.set_ylabel("Id  [nA]  (log)", fontsize=11)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 2e4)
        ax.set_title("Log scale", fontsize=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "phase3_idV_vth0p151.png")
plt.savefig(out, dpi=150)
plt.show()

print(f"Saved: {out}")
print(f"\nId vs V  (Vth={V0}V)")
print(f"{'V':>6}  {'Vgs':>8}  {'Vgd':>8}  {'Id [nA]':>12}")
print("-" * 42)
for v in [-1.0, -0.5, 0.0, 0.2, 0.5, 0.7, 1.0]:
    ig  = v + V0
    igd = V0
    id_ = float(I_single(np.array([ig])) - I_single(np.array([igd])))*1e9
    print(f"{v:>6.1f}  {ig:>8.3f}  {igd:>8.3f}  {id_:>12.2f}")
