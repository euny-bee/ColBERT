"""
Phase 3 L2 search - Id vs V graph
Dual 3T1C: M0 (V1>V2=0) + M3 (V2>V1=0)

M0: gate=VSN=V1-V2+Vth, drain=V1, source=V_ML=0V
    Vgs = V1+Vth,  Vgd = Vth  (V2=0 일때)
    I_M0 = I_single(V1+Vth) - I_single(Vth)

M3: gate=V_SN_bar=V2-V1+Vth, drain=V2, source=V_ML=0V
    Vgs = V2+Vth,  Vgd = Vth  (V1=0 일때)
    I_M3 = I_single(V2+Vth) - I_single(Vth)

-> 두 식 동일 -> 같은 S-curve
-> Vth shift해도 곡선 동일 (Vth compensation)
-> 최솟값: V=0에서 I=0 (V1=V2 crossover)
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

def I_single(vgs, V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE

# Vth shift 3 케이스
cases = [
    {"V0": -0.349, "label": "Vth = -0.349V (-0.5V shift)", "color": "tomato",    "ls": "-."},
    {"V0":  0.151, "label": "Vth = +0.151V (원래)",         "color": "steelblue", "ls": "-"},
    {"V0":  0.651, "label": "Vth = +0.651V (+0.5V shift)", "color": "green",     "ls": "--"},
]

V = np.linspace(-1, 1, 600)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Phase 3 L2 search:  Id vs V\n"
    "M0 (V1>V2=0):  Vgs=V1+Vth,  Vgd=Vth  →  I = I_single(V1+Vth) - I_single(Vth)\n"
    "M3 (V2>V1=0):  Vgs=V2+Vth,  Vgd=Vth  →  same curve  [Vth compensated, 모든 곡선 겹침]",
    fontsize=9)

for ax, scale in zip(axes, ["linear", "log"]):
    for case in cases:
        V0 = case["V0"]
        Vgs = V + V0          # V1(or V2) + Vth
        Vgd = np.full_like(V, V0)  # Vth (constant)
        Id = I_single(Vgs, V0) - I_single(Vgd, V0)   # bidirectional

        if scale == "log":
            ax.semilogy(V, np.clip(np.abs(Id) * 1e9, 1e-4, None),
                        color=case["color"], lw=2, ls=case["ls"], label=case["label"])
        else:
            ax.plot(V, Id * 1e9,
                    color=case["color"], lw=2, ls=case["ls"], label=case["label"])

    ax.axvline(0, color="gray", lw=1.2, ls=":", alpha=0.7,
               label="V=0  (V1=V2, minimum)")
    ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.3)
    ax.set_xlabel("V  [V]  (= V1 when V1>V2=0,  = V2 when V2>V1=0)", fontsize=10)
    ax.set_ylabel("Id  [nA]", fontsize=11)
    ax.set_xlim(-1, 1)
    ax.set_title("Linear scale" if scale == "linear" else "Log |Id| scale", fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8.5)
    if scale == "linear":
        ax.set_ylim(-200, 12000)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase3_id_V_dual.png")
plt.savefig(out, dpi=150)
plt.show()

print(f"Saved: {out}")
print(f"\n{'V':>6}  {'Vgs':>8}  {'Vgd':>8}  {'Id(nA)':>12}  (Vth=0.151V)")
print("-" * 50)
V0 = 0.151
for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    vgs = v + V0; vgd = V0
    id_val = float((I_single(np.array([vgs]), V0) - I_single(np.array([vgd]), V0))[0])
    print(f"{v:>6.1f}  {vgs:>8.3f}  {vgd:>8.3f}  {id_val*1e9:>12.3f}")
