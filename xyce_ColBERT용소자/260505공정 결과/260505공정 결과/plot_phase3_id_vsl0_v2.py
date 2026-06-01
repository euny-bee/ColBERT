"""
Phase 3 (2번): Id(M0) vs V_SL0  [CORRECTED]
- M0 drain = VDD = 1.7V (read supply)
- M0 source = V_ML ~ 0V (match line, pre-discharged)
- Vgs(M0) = VSN - V_ML = (Vth + V_SL0) - 0 = Vth + V_SL0
- Vds = VDD = 1.7V (saturation)
- Effective drive = Vgs - Vth = V_SL0  (Vth compensated!)
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
VDD = 1.7   # M0 drain (read supply)
V_ML = 0.0  # M0 source (match line, pre-discharged)

def I_single(vgs, V0):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + np.maximum(0, vgs - VSAT) * SLOPE

vsl0 = np.linspace(-2, 2, 600)

# Vth compensation 확인: 3가지 Vth case
cases = [
    {"V0": -0.349, "label": "Vth = -0.349V  (-0.5V shift)", "color": "tomato",    "ls": "-."},
    {"V0":  0.151, "label": "Vth = +0.151V  (원래)",         "color": "steelblue", "ls": "-"},
    {"V0":  0.651, "label": "Vth = +0.651V  (+0.5V shift)", "color": "green",     "ls": "--"},
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Phase 3 (2번):  Id(M0) vs V_SL0\n"
    "VSN = Vth + V_SL0  |  Vgs = VSN - V_ML = Vth + V_SL0  |  Vds = VDD = 1.7V\n"
    "Effective drive = Vgs - Vth = V_SL0  →  Vth 보상!  (모든 곡선 겹침)",
    fontsize=9)

for ax, scale in zip(axes, ["linear", "log"]):
    for case in cases:
        V0 = case["V0"]
        Vgs = V0 + vsl0          # Vth_stored + V_SL0
        Vgd = Vgs - VDD          # bidirectional
        Id = np.maximum(I_single(Vgs, V0) - I_single(Vgd, V0), 0)
        Id_nA = Id * 1e9

        if scale == "log":
            ax.semilogy(vsl0, np.clip(Id_nA, 1e-5, None),
                        color=case["color"], lw=2, ls=case["ls"], label=case["label"])
        else:
            ax.plot(vsl0, Id * 1e6,          # µA 단위
                    color=case["color"], lw=2, ls=case["ls"], label=case["label"])

    ax.axvline(0, color="gray", lw=1.2, ls=":", alpha=0.7, label="V_SL0=0V (threshold)")
    ax.axvspan(-1, 1, alpha=0.07, color="orange", label="V_SL0 op. range (-1~1V)")
    ax.set_xlabel("V_SL0  [V]", fontsize=11)
    ax.set_ylabel("Id  [nA]" if scale == "log" else "Id  [µA]", fontsize=11)
    ax.set_xlim(-2, 2)
    ax.set_title("Linear scale" if scale == "linear" else "Log scale", fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8.5)
    if scale == "linear":
        ax.set_ylim(-2, 30)     # µA 기준
    else:
        ax.set_ylim(1e-5, None)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase3_id_vsl0_v2.png")
plt.savefig(out, dpi=150)
plt.show()

print(f"Saved: {out}")
print(f"\nV_SL0  |  Vgs(M0)  |  Id [nA]  (Vth=0.151V)")
print("-" * 45)
for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    vg = 0.151 + v
    vgd = vg - VDD
    id_val = float(np.maximum(I_single(np.array([vg]), 0.151) - I_single(np.array([vgd]), 0.151), 0)[0])
    print(f"{v:+.1f}V  |  {vg:.3f}V   |  {id_val*1e9:.3f} nA")
