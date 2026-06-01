"""
Phase 2 VSN settling simulation
die4_ovl20_R0 - Vth shift 비교 (+0.5V / 원래 / -0.5V)

Circuit: M0 diode-connected (gate=SN, source=V_BL0=-1V)
         Cap C between SN and V_SL0=0V
         SN initial = 3V

ODE: C * dVSN/dt = -I_M0(VSN - V_BL0)
Equilibrium: VSN = V_BL0 + Vth
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.integrate import solve_ivp

_available = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _available:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# die4 fit params (Vth=V0만 케이스별로 변화)
L = 6.1120; K = 2.7319; B = -10.7139; VSAT = 2.88; SLOPE = 4.0332e-5

# 회로 조건
V_BL0  = -1.0       # M0 source line (Phase 2)
SN_init = 3.0       # SN 초기값
C = 100e-15         # storage cap 100fF

def I_single(vgs, V0):
    v = np.clip(vgs, -10, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + max(0, vgs - VSAT) * SLOPE

def make_ode(V0):
    def ode(t, y):
        SN = y[0]
        Vgs = SN - V_BL0   # gate-to-source
        # bidirectional: Vgd=0 (diode-connected, gate=drain via M1)
        # I = I(Vgs) - I(Vgd=0)  →  zero when SN=V_BL0, clamped below V_BL0
        I_net = I_single(Vgs, V0) - I_single(0.0, V0)
        return [-max(I_net, 0.0) / C]   # 역방향 전류 차단 (V_BL0 클램프)
    return ode

cases = [
    {"V0":  0.65, "label": "Vth = +0.65V  (+0.5V shift)", "color": "tomato",    "ls": "--"},
    {"V0":  0.15, "label": "Vth = +0.15V  (원래)",         "color": "steelblue", "ls": "-"},
    {"V0": -0.35, "label": "Vth = -0.35V  (-0.5V shift)", "color": "green",     "ls": "-."},
]

t_end = 5e-6   # 5 µs
t_eval = np.linspace(0, t_end, 5000)

fig, ax = plt.subplots(figsize=(10, 5))

for case in cases:
    V0 = case["V0"]
    sol = solve_ivp(make_ode(V0), [0, t_end], [SN_init],
                    t_eval=t_eval, method="Radau", rtol=1e-8, atol=1e-10)
    SN_eq = V_BL0 + V0
    ax.plot(sol.t * 1e6, sol.y[0],
            color=case["color"], lw=2, ls=case["ls"],
            label=f"{case['label']}   ->  SN_eq = {SN_eq:.2f}V")
    ax.axhline(SN_eq, color=case["color"], lw=0.9, ls=":", alpha=0.5)

ax.axhline(V_BL0, color="gray", lw=1.2, ls="--", alpha=0.7,
           label=f"V_BL0 = {V_BL0}V (M0 source line)")

ax.set_xlabel("Time [µs]", fontsize=11)
ax.set_ylabel("VSN [V]", fontsize=11)
ax.set_xlim(0, t_end * 1e6)
ax.set_ylim(-1.8, 3.5)
ax.set_title(
    f"Phase 2: VSN settling  (C={C*1e15:.0f}fF,  V_BL0={V_BL0}V,  SN_init={SN_init}V)\n"
    f"SN_eq = V_BL0 + Vth  (Vth compensation)",
    fontsize=10)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "phase2_VSN_settling_vthshift.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")
