"""
Phase 2 (Data Store / Vth Compensation) 쓰기 1회당 에너지 계산.

plot_phase2_VSN_settling.py의 ODE 모델(die4_ovl20_R0, C=100fF, V_BL0=-1V, SN_init=3V)을
그대로 재사용해서, settling 궤적 SN(t)/I_net(t)를 얻고

    P(t) = I_net(t) * (SN(t) - V_BL0)      [transistor channel에서 소모되는 순간 전력]
    E_write = ∫ P(t) dt   (0 ~ t_settle, trapezoidal)

로 쓰기 1회당 실제 에너지를 적분한다. 비교용으로 단순 커패시터 모델
E_cap = 0.5 * C * (SN_init - SN_eq)^2 도 같이 계산한다 (에너지 손실 없는 이상적 하한 개념).

t_settle은 |SN(t) - SN_eq| < tol * |SN_init - SN_eq| 을 처음 만족하는 시각으로 정의.
"""
import numpy as np
from scipy.integrate import solve_ivp

# --- die4 fit params (진행정리_260505.md, plot_phase2_VSN_settling.py와 동일) ---
L = 6.1120; K = 2.7319; B = -10.7139; VSAT = 2.88; SLOPE = 4.0332e-5

V_BL0   = -1.0        # M0 source line (Phase 2)
SN_init = 3.0         # SN 초기값
C       = 100e-15     # storage capacitor, 100fF

SETTLE_TOL = 0.01      # 평형값 대비 1% 이내로 들어오면 "정착"으로 간주
T_MAX      = 5e-6      # 적분 상한 (5us, 기존 플롯과 동일 윈도우)
N_EVAL     = 200_000    # 적분 해상도 (에너지 적분 정밀도 확보)


def I_single(vgs, V0):
    v = np.clip(vgs, -10, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + max(0.0, vgs - VSAT) * SLOPE


def make_ode(V0):
    def ode(t, y):
        SN = y[0]
        Vgs = SN - V_BL0
        I_net = I_single(Vgs, V0) - I_single(0.0, V0)
        return [-max(I_net, 0.0) / C]
    return ode


def simulate_write_energy(V0, label):
    SN_eq = V_BL0 + V0
    sol = solve_ivp(make_ode(V0), [0, T_MAX], [SN_init],
                     t_eval=np.linspace(0, T_MAX, N_EVAL),
                     method="Radau", rtol=1e-9, atol=1e-12)

    t = sol.t
    SN = sol.y[0]

    Vgs = SN - V_BL0
    I_net = np.array([max(I_single(v, V0) - I_single(0.0, V0), 0.0) for v in Vgs])
    P = I_net * (SN - V_BL0)   # transistor channel에 걸리는 전압 x 전류

    # 정착 시각: |SN-SN_eq| 가 처음으로 tol*|delta_total| 이하로 내려가는 시점
    delta_total = abs(SN_init - SN_eq)
    below = np.abs(SN - SN_eq) <= SETTLE_TOL * delta_total
    if below.any():
        idx_settle = np.argmax(below)   # 처음 True인 index
    else:
        idx_settle = len(t) - 1         # 못 정착하면 전체 윈도우 사용

    t_settle = t[idx_settle]
    E_write = np.trapz(P[:idx_settle + 1], t[:idx_settle + 1])

    E_cap = 0.5 * C * (SN_init - SN_eq) ** 2   # 이상적 커패시터 에너지(하한 비교용)

    print(f"--- {label} (Vth = {V0:+.3f} V) ---")
    print(f"  SN_eq            = {SN_eq:+.4f} V")
    print(f"  settling time     = {t_settle*1e9:.2f} ns  (tol={SETTLE_TOL*100:.0f}%, window {T_MAX*1e6:.0f}us)")
    print(f"  E_write (channel) = {E_write*1e15:.4f} fJ")
    print(f"  E_cap (ideal, 0.5*C*dV^2) = {E_cap*1e15:.4f} fJ")
    print(f"  ratio E_write / E_cap     = {E_write/E_cap:.2f}x")
    print()

    return dict(label=label, V0=V0, SN_eq=SN_eq, t_settle=t_settle, E_write=E_write, E_cap=E_cap)


if __name__ == "__main__":
    cases = [
        (0.65, "Vth +0.5V shift"),
        (0.15, "Vth nominal (die4)"),
        (-0.35, "Vth -0.5V shift"),
    ]
    results = [simulate_write_energy(v0, lbl) for v0, lbl in cases]

    print("=== 요약 ===")
    print(f"{'case':<22}{'t_settle[ns]':>14}{'E_write[fJ]':>14}{'E_cap[fJ]':>12}")
    for r in results:
        print(f"{r['label']:<22}{r['t_settle']*1e9:>14.2f}{r['E_write']*1e15:>14.4f}{r['E_cap']*1e15:>12.4f}")
