"""
Phase 1(Pre-charge) + Phase 2(Data store) dual(M0+M3, 6T2C) 구조 재검증.

토폴로지 (슬라이드2 + M1/M2 always-on pass-transistor 가정):
  Phase 1: VML(0V) --M2(gate=VREAD=3V)-- mid --M1(gate=VCOMP=3V)-- SN
  Phase 2: SN --M1(gate=VCOMP=3V)-- mid --M0(gate=SN, source=bitline)--   (M2/M5는 VREAD=-3V로 OFF)

M0/M1(그리고 대칭 M3/M4)은 단방향(diode/threshold-limited) 전류만 사용한다.
(양방향 I(Vgs)-I(Vgd) 트릭은 Phase3 |V2-V1| 대칭 계산 전용이라 여기서는 쓰지 않음 — 안 그러면
 지난번처럼 Vth와 무관한 인위적 평형점이 생김.)

이번 스텝 목표:
  1) Phase1: SN이 VML=0V로 가는지 확인
  2) Phase2: Vth_orig=0.03V, shift 범위 [0,3]V (양수만) 에서 SN이 Vth별로 서로 다른 지점에
     정착하는지 확인 (에너지 계산은 다음 단계)
"""
import numpy as np
from scipy.integrate import solve_ivp

# die4 fit params
L = 6.1120; K = 2.7319; B = -10.7139; VSAT = 2.88; SLOPE = 4.0332e-5
C = 100e-15          # storage capacitor
VCOMP = 3.0
VREAD_PRECHARGE = 3.0
VML = 0.0

def I_single(vgs, V0):
    v = np.clip(vgs, -10, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - V0)))) + max(0.0, vgs - VSAT) * SLOPE


# ── Phase 1: VML --M2(VREAD)-- mid --M1(VCOMP)-- SN ──────────────────────────
def phase1_ode(V0):
    def ode(t, y):
        SN = y[0]
        # M2: gate=VREAD, drain=VML, source=mid  -> unidirectional 가정으로 mid=SN 근사 대신
        # 직접 KCL: I_M2(VML->mid) = I_M1(mid->SN)
        # monotonic 성질 이용해 대수적으로 mid 소거: I_single(VREAD-mid)=I_single(VCOMP-SN) => mid = VREAD-VCOMP+SN
        mid = VREAD_PRECHARGE - VCOMP + SN
        I_in = I_single(VREAD_PRECHARGE - mid, V0)   # M2 전류 (= M1 전류, KCL로 이미 같음)
        return [I_in / C]
    return ode


def run_phase1(SN0, V0, t_max=5e-6, n=20000):
    sol = solve_ivp(phase1_ode(V0), [0, t_max], [SN0],
                     t_eval=np.linspace(0, t_max, n), method="Radau", rtol=1e-9, atol=1e-12)
    return sol


# ── Phase 2: SN --M1(VCOMP)-- mid --M0(gate=SN, source=bitline)-- ───────────
def phase2_ode(V0, bitline):
    def ode(t, y):
        SN = y[0]
        # M1: gate=VCOMP, drain=mid, source=SN  -> I_M1 = I_single(VCOMP - SN, V0)  (mid으로부터 SN으로 유입)
        # M0: gate=SN,   drain=mid, source=bitline -> I_M0 = I_single(SN - bitline, V0) (mid에서 bitline으로 유출)
        # KCL(mid, 무용량): I_M1 = I_M0 이어야 하지만 mid이 양쪽 식에 없으므로 그대로 두 식 사용:
        #   SN으로 들어오는 전류 = I_M1, SN 커패시터는 이 전류로 충전/방전됨.
        #   정상상태에서 I_M1=I_M0가 되어야 하므로, dSN/dt는 I_M1 (M1을 통해 SN으로 들어오는 전류)로 결정.
        I_M1 = I_single(VCOMP - SN, V0)
        return [I_M1 / C]
    return ode


def run_phase2(SN0, V0, bitline, t_max=5e-6, n=20000):
    sol = solve_ivp(phase2_ode(V0, bitline), [0, t_max], [SN0],
                     t_eval=np.linspace(0, t_max, n), method="Radau", rtol=1e-9, atol=1e-12)
    return sol


if __name__ == "__main__":
    print("=== Phase 1: SN -> VML(0V) 확인 ===")
    for SN0_test in [-1.5, -0.5, 1.0]:
        for t_max in [5e-6, 5e-5, 5e-4]:
            sol = run_phase1(SN0_test, V0=0.03, t_max=t_max)
            print(f"  SN0={SN0_test:+.2f}V  t_max={t_max:.0e}s  ->  SN_final={sol.y[0,-1]:+.6f}V")
        print()

    print("=== Phase 2: Vth_orig=0.03V, shift [0,3]V, SN이 Vth별로 다른 지점에 정착하는지 ===")
    SN_after_phase1 = 0.0   # Phase1 결과를 그대로 이어받음
    bitline = -1.0           # 예시 데이터 값 (-V1), 기존 문서에서 쓰던 값과 동일
    for shift in [0.0, 0.5, 1.0, 2.0, 3.0]:
        V0 = 0.03 + shift
        for t_max in [5e-6, 5e-5, 5e-4, 5e-3]:
            sol = run_phase2(SN_after_phase1, V0, bitline, t_max=t_max)
            print(f"  Vth={V0:.3f}V (shift {shift:+.1f})  t_max={t_max:.0e}s  ->  SN_final={sol.y[0,-1]:+.6f}V"
                  f"   (bitline+Vth = {bitline+V0:+.4f}V)")
        print()
