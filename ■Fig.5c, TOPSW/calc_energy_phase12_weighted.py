"""Phase1/Phase2 에너지 -- 방법2(고정 worst-case write-time) + PBS Vth 분포(v2/v3/v4) 가중평균."""
import numpy as np

Vth_orig = 0.151045
V0_list = np.array([0.03, 0.53, 1.03, 2.03, 3.03])
shift_list = V0_list - Vth_orig   # -0.121, 0.379, 0.879, 1.879, 2.879

E1 = 0.0072  # fJ, Vth 무관 (Phase1)
E2_list = np.array([1565.3937, 1356.1262, 535.9825, 142.7205, 83.8550])  # fJ, 방법2 (fixed write-time=3200.1us)
Etot_list = E1 + E2_list

VTH_CONDS = [('v2', 0.0, 1.0, 0.30), ('v3', 0.0, 2.0, 0.60), ('v4', 0.0, 3.0, 0.90)]

def truncnorm_pdf(x, lo, hi, std):
    from math import erf, sqrt
    Phi = lambda z: 0.5*(1+erf(z/(std*sqrt(2))))
    norm = Phi(hi) - Phi(lo)
    pdf = np.exp(-0.5*(x/std)**2) / (std*np.sqrt(2*np.pi))
    return pdf / norm

print(f"{'cond':4s} {'range':10s} {'E_phase2_expected(fJ)':>22s} {'E_total_expected(fJ)':>22s}")
for name, lo, hi, std in VTH_CONDS:
    grid = np.linspace(lo, hi, 3001)
    E2_grid = np.interp(grid, shift_list, E2_list)
    Etot_grid = np.interp(grid, shift_list, Etot_list)
    w = truncnorm_pdf(grid, lo, hi, std)
    w_norm = np.trapz(w, grid)
    E2_exp = np.trapz(E2_grid*w, grid) / w_norm
    Etot_exp = np.trapz(Etot_grid*w, grid) / w_norm
    print(f"{name:4s} [{lo:.1f},{hi:.1f}]V  {E2_exp:22.4f} {Etot_exp:22.4f}")

print(f"\nE_phase1 (write, Vth 무관) = {E1:.4f} fJ")
print("참고: E_total_expected = E_phase1 + E_phase2_expected (Option A 1회 write 에너지 기댓값)")
print("Option C 쓰기(=Phase1만, Vth 무관) = 0.0072 fJ (모든 PBS 조건 동일)")
