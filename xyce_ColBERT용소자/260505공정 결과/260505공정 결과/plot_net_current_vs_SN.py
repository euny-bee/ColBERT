"""M0 루프의 SN에 흘러들어가는 net 전류를 SN의 함수로 직접 계산 -- 고정점(=SN이 멈추는 곳) 시각화.
midL은 커패시터가 없는 저항성 노드라, 주어진 SN에 대해 KCL(M2+M1+RMID_L)이 balance되는 midL을 매 순간 풀어야 함.
Phase2 조건: VCOMP=3V(M1 게이트), VREAD=-3V(M2 게이트, 거의 꺼짐), VBL_L=-1V, VML=0V.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

L, K, B, VSAT, SLOPE = 6.1120, 2.7319, -10.7139, 2.88, 4.0332e-5
RLEAK = 10e9
RMID = 10e9
VCOMP, VREAD, VBL_L, VML = 3.0, -3.0, -1.0, 0.0

def Ids_1dir(vgs, vth):
    v = min(vgs, VSAT)
    return 10**(B + L/(1+np.exp(-K*(v-vth)))) + max(0, vgs-VSAT)*SLOPE

def I_dev(vgs, vgd, vth):
    return Ids_1dir(vgs, vth) - Ids_1dir(vgd, vth)

def net_current_into_midL(midL, sn, vth):
    # M2: drain=VML, gate=VREAD, source=midL -> current flows VML->midL is I_M2(Vgs=VREAD-midL, Vgd=VREAD-VML)
    I_M2 = I_dev(VREAD-midL, VREAD-VML, vth)   # M2 current from VML side into midL (drain=VML perspective flips) - 부호는 아래서 통일
    # 실제로: XM2 drain=VML gate=VREAD source=midL -> I(drain->source 방향) = I_dev(Vgs=V(gate,source)=VREAD-midL, Vgd=V(gate,drain)=VREAD-VML)
    # 이 전류는 drain(VML)에서 source(midL)로 흘러들어가는 걸 양수로 정의 (device convention)
    # M1: drain=midL, gate=VCOMP, source=SN -> I(drain(midL)->source(SN)) = I_dev(Vgs=VCOMP-SN, Vgd=VCOMP-midL)
    I_M1 = I_dev(VCOMP-sn, VCOMP-midL, vth)   # midL->SN 방향으로 흘러나가는 전류
    I_RMID = midL/RMID   # midL -> gnd
    # midL 노드 KCL: I_M2(들어옴) - I_M1(나감) - I_RMID(나감) = 0
    return I_M2 - I_M1 - I_RMID

def solve_midL(sn, vth):
    f = lambda m: net_current_into_midL(m, sn, vth)
    # 넓은 범위에서 부호 바뀌는 구간 찾기
    grid = np.linspace(-6, 6, 400)
    vals = [f(g) for g in grid]
    for i in range(len(grid)-1):
        if vals[i]*vals[i+1] < 0:
            return brentq(f, grid[i], grid[i+1])
    return grid[np.argmin(np.abs(vals))]

def net_current_into_SN(sn, vth):
    midL = solve_midL(sn, vth)
    I_M1 = I_dev(VCOMP-sn, VCOMP-midL, vth)   # midL -> SN 로 들어옴
    I_M0 = I_dev(sn-VBL_L, sn-midL, vth)      # SN -> VBL_L 로 나감 (gate=SN, drain=midL, source=VBL_L)
    I_RLEAK = sn/RLEAK                         # SN -> gnd 로 나감
    return I_M1 - I_M0 - I_RLEAK

sn_grid = np.linspace(-2, 5, 400)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, vth in zip(axes, [2.03, 3.03]):
    net = [net_current_into_SN(sn, vth) for sn in sn_grid]
    ax.axhline(0, color='k', lw=0.8)
    ax.plot(sn_grid, net, color='tab:blue', lw=1.8)
    ax.axvline(-1.0+vth, color='red', ls='--', lw=1, label=f'이상적 목표 (-V1+Vth={-1+vth:.2f})')
    ax.set_xlabel('SN [V]')
    ax.set_ylabel('SN으로 들어가는 net 전류 [A]')
    ax.set_title(f'Vth={vth}V')
    ax.set_yscale('symlog', linthresh=1e-13)
    ax.legend(fontsize=8)
    ax.grid(True, ls='--', alpha=0.4)

fig.suptitle('SN에 대한 net 전류(dSN/dt에 비례) -- 0을 지나는 지점(부호가 +->-)이 안정된 고정점')
fig.tight_layout()
fig.savefig('net_current_vs_SN.png', dpi=150)
print("Saved: net_current_vs_SN.png")

# 근(roots) 출력
from scipy.optimize import brentq as bq
for vth in [2.03, 3.03]:
    net = np.array([net_current_into_SN(sn, vth) for sn in sn_grid])
    roots = []
    for i in range(len(sn_grid)-1):
        if net[i]*net[i+1] < 0:
            r = bq(lambda s: net_current_into_SN(s, vth), sn_grid[i], sn_grid[i+1])
            slope_sign = 'stable' if net[i] > net[i+1] else 'unstable'
            roots.append((r, slope_sign))
    print(f"Vth={vth}: roots(SN, stability) = {roots}")
