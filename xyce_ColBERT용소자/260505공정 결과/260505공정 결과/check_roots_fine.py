import numpy as np
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
    I_M2 = I_dev(VREAD-midL, VREAD-VML, vth)
    I_M1 = I_dev(VCOMP-sn, VCOMP-midL, vth)
    I_RMID = midL/RMID
    return I_M2 - I_M1 - I_RMID

def solve_midL(sn, vth):
    f = lambda m: net_current_into_midL(m, sn, vth)
    grid = np.linspace(-6, 6, 2000)
    vals = [f(g) for g in grid]
    for i in range(len(grid)-1):
        if vals[i]*vals[i+1] < 0:
            return brentq(f, grid[i], grid[i+1])
    return grid[np.argmin(np.abs(vals))]

def net_current_into_SN(sn, vth):
    midL = solve_midL(sn, vth)
    I_M1 = I_dev(VCOMP-sn, VCOMP-midL, vth)
    I_M0 = I_dev(sn-VBL_L, sn-midL, vth)
    I_RLEAK = sn/RLEAK
    return I_M1 - I_M0 - I_RLEAK

sn_grid = np.linspace(-2, 5, 4000)   # 10배 더 촘촘
for vth in [2.03, 3.03]:
    net = np.array([net_current_into_SN(sn, vth) for sn in sn_grid])
    roots = []
    for i in range(len(sn_grid)-1):
        if net[i]*net[i+1] < 0:
            r = brentq(lambda s: net_current_into_SN(s, vth), sn_grid[i], sn_grid[i+1])
            stability = 'stable' if net[i] > net[i+1] else 'unstable'
            roots.append((round(r,4), stability))
    print(f"Vth={vth}: roots = {roots}")
    # 4V 근처에서 net current 값도 출력
    print(f"   net_current(SN=4.0) = {net_current_into_SN(4.0, vth):.3e}")
    print(f"   net_current(SN=0.0) = {net_current_into_SN(0.0, vth):.3e}")
    print(f"   net_current(SN=1.0) = {net_current_into_SN(1.0, vth):.3e}")
    print(f"   net_current(SN=2.0) = {net_current_into_SN(2.0, vth):.3e}")
