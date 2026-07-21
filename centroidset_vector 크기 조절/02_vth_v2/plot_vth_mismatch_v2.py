"""
Dual 3T1C Vth 불일치 효과 — 개별 보상 모델 (v2)

Option A: M0/M3 각각 별도 vth_stored 보유 → 개별 calibration 시 완벽 보상
Option C: 보상 없음 → 불일치 시 dead zone 비대칭

서브플롯: 2행(log/linear) × 4열(A원래 / A불일치+개별보상 / C원래 / C불일치)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# ── die4 파라미터 ──────────────────────────────────────────────────────────────
L = 6.1120; K = 2.7319; B = -10.7139; VSAT = 2.88; SLOPE = 4.0332e-5
VDD = 1.7

# ── Vth 시나리오 ───────────────────────────────────────────────────────────────
Vth_orig = 0.151
Vth_M0   = 0.1
Vth_M3   = 0.5

# ── 트랜지스터 전류 모델 ───────────────────────────────────────────────────────
def I_single(vgs, vth):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v   = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - vth)))) + np.maximum(0, vgs - VSAT) * SLOPE

def cell_optA(d, vth_m0, vth_m3, vth_stored_m0, vth_stored_m3):
    """Option A: M0/M3 각각 독립적인 vth_stored로 개별 보상"""
    d = np.asarray(d, dtype=float)

    vgd_m0 = d + vth_stored_m0
    I_m0 = np.where(d >= 0,
                    np.maximum(I_single(vgd_m0, vth_m0)
                               - I_single(vgd_m0 - VDD, vth_m0), 0.0),
                    0.0)

    vgd_m3 = -d + vth_stored_m3
    I_m3 = np.where(d < 0,
                    np.maximum(I_single(vgd_m3, vth_m3)
                               - I_single(vgd_m3 - VDD, vth_m3), 0.0),
                    0.0)

    clip = 1e-14
    return (np.clip(I_m0, clip, None) * 1e9,
            np.clip(I_m3, clip, None) * 1e9,
            np.clip(I_m0 + I_m3, clip, None) * 1e9)

def cell_optC(d, vth_m0, vth_m3):
    """Option C: 보상 없음"""
    d = np.asarray(d, dtype=float)

    vgd_m0 = d
    I_m0 = np.where(d >= 0,
                    np.maximum(I_single(vgd_m0, vth_m0)
                               - I_single(vgd_m0 - VDD, vth_m0), 0.0),
                    0.0)

    vgd_m3 = -d
    I_m3 = np.where(d < 0,
                    np.maximum(I_single(vgd_m3, vth_m3)
                               - I_single(vgd_m3 - VDD, vth_m3), 0.0),
                    0.0)

    clip = 1e-14
    return (np.clip(I_m0, clip, None) * 1e9,
            np.clip(I_m3, clip, None) * 1e9,
            np.clip(I_m0 + I_m3, clip, None) * 1e9)

# ── 데이터 계산 ────────────────────────────────────────────────────────────────
d_arr = np.linspace(-2.0, 2.0, 1000)

Im0_A0, Im3_A0, Itot_A0 = cell_optA(d_arr, Vth_orig, Vth_orig, Vth_orig, Vth_orig)
Im0_A1, Im3_A1, Itot_A1 = cell_optA(d_arr, Vth_M0,   Vth_M3,   Vth_M0,   Vth_M3)
Im0_C0, Im3_C0, Itot_C0 = cell_optC(d_arr, Vth_orig, Vth_orig)
Im0_C1, Im3_C1, Itot_C1 = cell_optC(d_arr, Vth_M0,   Vth_M3)

C_M0  = '#2196F3'
C_M3  = '#F44336'
C_TOT = '#4CAF50'

panels = [
    (Im0_A0, Im3_A0, Itot_A0,
     f"Option A — 원래\nVth_M0=Vth_M3={Vth_orig}V\nstored_M0=stored_M3={Vth_orig}V",
     None, None),
    (Im0_A1, Im3_A1, Itot_A1,
     f"Option A — 불일치 + 개별 보상\nVth_M0={Vth_M0}V, Vth_M3={Vth_M3}V\nstored_M0={Vth_M0}V, stored_M3={Vth_M3}V",
     None, None),
    (Im0_C0, Im3_C0, Itot_C0,
     f"Option C — 원래\nVth_M0=Vth_M3={Vth_orig}V",
     Vth_orig, Vth_orig),
    (Im0_C1, Im3_C1, Itot_C1,
     f"Option C — 불일치\nVth_M0={Vth_M0}V, Vth_M3={Vth_M3}V",
     Vth_M0, Vth_M3),
]

# ── 그림 ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
fig.suptitle(
    "Dual 3T1C Vth 불일치 효과  —  Option A (개별 보상) vs Option C (보상 없음)\n"
    f"원래: Vth_M0=Vth_M3={Vth_orig}V  |  불일치: Vth_M0={Vth_M0}V, Vth_M3={Vth_M3}V",
    fontsize=12, fontweight='bold', y=1.01
)

for col, (Im0, Im3, Itot, title, dz_pos, dz_neg) in enumerate(panels):
    for row, (ax, use_log) in enumerate(zip(axes[:, col], [True, False])):
        if use_log:
            ax.semilogy(d_arr, Im0,  color=C_M0,  lw=1.8, ls='--', label='M0 (d>0)')
            ax.semilogy(d_arr, Im3,  color=C_M3,  lw=1.8, ls=':',  label='M3 (d<0)')
            ax.semilogy(d_arr, Itot, color=C_TOT, lw=2.5, ls='-',  label='Total')
            ax.set_ylim(1e-4, 1e6)
            ax.set_ylabel("IDS [nA]  (log)", fontsize=8)
        else:
            ax.plot(d_arr, Im0,  color=C_M0,  lw=1.8, ls='--', label='M0 (d>0)')
            ax.plot(d_arr, Im3,  color=C_M3,  lw=1.8, ls=':',  label='M3 (d<0)')
            ax.plot(d_arr, Itot, color=C_TOT, lw=2.5, ls='-',  label='Total')
            ax.set_ylabel("IDS [nA]  (linear)", fontsize=8)

        ax.axvline(0, color='gray', lw=1.2, ls='--', alpha=0.5, label='d=0')
        if dz_pos is not None:
            ax.axvline( dz_pos, color='purple', lw=1.2, ls=':', alpha=0.7,
                        label=f'+dz {dz_pos:.3f}V')
        if dz_neg is not None:
            ax.axvline(-dz_neg, color='brown',  lw=1.2, ls=':', alpha=0.7,
                        label=f'-dz {dz_neg:.3f}V')

        ax.set_xlim(-2.1, 2.1)
        ax.set_xlabel("d = V1 - V2  [V]", fontsize=8)
        if row == 0:
            ax.set_title(title, fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, loc='upper center')
        ax.grid(True, which='both', ls='--', alpha=0.3)

plt.tight_layout()
out = os.path.join(
    r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\02_vth_v2',
    'plot_vth_mismatch_v2.png'
)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
