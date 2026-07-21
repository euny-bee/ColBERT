"""
V1-V2 vs IDS: Vth 불일치(mismatch) 효과 시각화
Dual 3T1C 회로 모델 (M0 + M3)

x축: d = V1 - V2  (부호 있음, 절댓값 아님)
  d > 0: M0 담당  (query > stored data)
  d < 0: M3 담당  (query < stored data)

Option A (Vth 보상):
  M0: Vgd = d  + Vth_stored,  I = I_single(Vgd, Vth_M0) - I_single(Vgd-VDD, Vth_M0)
  M3: Vgd = -d + Vth_stored,  I = I_single(Vgd, Vth_M3) - I_single(Vgd-VDD, Vth_M3)
  동일 Vth → Vth 상쇄 → 곡선 대칭
  불일치  → Vth_stored 기준으로 한쪽만 잘 보상됨

Option C (보상 없음):
  M0: Vgd = d,   I = I_single(Vgd, Vth_M0) - I_single(Vgd-VDD, Vth_M0)
  M3: Vgd = -d,  I = I_single(Vgd, Vth_M3) - I_single(Vgd-VDD, Vth_M3)
  동일 Vth → dead zone ±Vth (대칭)
  불일치  → dead zone 비대칭 (M0쪽 Vth_M0, M3쪽 Vth_M3)
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

# -- die4 파라미터 (run_dual_phase2_die4.py 기준) ------------------------------
L = 6.1120; K = 2.7319; B = -10.7139; VSAT = 2.88; SLOPE = 4.0332e-5
VDD = 1.0   # 회로 공급전압

# -- 시나리오 설정 --------------------------------------------------------------
Vth_orig = 0.151   # 원래 (M0=M3 동일)
Vth_M0   = 0.1     # 불일치: M0 Vth
Vth_M3   = 0.5     # 불일치: M3 Vth

# -- 단방향 전류 모델 (Vth 파라미터화) -----------------------------------------
def I_single(vgs, vth):
    vgs = np.atleast_1d(np.array(vgs, dtype=float))
    v   = np.minimum(vgs, VSAT)
    return 10 ** (B + L / (1 + np.exp(-K * (v - vth)))) + np.maximum(0, vgs - VSAT) * SLOPE

# -- 셀 전류 계산 ---------------------------------------------------------------
def cell_optA(d, vth_m0, vth_m3, vth_stored):
    """
    Option A: Vth 보상 회로
    d: V1-V2 array (부호 있음)
    반환: (I_M0, I_M3, I_total) [nA]
    """
    d = np.asarray(d, dtype=float)

    # M0: d>0 구간
    vgd_m0 = d + vth_stored
    I_m0 = np.where(d >= 0,
                    np.maximum(I_single(vgd_m0, vth_m0)
                               - I_single(vgd_m0 - VDD, vth_m0), 0.0),
                    0.0)

    # M3: d<0 구간
    vgd_m3 = -d + vth_stored
    I_m3 = np.where(d < 0,
                    np.maximum(I_single(vgd_m3, vth_m3)
                               - I_single(vgd_m3 - VDD, vth_m3), 0.0),
                    0.0)

    clip = 1e-14
    return np.clip(I_m0, clip, None)*1e9, np.clip(I_m3, clip, None)*1e9, \
           np.clip(I_m0+I_m3, clip, None)*1e9   # nA

def cell_optC(d, vth_m0, vth_m3):
    """
    Option C: 보상 없음
    반환: (I_M0, I_M3, I_total) [nA]
    """
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
    return np.clip(I_m0, clip, None)*1e9, np.clip(I_m3, clip, None)*1e9, \
           np.clip(I_m0+I_m3, clip, None)*1e9

# -- 그래프 설정 ----------------------------------------------------------------
d_arr = np.linspace(-2.0, 2.0, 1000)

# 서브플롯 배치:
#  행 1: Option A  |  열 0: 원래  |  열 1: 불일치 stored=Vth_M0  |  열 2: 불일치 stored=Vth_M3
#  행 2: Option C  |  열 0: 원래  |  열 1: 불일치                 |  열 2: 두 옵션 total 비교
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    "d = V1-V2 vs IDS  —  Dual 3T1C Vth 불일치 효과\n"
    f"원래: Vth_M0=Vth_M3={Vth_orig}V  |  불일치: Vth_M0={Vth_M0}V,  Vth_M3={Vth_M3}V",
    fontsize=12, fontweight='bold', y=0.99
)

C_M0  = '#2196F3'   # M0 전류색 (파랑)
C_M3  = '#F44336'   # M3 전류색 (빨강)
C_TOT = '#4CAF50'   # total (초록)
C_TOT2= '#FF9800'   # 비교용 total (주황)

def draw_panel(ax, Im0, Im3, Itot, title, subtitle='',
               extra_tot=None, extra_label=None,
               vth_dz_pos=None, vth_dz_neg=None):
    ax.semilogy(d_arr, Im0,  color=C_M0,  lw=1.8, ls='--', label='M0 (d>0 담당)')
    ax.semilogy(d_arr, Im3,  color=C_M3,  lw=1.8, ls=':',  label='M3 (d<0 담당)')
    ax.semilogy(d_arr, Itot, color=C_TOT, lw=2.5, ls='-',  label='Total')
    if extra_tot is not None:
        ax.semilogy(d_arr, extra_tot, color=C_TOT2, lw=2.5, ls='-.',
                    label=extra_label)
    ax.axvline(0, color='gray', lw=1.2, ls='--', alpha=0.5, label='d=0 (V1=V2)')
    if vth_dz_pos is not None:
        ax.axvline( vth_dz_pos, color='purple', lw=1.2, ls=':', alpha=0.7,
                    label=f'+dead zone {vth_dz_pos:.2f}V')
    if vth_dz_neg is not None:
        ax.axvline(-vth_dz_neg, color='brown',  lw=1.2, ls=':', alpha=0.7,
                    label=f'-dead zone {vth_dz_neg:.2f}V')
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(1e-4, 1e6)
    ax.set_xlabel("d = V1 - V2  [V]", fontsize=9)
    ax.set_ylabel("IDS  [nA]", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    if subtitle:
        ax.text(0.5, 0.01, subtitle, transform=ax.transAxes,
                ha='center', fontsize=8, color='navy',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))
    ax.legend(fontsize=7.5, loc='upper center')
    ax.grid(True, which='both', ls='--', alpha=0.3)

# ── [0,0] Option A 원래 ───────────────────────────────────────────────────────
Im0, Im3, Itot = cell_optA(d_arr, Vth_orig, Vth_orig, Vth_orig)
draw_panel(axes[0,0], Im0, Im3, Itot,
           f"Option A — 원래\nVth_M0=Vth_M3={Vth_orig}V (Vth_stored={Vth_orig}V)",
           "Vth 완전 상쇄 → 완전 대칭 U자",
           vth_dz_pos=None, vth_dz_neg=None)

# ── [0,1] Option A 불일치, Vth_stored=Vth_M0 ──────────────────────────────────
Im0a, Im3a, Itota = cell_optA(d_arr, Vth_M0, Vth_M3, vth_stored=Vth_M0)
draw_panel(axes[0,1], Im0a, Im3a, Itota,
           f"Option A — 불일치 (stored=Vth_M0={Vth_M0}V)\nVth_M0={Vth_M0}V, Vth_M3={Vth_M3}V",
           f"M0쪽 정확히 보상\nM3쪽: dead zone = Vth_M3-Vth_stored={Vth_M3-Vth_M0:.1f}V",
           vth_dz_pos=None, vth_dz_neg=Vth_M3-Vth_M0)

# ── [0,2] Option A 불일치, Vth_stored=Vth_M3 ──────────────────────────────────
Im0b, Im3b, Itotb = cell_optA(d_arr, Vth_M0, Vth_M3, vth_stored=Vth_M3)
draw_panel(axes[0,2], Im0b, Im3b, Itotb,
           f"Option A — 불일치 (stored=Vth_M3={Vth_M3}V)\nVth_M0={Vth_M0}V, Vth_M3={Vth_M3}V",
           f"M3쪽 정확히 보상\nM0쪽: 과보상 +{Vth_M3-Vth_M0:.1f}V (d>0에서 전류 증가)",
           vth_dz_pos=None, vth_dz_neg=None)

# ── [1,0] Option C 원래 ───────────────────────────────────────────────────────
Im0c, Im3c, Itotc = cell_optC(d_arr, Vth_orig, Vth_orig)
draw_panel(axes[1,0], Im0c, Im3c, Itotc,
           f"Option C — 원래\nVth_M0=Vth_M3={Vth_orig}V",
           f"dead zone 대칭: ±{Vth_orig}V",
           vth_dz_pos=Vth_orig, vth_dz_neg=Vth_orig)

# ── [1,1] Option C 불일치 ──────────────────────────────────────────────────────
Im0d, Im3d, Itotd = cell_optC(d_arr, Vth_M0, Vth_M3)
draw_panel(axes[1,1], Im0d, Im3d, Itotd,
           f"Option C — 불일치\nVth_M0={Vth_M0}V, Vth_M3={Vth_M3}V",
           f"dead zone 비대칭: M0쪽 {Vth_M0}V / M3쪽 {Vth_M3}V",
           vth_dz_pos=Vth_M0, vth_dz_neg=Vth_M3)

# ── [1,2] 비교: total 전류 4개 겹쳐 보기 ─────────────────────────────────────
ax = axes[1,2]
ax.semilogy(d_arr, Itot,  color='black',    lw=2.5, ls='-',  label=f'A 원래')
ax.semilogy(d_arr, Itota, color=C_M0,       lw=2.0, ls='--', label=f'A 불일치 stored=M0')
ax.semilogy(d_arr, Itotb, color=C_M3,       lw=2.0, ls=':',  label=f'A 불일치 stored=M3')
ax.semilogy(d_arr, Itotc, color='gray',     lw=2.0, ls='-',  label=f'C 원래')
ax.semilogy(d_arr, Itotd, color=C_TOT2,     lw=2.0, ls='-.', label=f'C 불일치')
ax.axvline(0, color='gray', lw=1, ls='--', alpha=0.4)
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(1e-4, 1e6)
ax.set_xlabel("d = V1 - V2  [V]", fontsize=9)
ax.set_ylabel("IDS  [nA]", fontsize=9)
ax.set_title("Total IDS 전체 비교\n(A vs C, 원래 vs 불일치)", fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper center')
ax.grid(True, which='both', ls='--', alpha=0.3)

plt.tight_layout()
out = os.path.join(r'C:\Users\nmdl-khb\ColBERT\centroidset', 'plot_vth_mismatch_vi.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
