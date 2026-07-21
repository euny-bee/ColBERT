"""
Dual Vth 독립 샘플링 ColBERT centroid 선택 시뮬레이션

각 셀 (query i, centroid j, dim k) 마다:
  Vth_M0[i,j,k] ~ TruncGauss (독립)
  Vth_M3[i,j,k] ~ TruncGauss (독립, M0와 무관)

d = Q[i,k] - C[j,k]  (부호 있음)
  d >= 0: M0 담당, Vth_M0 사용
  d <  0: M3 담당, Vth_M3 사용

Option A (Vth 보상):
  Vth_stored 3가지 케이스:
    stored_M0: 셀마다 Vth_M0 저장 → M0 완전 보상, M3 일부 미보상
    stored_M3: 셀마다 Vth_M3 저장 → M3 완전 보상, M0 과보상
    stored_rnd: 셀마다 M0/M3 중 랜덤 선택

Option C (보상 없음):
  dead zone 비대칭: M0쪽 Vth_M0, M3쪽 Vth_M3
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import truncnorm
import os

_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# -- 회로 파라미터 (die4 logistic, VDS=1.7V target) ----------------------------
L_f  =  6.112020
K_f  =  2.731949
B_f  = -10.713911
VDS_m = 1.0
VDS_t = 1.7

# -- Vth 샘플링 설정 (현재: Option C 기존 코드와 동일 분포) --------------------
# NOTE: 나중에 분포 변경 예정
Vth_orig = 0.151045
VTH_MEAN, VTH_STD = 0.0, 0.15
VTH_LO,   VTH_HI  = 0.0, 0.5
_a = (VTH_LO - VTH_MEAN) / VTH_STD
_b = (VTH_HI - VTH_MEAN) / VTH_STD

def sample_vth(shape, rng):
    shifts = truncnorm.rvs(_a, _b, loc=VTH_MEAN, scale=VTH_STD,
                           size=shape, random_state=rng)
    return Vth_orig + shifts

# -- IDS 계산 함수 (Vth 파라미터화) -------------------------------------------
def logistic_ids(vgs, vth):
    return 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - vth))))

def vds_correction(vgs, vth):
    VoD = vgs - vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > VDS_m, Vo*VDS_m - VDS_m**2/2, Vo**2/2)
    It = np.where(Vo > VDS_t, Vo*VDS_t - VDS_t**2/2, Vo**2/2)
    factor[on] = np.where(Im > 0, It/Im, 1.0)
    return factor

def calc_ids(vgs, vth):
    return logistic_ids(vgs, vth) * vds_correction(vgs, vth)

# -- Option A: 셀 전류 (부호 있는 d, 독립 Vth) --------------------------------
def optA_cell_current(d, vth_m0, vth_m3, vth_stored):
    """
    d, vth_m0, vth_m3, vth_stored: 같은 shape (broadcast 가능)
    반환: I_k [A]
    """
    # M0 (d >= 0): Vgd = d + vth_stored, calc_ids with Vth_M0
    vgd_m0 = d + vth_stored
    I_m0 = np.maximum(calc_ids(vgd_m0, vth_m0)
                      - calc_ids(vgd_m0 - VDS_t, vth_m0), 0.0)

    # M3 (d < 0): Vgd = -d + vth_stored, calc_ids with Vth_M3
    vgd_m3 = -d + vth_stored
    I_m3 = np.maximum(calc_ids(vgd_m3, vth_m3)
                      - calc_ids(vgd_m3 - VDS_t, vth_m3), 0.0)

    return np.where(d >= 0, I_m0, I_m3)

# -- Option C: 셀 전류 (보상 없음) --------------------------------------------
def optC_cell_current(d, vth_m0, vth_m3):
    """
    반환: I_k [A]
    """
    vgd_m0 = d
    I_m0 = np.maximum(calc_ids(vgd_m0, vth_m0)
                      - calc_ids(vgd_m0 - VDS_t, vth_m0), 0.0)

    vgd_m3 = -d
    I_m3 = np.maximum(calc_ids(vgd_m3, vth_m3)
                      - calc_ids(vgd_m3 - VDS_t, vth_m3), 0.0)

    return np.where(d >= 0, I_m0, I_m3)

# -- 데이터 로드 ---------------------------------------------------------------
base = r'C:\Users\nmdl-khb\ColBERT\centroidset'
C_mat = pd.read_excel(os.path.join(base, 'centroids_100x128.xlsx'),
                      header=0, index_col=0).values.astype(float)   # (100, 128)
Q_mat = pd.read_excel(os.path.join(base, 'query_embs_96x128.xlsx'),
                      header=0, index_col=0).values.astype(float)   # (96,  128)

print(f"Q: {Q_mat.shape}  C: {C_mat.shape}")

# dot product 기준 이상적 best centroid
dot_matrix  = Q_mat @ C_mat.T                      # (96, 100)
best_ideal  = np.argmax(dot_matrix, axis=1)        # (96,)
sorted_ideal = np.argsort(-dot_matrix, axis=1)     # (96, 100) descending

# -- signed difference d[i,j,k] -----------------------------------------------
d_all = Q_mat[:, np.newaxis, :] - C_mat[np.newaxis, :, :]  # (96, 100, 128)
shape = d_all.shape  # (96, 100, 128)

# -- 시뮬레이션 실행 ------------------------------------------------------------
SEED_BASE = 42
results = {}

# ── 기준: 기존 Option A (단일 고정 Vth, |d| 사용) ───────────────────────────
diff_abs = np.abs(d_all)
VGS_orig = diff_abs + Vth_orig
I_orig   = calc_ids(VGS_orig, Vth_orig).sum(axis=2) * 1e6  # (96,100) uA
results['A_baseline'] = {
    'I_total': I_orig,
    'label':   f'Option A 기준\n(고정 Vth={Vth_orig:.3f}, |d| 사용)',
    'color':   'black'
}

# ── Option A: Vth_M0, Vth_M3 독립 샘플링 3 케이스 ───────────────────────────
for case, stored_key in [('stored_M0', 'M0'), ('stored_M3', 'M3'), ('stored_rnd', 'Rnd')]:
    rng_m0 = np.random.default_rng(SEED_BASE)
    rng_m3 = np.random.default_rng(SEED_BASE + 1)   # 독립

    Vth_M0_s = sample_vth(shape, rng_m0)  # (96,100,128)
    Vth_M3_s = sample_vth(shape, rng_m3)  # (96,100,128) 독립

    if stored_key == 'M0':
        Vth_stored = Vth_M0_s
        label = 'Option A 독립 Vth\n(stored=Vth_M0)'
        color = '#2196F3'
    elif stored_key == 'M3':
        Vth_stored = Vth_M3_s
        label = 'Option A 독립 Vth\n(stored=Vth_M3)'
        color = '#9C27B0'
    else:
        # 셀마다 M0 또는 M3 중 랜덤 선택
        rng_rnd = np.random.default_rng(SEED_BASE + 2)
        mask = rng_rnd.integers(0, 2, size=shape).astype(bool)
        Vth_stored = np.where(mask, Vth_M0_s, Vth_M3_s)
        label = 'Option A 독립 Vth\n(stored=Random)'
        color = '#FF9800'

    I_k = optA_cell_current(d_all, Vth_M0_s, Vth_M3_s, Vth_stored)
    I_total = I_k.sum(axis=2) * 1e6
    results[f'A_{case}'] = {'I_total': I_total, 'label': label, 'color': color}

# ── Option C: Vth_M0, Vth_M3 독립 샘플링 ────────────────────────────────────
rng_m0 = np.random.default_rng(SEED_BASE + 3)
rng_m3 = np.random.default_rng(SEED_BASE + 4)
Vth_M0_c = sample_vth(shape, rng_m0)
Vth_M3_c = sample_vth(shape, rng_m3)

I_kc    = optC_cell_current(d_all, Vth_M0_c, Vth_M3_c)
I_totalc = I_kc.sum(axis=2) * 1e6
results['C_dual'] = {
    'I_total': I_totalc,
    'label':   'Option C 독립 Vth\n(보상 없음)',
    'color':   '#F44336'
}

# ── Option C 기준: 기존 단일 Vth TruncGauss (재현용) ────────────────────────
rng_ref = np.random.default_rng(SEED_BASE)
Vth_ref  = sample_vth((96, 100), rng_ref)   # (96,100) 기존 방식
I_ref    = (calc_ids(diff_abs, Vth_ref[:, :, np.newaxis])
            .sum(axis=2) * 1e6)
results['C_baseline'] = {
    'I_total': I_ref,
    'label':   'Option C 기준\n(단일 TruncGauss Vth)',
    'color':   'gray'
}

# -- 정확도 계산 ---------------------------------------------------------------
def calc_metrics(I_total):
    best = np.argmin(I_total, axis=1)
    top1 = np.mean(best == best_ideal) * 100
    topk = {k: np.mean([best[i] in sorted_ideal[i, :k] for i in range(96)]) * 100
            for k in [1, 3, 5, 10]}
    return best, topk

print(f"\n{'':30s}  Top-1   Top-3   Top-5  Top-10")
print("-" * 60)
metrics = {}
for key, val in results.items():
    best, topk = calc_metrics(val['I_total'])
    metrics[key] = topk
    print(f"  {val['label'][:28]:30s}  {topk[1]:5.1f}%  {topk[3]:5.1f}%  "
          f"{topk[5]:5.1f}%  {topk[10]:5.1f}%")

# -- 시각화 -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Dual Vth 독립 샘플링 — ColBERT Centroid 선택 정확도\n"
    f"Vth_M0, Vth_M3 각각 TruncGauss(mean={Vth_orig:.3f}, std={VTH_STD}, "
    f"range=[{VTH_LO},{VTH_HI}]) 독립 샘플링  |  seed M0={SEED_BASE}, M3={SEED_BASE+1}",
    fontsize=10, fontweight='bold'
)

# ── 왼쪽: Top-k accuracy 비교 바 차트 ────────────────────────────────────────
ax = axes[0]
keys_order = ['A_baseline', 'A_stored_M0', 'A_stored_M3', 'A_stored_rnd',
              'C_baseline', 'C_dual']
ks = [1, 3, 5, 10]
x  = np.arange(len(ks))
w  = 0.12
for i, key in enumerate(keys_order):
    if key not in results:
        continue
    vals  = [metrics[key][k] for k in ks]
    color = results[key]['color']
    label = results[key]['label'].replace('\n', ' ')
    bars  = ax.bar(x + i*w, vals, w, color=color, alpha=0.85, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.5,
                f'{v:.0f}', ha='center', fontsize=6, rotation=90)

ax.set_xticks(x + w*2.5)
ax.set_xticklabels([f'Top-{k}' for k in ks], fontsize=10)
ax.set_ylabel("Accuracy [%]", fontsize=10)
ax.set_title("Top-k Centroid 선택 정확도", fontsize=11, fontweight='bold')
ax.set_ylim(0, 120)
ax.axhline(100, color='gray', ls='--', lw=1, alpha=0.4)
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, axis='y', ls='--', alpha=0.3)

# ── 오른쪽: Top-1 accuracy 변화량 (기준 대비) ─────────────────────────────────
ax2 = axes[1]
base_A = metrics['A_baseline'][1]
base_C = metrics['C_baseline'][1]

compare_pairs = [
    ('A_stored_M0', base_A, 'A 기준'),
    ('A_stored_M3', base_A, 'A 기준'),
    ('A_stored_rnd', base_A, 'A 기준'),
    ('C_dual',       base_C, 'C 기준'),
]
labels2 = [results[k]['label'].replace('\n',' ') for k,_,_ in compare_pairs]
deltas  = [metrics[k][1] - ref for k,ref,_ in compare_pairs]
colors2 = [results[k]['color'] for k,_,_ in compare_pairs]

bars2 = ax2.bar(range(len(deltas)), deltas,
                color=colors2, alpha=0.85, width=0.5)
for b, v in zip(bars2, deltas):
    ax2.text(b.get_x()+b.get_width()/2,
             v + (0.3 if v >= 0 else -0.8),
             f'{v:+.1f}%', ha='center', fontsize=10, fontweight='bold')
ax2.axhline(0, color='black', lw=1.2)
ax2.set_xticks(range(len(labels2)))
ax2.set_xticklabels(labels2, fontsize=7.5, rotation=15, ha='right')
ax2.set_ylabel("Top-1 Accuracy 변화량 [%]", fontsize=10)
ax2.set_title("독립 Vth 샘플링 시 Top-1 변화\n(각 기준 대비)", fontsize=11, fontweight='bold')
ax2.grid(True, axis='y', ls='--', alpha=0.3)

plt.tight_layout()
out = os.path.join(base, 'sim_vth_dual.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
