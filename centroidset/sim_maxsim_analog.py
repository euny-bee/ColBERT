"""
Analog MaxSim simulation using die4 logistic circuit model (Option A)

Circuit:  VGS = |Q[i,k] - C[j,k]| + Vth  (Option A, Vth compensation)
          IDS = logistic(VGS, Vth) * vds_correction(VGS, VDS=1.7V)
Match line: I_total[i,j] = sum_k IDS_k   (KCL, 128 cells per row)

MaxSim:
  Ideal  -> best centroid = argmax_j dot(Q[i], C[j])
  Analog -> best centroid = argmin_j I_total[i,j]  (min I = min dist = max sim)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr
import os

# -- font ------------------------------------------------------------------
_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# -- die4 circuit parameters (VDS=1V fitting, corrected to 1.7V) -----------
L_f   =  6.112020
K_f   =  2.731949
B_f   = -10.713911
Vth   =  0.151045
VDS_m =  1.0    # measured (fit) VDS
VDS_t =  1.7    # target VDS

def logistic_ids(vgs):
    return 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - Vth))))

def vds_correction(vgs_arr):
    VoD = vgs_arr - Vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > VDS_m, Vo * VDS_m - VDS_m**2 / 2, Vo**2 / 2)
    It = np.where(Vo > VDS_t, Vo * VDS_t - VDS_t**2 / 2, Vo**2 / 2)
    factor[on] = np.where(Im > 0, It / Im, 1.0)
    return factor

def calc_ids(vgs_arr):
    return logistic_ids(vgs_arr) * vds_correction(vgs_arr)   # [A]

# -- load data -------------------------------------------------------------
base   = r'C:\Users\nmdl-khb\ColBERT\centroidset'
C = pd.read_excel(os.path.join(base, 'centroids_100x128.xlsx'),
                  header=0, index_col=0).values.astype(float)    # (100, 128)
Q = pd.read_excel(os.path.join(base, 'query_embs_96x128.xlsx'),
                  header=0, index_col=0).values.astype(float)    # (96,  128)

print(f"C: {C.shape}  Q: {Q.shape}")
print(f"Embedding range  C: [{C.min():.4f}, {C.max():.4f}]  "
      f"Q: [{Q.min():.4f}, {Q.max():.4f}]")
print(f"L2 norms  C: {np.linalg.norm(C, axis=1).mean():.4f}  "
      f"Q: {np.linalg.norm(Q, axis=1).mean():.4f}\n")

# -- ideal dot product (scale-independent ranking) -------------------------
dot_matrix = Q @ C.T                         # (96, 100)
best_ideal  = np.argmax(dot_matrix, axis=1)  # (96,)
sorted_ideal = np.argsort(-dot_matrix, axis=1)  # (96, 100) descending

# -- analog simulation per voltage scale -----------------------------------
SCALES = [1, 2, 3]
scale_colors = ['#2196F3', '#4CAF50', '#FF9800']
results = {}

print("=" * 58)
for scale in SCALES:
    Qs = Q * scale   # (96, 128)
    Cs = C * scale   # (100, 128)

    # diff[i,j,k] = |Qs[i,k] - Cs[j,k]|  ->  (96, 100, 128)
    diff     = np.abs(Qs[:, np.newaxis, :] - Cs[np.newaxis, :, :])
    VGS_arr  = diff + Vth                  # Option A: VGS = |V2-V1| + Vth
    I_arr    = calc_ids(VGS_arr)           # (96, 100, 128) [A]
    I_total  = I_arr.sum(axis=2) * 1e6    # (96, 100) [uA]

    best_analog = np.argmin(I_total, axis=1)   # (96,)

    # top-k accuracy
    topk = {}
    for k in [1, 2, 3, 5, 10]:
        hit = np.array([best_analog[i] in sorted_ideal[i, :k] for i in range(96)])
        topk[k] = hit.mean()

    # per-query top-1
    slices = [slice(0, 32), slice(32, 64), slice(64, 96)]
    per_q = [np.mean(best_ideal[sl] == best_analog[sl]) for sl in slices]

    # Spearman rho per token
    rhos = [spearmanr(dot_matrix[i], -I_total[i])[0] for i in range(96)]

    results[scale] = dict(
        I_total=I_total,
        best_analog=best_analog,
        topk=topk,
        per_q=per_q,
        rhos=rhos,
    )

    vrange = 0.34 * scale   # approx max embedding magnitude after scaling
    print(f"Scale x{scale}  (|V2-V1| max ~ {2*vrange:.2f} V,  VGS max ~ {2*vrange+Vth:.2f} V)")
    print(f"  Top-1 : {topk[1]*100:5.1f}%   Top-3: {topk[3]*100:5.1f}%   "
          f"Top-5: {topk[5]*100:5.1f}%   Top-10: {topk[10]*100:5.1f}%")
    print(f"  Per query  q0:{per_q[0]*100:.1f}%  q1:{per_q[1]*100:.1f}%  "
          f"q2:{per_q[2]*100:.1f}%")
    print(f"  Avg Spearman rho: {np.mean(rhos):.4f}\n")

# -- visualization ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f"Analog MaxSim Simulation  (die4 Option A,  Vth={Vth:.3f}V,  VDS=1.7V)\n"
    f"Q: {Q.shape[0]} tokens x {Q.shape[1]} dims   C: {C.shape[0]} centroids",
    fontsize=11, fontweight="bold"
)

# ── panel 1: dot product heatmap ─────────────────────────────────────────
ax = axes[0, 0]
im = ax.imshow(dot_matrix, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
for i in range(96):
    ax.plot(best_ideal[i], i, 'k.', markersize=1.5)
ax.set_title("Ideal: Dot Product Score", fontsize=10)
ax.set_xlabel("Centroid index", fontsize=9)
ax.set_ylabel("Query token index", fontsize=9)
# query boundaries
for y in [32, 64]:
    ax.axhline(y - 0.5, color='lime', lw=1.2)
ax.text(102, 16, 'q0', fontsize=8, va='center', color='lime')
ax.text(102, 48, 'q1', fontsize=8, va='center', color='lime')
ax.text(102, 80, 'q2', fontsize=8, va='center', color='lime')
plt.colorbar(im, ax=ax, shrink=0.8)

# ── panel 2: analog heatmap (scale x1) ──────────────────────────────────
ax = axes[0, 1]
I1 = results[1]['I_total']
im = ax.imshow(I1, aspect='auto', cmap='RdBu_r')
for i in range(96):
    ax.plot(results[1]['best_analog'][i], i, 'k.', markersize=1.5)
for y in [32, 64]:
    ax.axhline(y - 0.5, color='lime', lw=1.2)
ax.set_title("Analog: Match Line Current [uA]  (scale x1)", fontsize=10)
ax.set_xlabel("Centroid index", fontsize=9)
ax.set_ylabel("Query token index", fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.8)

# ── panel 3: scatter dot vs analog (scale x1, x2, x3) ──────────────────
ax = axes[0, 2]
dot_flat = dot_matrix.flatten()
for scale, col in zip(SCALES, scale_colors):
    I_flat = results[scale]['I_total'].flatten()
    ax.scatter(dot_flat, I_flat, s=0.4, alpha=0.25, color=col,
               label=f'x{scale}')
ax.set_xlabel("Dot Product Score", fontsize=9)
ax.set_ylabel("Analog Current [uA]", fontsize=9)
ax.set_title("Dot Product vs Analog Current", fontsize=10)
ax.grid(True, ls='--', alpha=0.3)
# overall spearman per scale
for scale, col in zip(SCALES, scale_colors):
    I_flat = results[scale]['I_total'].flatten()
    rho, _ = spearmanr(dot_flat, -I_flat)
    ax.text(0.03, 0.97 - SCALES.index(scale) * 0.08,
            f'x{scale}: rho={rho:.4f}',
            transform=ax.transAxes, fontsize=8, va='top', color=col,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))
ax.legend(fontsize=8, markerscale=6)

# ── panel 4: top-k accuracy ──────────────────────────────────────────────
ax = axes[1, 0]
ks = [1, 2, 3, 5, 10]
x = np.arange(len(ks))
w = 0.25
for idx, (scale, col) in enumerate(zip(SCALES, scale_colors)):
    accs = [results[scale]['topk'][k] * 100 for k in ks]
    bars = ax.bar(x + idx * w, accs, w, label=f'scale x{scale}',
                  color=col, alpha=0.85)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8,
                f'{v:.0f}', ha='center', fontsize=6.5)
ax.set_xticks(x + w)
ax.set_xticklabels([f'Top-{k}' for k in ks])
ax.set_ylabel("Accuracy [%]", fontsize=9)
ax.set_title("Top-k Accuracy vs Voltage Scale", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0, 112)
ax.grid(True, axis='y', ls='--', alpha=0.3)

# ── panel 5: per-query top-1 ─────────────────────────────────────────────
ax = axes[1, 1]
q_labels = ['q0\n(t0-31)', 'q1\n(t0-31)', 'q2\n(t0-31)']
x = np.arange(3)
for idx, (scale, col) in enumerate(zip(SCALES, scale_colors)):
    vals = [v * 100 for v in results[scale]['per_q']]
    bars = ax.bar(x + idx * w, vals, w, label=f'scale x{scale}',
                  color=col, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8,
                f'{v:.0f}', ha='center', fontsize=7)
ax.set_xticks(x + w)
ax.set_xticklabels(q_labels, fontsize=9)
ax.set_ylabel("Top-1 Accuracy [%]", fontsize=9)
ax.set_title("Top-1 Accuracy per Query", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0, 112)
ax.grid(True, axis='y', ls='--', alpha=0.3)

# ── panel 6: Spearman rho distribution ───────────────────────────────────
ax = axes[1, 2]
for scale, col in zip(SCALES, scale_colors):
    rhos = results[scale]['rhos']
    ax.hist(rhos, bins=20, alpha=0.55, color=col,
            label=f'x{scale}  mean={np.mean(rhos):.3f}')
ax.set_xlabel("Spearman rho  (dot vs -I_total)", fontsize=9)
ax.set_ylabel("Count (query tokens)", fontsize=9)
ax.set_title("Ranking Correlation Distribution", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, ls='--', alpha=0.3)

plt.tight_layout()
out = os.path.join(base, "sim_maxsim_analog.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")
