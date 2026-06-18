"""
Improved visualization for analog MaxSim simulation (scale x1)

6 panels:
  P1  Centroid selection agreement scatter
  P2  Score fidelity (ideal top-1 score vs analog-selected score)
  P3  Top-k accuracy horizontal bar
  P4  V-I curve + actual operating-point distribution
  P5  Failure margin analysis (why 5% fail)
  P6  Spearman rho boxplot per query
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import os

# -- font ------------------------------------------------------------------
_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

# -- die4 circuit params ---------------------------------------------------
L_f = 6.112020;  K_f = 2.731949;  B_f = -10.713911
Vth = 0.151045;  VDS_m = 1.0;     VDS_t = 1.7

def logistic_ids(vgs):
    return 10 ** (B_f + L_f / (1 + np.exp(-K_f * (vgs - Vth))))

def vds_correction(vgs_arr):
    VoD = vgs_arr - Vth
    factor = np.ones_like(VoD, dtype=float)
    on = VoD > 0
    Vo = VoD[on]
    Im = np.where(Vo > VDS_m, Vo*VDS_m - VDS_m**2/2, Vo**2/2)
    It = np.where(Vo > VDS_t, Vo*VDS_t - VDS_t**2/2, Vo**2/2)
    factor[on] = np.where(Im > 0, It/Im, 1.0)
    return factor

def calc_ids(vgs_arr):
    return logistic_ids(vgs_arr) * vds_correction(vgs_arr)

# -- load data -------------------------------------------------------------
base = r'C:\Users\nmdl-khb\ColBERT\centroidset'
C = pd.read_excel(os.path.join(base, 'centroids_100x128.xlsx'),
                  header=0, index_col=0).values.astype(float)   # (100, 128)
Q = pd.read_excel(os.path.join(base, 'query_embs_96x128.xlsx'),
                  header=0, index_col=0).values.astype(float)   # (96,  128)

# -- ideal dot product (scale x1) -----------------------------------------
dot_matrix   = Q @ C.T                           # (96, 100)
best_ideal   = np.argmax(dot_matrix, axis=1)     # (96,)
sorted_ideal = np.argsort(-dot_matrix, axis=1)   # (96, 100) descending

# -- analog simulation (scale x1) ----------------------------------------
diff    = np.abs(Q[:, np.newaxis, :] - C[np.newaxis, :, :])  # (96,100,128)
VGS_arr = diff + Vth
I_total = calc_ids(VGS_arr).sum(axis=2) * 1e6    # (96, 100) [uA]
best_analog = np.argmin(I_total, axis=1)          # (96,)

# -- derived metrics -------------------------------------------------------
success_mask = (best_ideal == best_analog)
n_ok  = success_mask.sum()
n_all = len(success_mask)

ks   = [1, 2, 3, 5, 10]
topk = {k: np.mean([best_analog[i] in sorted_ideal[i, :k]
                    for i in range(n_all)]) for k in ks}

rhos = np.array([spearmanr(dot_matrix[i], -I_total[i])[0]
                 for i in range(n_all)])

# score of the centroid each method selects
ideal_top1_score   = dot_matrix[np.arange(n_all), best_ideal]
analog_sel_score   = dot_matrix[np.arange(n_all), best_analog]
score_loss         = ideal_top1_score - analog_sel_score   # 0 when correct

# margin: how close top-1 and top-2 are in ideal space
margin = (dot_matrix[np.arange(n_all), sorted_ideal[:, 0]] -
          dot_matrix[np.arange(n_all), sorted_ideal[:, 1]])

# query coloring
Q_COLORS = np.array(['#2196F3']*32 + ['#4CAF50']*32 + ['#FF9800']*32)
Q_NAMES  = ['q0', 'q1', 'q2']
Q_COL3   = ['#2196F3', '#4CAF50', '#FF9800']

# ── FIGURE ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 11))
fig.suptitle(
    f"Analog MaxSim vs Ideal Dot Product  —  die4 Option A  "
    f"(Vth={Vth:.3f}V, VDS=1.7V, scale x1)\n"
    f"Top-1: {n_ok}/{n_all} = {n_ok/n_all*100:.1f}%     "
    f"Top-3: {topk[3]*100:.0f}%     "
    f"Top-5: {topk[5]*100:.0f}%     "
    f"Avg Spearman rho: {rhos.mean():.4f}",
    fontsize=12, fontweight="bold", y=0.99
)
gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.33,
                       left=0.07, right=0.97, top=0.92, bottom=0.07)

# ── P1: Centroid selection agreement ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot([-2, 101], [-2, 101], 'k--', lw=1, alpha=0.35, zorder=0)
ax1.scatter(best_ideal[success_mask], best_analog[success_mask],
            c=Q_COLORS[success_mask], s=28, alpha=0.85, zorder=2, edgecolors='none')
ax1.scatter(best_ideal[~success_mask], best_analog[~success_mask],
            c='red', s=100, marker='X', linewidths=1.5, zorder=4,
            edgecolors='darkred', label=f'Error ({(~success_mask).sum()})')
ax1.set_xlim(-2, 103); ax1.set_ylim(-2, 103)
ax1.set_xlabel("Ideal best centroid index", fontsize=9)
ax1.set_ylabel("Analog best centroid index", fontsize=9)
ax1.set_title(f"Centroid Selection Agreement  ({n_ok/n_all*100:.1f}%)",
              fontsize=10, fontweight='bold')
legend_handles = (
    [Patch(color=c, label=n) for c, n in zip(Q_COL3, Q_NAMES)] +
    [Line2D([0],[0], marker='X', color='w', markerfacecolor='red',
            markeredgecolor='darkred', markersize=9,
            label=f'Error ({(~success_mask).sum()})')]
)
ax1.legend(handles=legend_handles, fontsize=8, loc='upper left')
ax1.grid(True, ls='--', alpha=0.25)

# ── P2: Score fidelity ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
lo = min(ideal_top1_score.min(), analog_sel_score.min()) - 0.01
hi = max(ideal_top1_score.max(), analog_sel_score.max()) + 0.01
ax2.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.4)
ax2.scatter(ideal_top1_score[success_mask], analog_sel_score[success_mask],
            c=Q_COLORS[success_mask], s=28, alpha=0.85, edgecolors='none',
            label='Correct')
ax2.scatter(ideal_top1_score[~success_mask], analog_sel_score[~success_mask],
            c='red', s=100, marker='X', linewidths=1.5, zorder=5,
            edgecolors='darkred', label=f'Error')
ax2.set_xlabel("Ideal top-1 dot product score", fontsize=9)
ax2.set_ylabel("Analog-selected centroid's\ndot product score", fontsize=9)
ax2.set_title("Score Fidelity\n(points on diagonal = correct)", fontsize=10, fontweight='bold')
ax2.grid(True, ls='--', alpha=0.25)
err_loss = score_loss[~success_mask]
info = (f"Score loss at errors:\n"
        f"  mean = {err_loss.mean():.4f}\n"
        f"  max  = {err_loss.max():.4f}")
ax2.text(0.04, 0.97, info, transform=ax2.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
ax2.legend(handles=[Patch(color='gray', label='Correct'),
                    Line2D([0],[0], marker='X', color='w',
                           markerfacecolor='red', markeredgecolor='darkred',
                           markersize=8, label='Error')],
           fontsize=8, loc='lower right')

# ── P3: Top-k accuracy horizontal bar ────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
k_labels = [f'Top-{k}' for k in ks]
k_vals   = [topk[k]*100 for k in ks]
bar_colors = ['#c0392b', '#e67e22', '#f1c40f', '#27ae60', '#16a085']
bars = ax3.barh(k_labels, k_vals, color=bar_colors, alpha=0.88, height=0.52)
for bar, val in zip(bars, k_vals):
    ax3.text(min(val+1, 116), bar.get_y()+bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
ax3.set_xlim(0, 118)
ax3.set_xlabel("Accuracy [%]", fontsize=9)
ax3.set_title("Top-k Accuracy  (scale x1)", fontsize=10, fontweight='bold')
ax3.axvline(100, color='gray', ls='--', lw=1.2, alpha=0.5)
ax3.grid(True, axis='x', ls='--', alpha=0.25)
ax3.text(0.98, 0.04,
         f"Q: {Q.shape[0]} tokens x {Q.shape[1]} dim\n"
         f"C: {C.shape[0]} centroids\n"
         f"emb range: +/-{max(abs(Q.min()),abs(Q.max())):.2f} V\n"
         f"VGS range: {Vth:.2f}~{diff.max()+Vth:.2f} V",
         transform=ax3.transAxes, fontsize=7.5, va='bottom', ha='right',
         bbox=dict(boxstyle='round', fc='lightcyan', alpha=0.9))

# ── P4: V-I curve + operating-point distribution ─────────────────────────
ax4  = fig.add_subplot(gs[1, 0])
ax4r = ax4.twinx()

x_vi  = np.linspace(0, 2.0, 600)
ids_vi = calc_ids(x_vi + Vth) * 1e6
ax4.plot(x_vi, ids_vi, color='navy', lw=2.5, label='V-I (Option A)')
ax4.set_ylabel("IDS  [uA]", fontsize=9, color='navy')
ax4.tick_params(axis='y', labelcolor='navy')
ax4.set_ylim(0, 35); ax4.set_xlim(0, 2)
ax4.set_xlabel("|V2-V1|  [V]", fontsize=9)

diff_flat = diff.flatten()
ax4r.hist(diff_flat, bins=100, color='coral', alpha=0.45, label='|Q-C| dist.')
ax4r.set_ylabel("Count (all 96x100x128 pairs)", fontsize=8, color='coral')
ax4r.tick_params(axis='y', labelcolor='coral')

p95 = np.percentile(diff_flat, 95)
ax4.axvline(p95, color='firebrick', ls=':', lw=1.8)
ax4.text(p95+0.02, 30, f'95th pct\n{p95:.3f}V', fontsize=7.5, color='firebrick')

ax4.set_title("V-I Curve & Operating-Point Distribution", fontsize=10, fontweight='bold')
h1, l1 = ax4.get_legend_handles_labels()
h2, l2 = ax4r.get_legend_handles_labels()
ax4.legend(h1+h2, l1+l2, fontsize=8, loc='upper left')

# ── P5: Failure margin analysis ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
bins = np.linspace(0, margin.max()+0.005, 35)
ax5.hist(margin[success_mask], bins=bins, color='#2ecc71', alpha=0.72,
         edgecolor='white', label=f'Correct ({success_mask.sum()})')
ax5.hist(margin[~success_mask], bins=bins, color='#e74c3c', alpha=0.88,
         edgecolor='white', label=f'Error ({(~success_mask).sum()})')
ax5.set_xlabel("Score margin  (top-1 minus top-2 dot product)", fontsize=9)
ax5.set_ylabel("Count (query tokens)", fontsize=9)
ax5.set_title("Why Does Analog Fail?\n(Ideal Score Margin Analysis)", fontsize=10, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, ls='--', alpha=0.25)
ax5.text(0.97, 0.95,
         "Errors cluster at\nsmall margin\n(near-tie centroids\nare hard to separate)",
         transform=ax5.transAxes, fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

# ── P6: Spearman rho boxplot per query ──────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
q_rho_groups = [rhos[:32], rhos[32:64], rhos[64:]]
bplot = ax6.boxplot(q_rho_groups, patch_artist=True, widths=0.45,
                    medianprops=dict(color='black', lw=2.5),
                    whiskerprops=dict(lw=1.5),
                    capprops=dict(lw=1.5))
for patch, col in zip(bplot['boxes'], Q_COL3):
    patch.set_facecolor(col); patch.set_alpha(0.7)
rng = np.random.default_rng(42)
for i, (grp, col) in enumerate(zip(q_rho_groups, Q_COL3)):
    jitter = rng.uniform(-0.16, 0.16, len(grp))
    ax6.scatter(np.full(len(grp), i+1) + jitter, grp,
                color=col, s=18, alpha=0.75, zorder=3, edgecolors='none')
    ax6.text(i+1, rhos.min()-0.008, f'mean\n{grp.mean():.4f}',
             ha='center', fontsize=8.5, color='black')
ax6.set_xticklabels(Q_NAMES, fontsize=11)
ax6.set_ylabel("Spearman rho  (dot vs -I_total)", fontsize=9)
ax6.set_title("Ranking Correlation per Query", fontsize=10, fontweight='bold')
ax6.set_ylim(rhos.min()-0.025, 1.01)
ax6.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.4)
ax6.grid(True, axis='y', ls='--', alpha=0.25)

out = os.path.join(base, "sim_maxsim_v2.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")
