"""
plot_q2_residual_violin_inset.py의 색상 변형본 -- Vth comp 파랑을 #2196F3 -> #1565C0(TOPSW "this work" 파랑)로 교체.
원본 파일/출력은 그대로 두고, 출력 파일명에 "_1565C0" suffix를 붙여 별도 저장.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from scipy import stats

def _setup_matplotlib():
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Malgun Gothic", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": 13,
        "font.weight": "bold", "axes.labelsize": 14, "axes.labelweight": "bold",
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.unicode_minus": False,
    })

_setup_matplotlib()

IN_BASE  = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
OUT_BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
C3 = {'optA': '#1565C0', 'optC': '#FF7811'}   # optA: #2196F3 -> #1565C0

step6 = pd.read_excel(f'{IN_BASE}/[seed19]step6_all_results.xlsx', sheet_name=None)
q_id = 2
df_dig = step6[f'q{q_id}_digital'].set_index('pid')
df_A   = step6[f'q{q_id}_optA'].set_index('pid')
df_C   = step6[f'q{q_id}_optC'].set_index('pid')

common_A = [p for p in df_dig.index if p in df_A.index]
xA = np.array([int(df_dig.loc[p, 'rank_dig_f32']) for p in common_A])
yA = np.array([int(df_A.loc[p, 'rank_optA'])      for p in common_A])
resA = yA - xA

common_C = [p for p in df_dig.index if p in df_C.index]
xC = np.array([int(df_dig.loc[p, 'rank_dig_f32']) for p in common_C])
yC = np.array([int(df_C.loc[p, 'rank_optC'])      for p in common_C])
resC = yC - xC

lev_stat, lev_p = stats.levene(resA, resC)
var_ratio = np.var(resC, ddof=1) / np.var(resA, ddof=1)
ks_stat, ks_p = stats.ks_2samp(resA, resC)

BASE_H = 5.5
BASE_W = 5.5

def draw_violin_panel(width_scale, out_path, *, title_fs, ylabel_fs, title_wrap=False):
    width_in = BASE_W * width_scale

    for _ in range(60):
        fig, ax = plt.subplots(1, 1, figsize=(width_in, BASE_H))

        data = [resA, resC]
        colors = [C3['optA'], C3['optC']]

        parts = ax.violinplot(data, positions=[0, 1], widths=0.7, showextrema=False)
        for pc, col in zip(parts['bodies'], colors):
            pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor(col); pc.set_linewidth(1.5)

        bp = ax.boxplot(data, positions=[0, 1], widths=0.15, patch_artist=True,
                         showfliers=False, medianprops=dict(color='black', lw=2))
        for box, col in zip(bp['boxes'], colors):
            box.set_facecolor('white'); box.set_edgecolor(col); box.set_linewidth(1.5)

        rng = np.random.default_rng(0)
        for i, (d, col) in enumerate(zip(data, colors)):
            jitter = rng.uniform(-0.06, 0.06, size=len(d))
            ax.scatter(np.full(len(d), i) + jitter, d, s=14, color=col, alpha=0.5, zorder=3)

        ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Vth\ncomp', 'No\ncomp'], fontsize=ylabel_fs)
        ax.tick_params(axis='y', labelsize=ylabel_fs)
        ax.set_ylabel('ΔRank (analog-digital)', fontsize=ylabel_fs, labelpad=0.5)

        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lbl0, lbl1 = ax.get_xticklabels()
        gap_px = lbl1.get_window_extent(renderer).x0 - lbl0.get_window_extent(renderer).x1

        MIN_GAP_PX = 20
        if gap_px >= MIN_GAP_PX:
            break
        extra_in = 2.0 * (MIN_GAP_PX - gap_px) / fig.dpi
        plt.close(fig)
        width_in += extra_in
    else:
        raise RuntimeError(f"xtick 라벨 겹침을 해소하지 못함: {out_path}")

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"저장: {out_path}  (figsize={width_in:.2f} x {BASE_H})")

draw_violin_panel(0.5, f'{OUT_BASE}/[newq_margin_seed19]q2_residual_violin_w50_1565C0.png',
                   title_fs=11, ylabel_fs=11 * 1.2 * 1.2 * 1.2 * 1.2, title_wrap=True)
draw_violin_panel(0.7, f'{OUT_BASE}/[newq_margin_seed19]q2_residual_violin_w70_1565C0.png',
                   title_fs=13, ylabel_fs=13 * 1.2 * 1.2 * 1.2 * 1.2, title_wrap=False)

# --- 범례만 별도 이미지로 ---
fig_leg, ax_leg = plt.subplots(figsize=(2.6, 0.9))
ax_leg.axis('off')
handles = [
    mpatches.Patch(facecolor=C3['optA'], edgecolor=C3['optA'], alpha=0.5, label='Vth Comp'),
    mpatches.Patch(facecolor=C3['optC'], edgecolor=C3['optC'], alpha=0.5, label='No Comp'),
]
ax_leg.legend(handles=handles, loc='center', ncol=2, frameon=False, fontsize=13)
fig_leg.tight_layout()
leg_out = f'{OUT_BASE}/[newq_margin_seed19]q2_residual_legend_1565C0.png'
fig_leg.savefig(leg_out, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig_leg)
print(f"저장: {leg_out}")

# --- 통계 텍스트박스만 별도 이미지로 ---
fig_txt, ax_txt = plt.subplots(figsize=(3.4, 1.5))
ax_txt.axis('off')
stats_txt = (f"std  Vth Comp = {resA.std(ddof=1):.1f}\n"
             f"std  No Comp  = {resC.std(ddof=1):.1f}\n"
             f"Levene  p = {lev_p:.2e}")
ax_txt.text(0.5, 0.5, stats_txt, transform=ax_txt.transAxes, fontsize=11, va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.6', alpha=0.9))
fig_txt.tight_layout()
stats_out = f'{OUT_BASE}/[newq_margin_seed19]q2_residual_stats_box_1565C0.png'
fig_txt.savefig(stats_out, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig_txt)
print(f"저장: {stats_out}")

from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox

heading_fs = 11 * 1.2
body_fs = 11
body_txt = (f"Vth comp = {resA.std(ddof=1):.1f}\n"
            f"No comp  = {resC.std(ddof=1):.1f}")

def draw_stats_box(edge_color, out_path):
    fig, ax = plt.subplots(figsize=(3.4, 1.2))
    ax.axis('off')
    heading_ta = TextArea("Std dev", textprops=dict(fontsize=heading_fs, fontweight='bold', ha='center'))
    body_ta = TextArea(body_txt, textprops=dict(fontsize=body_fs, fontweight='bold', ha='center', multialignment='center'))
    pack = VPacker(children=[heading_ta, body_ta], pad=0, sep=4, align='center')
    ab = AnnotationBbox(pack, (0.5, 0.5), xycoords='axes fraction', frameon=True, box_alignment=(0.5, 0.5),
                         bboxprops=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=edge_color, alpha=0.9))
    ax.add_artist(ab)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"저장: {out_path}")

draw_stats_box('0.6', f'{OUT_BASE}/[newq_margin_seed19]q2_residual_stats_box_no_levene_1565C0.png')
draw_stats_box('black', f'{OUT_BASE}/[newq_margin_seed19]q2_residual_stats_box_no_levene_blackedge_1565C0.png')
