"""
newq_margin pool, Vth SEED=19 결과 중 q2("where is hartwell ga")만 단독 패널로 시각화하고,
y=x 기준 산포 지표(R², NSE 방식: 1 - SS_res/SS_tot, 기준선을 y=x로 고정)를
Vth Comp / No Comp 계열별로 계산해 패널에 표기.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm

FONT_BASE, FONT_AXIS, FONT_PANEL, GRID_ALPHA = 13, 14, 15, 0.3

def _setup_matplotlib():
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": FONT_BASE,
        "font.weight": "bold", "axes.labelsize": FONT_AXIS, "axes.labelweight": "bold",
        "axes.titlesize": FONT_PANEL, "axes.titleweight": "bold",
        "xtick.labelsize": FONT_BASE, "ytick.labelsize": FONT_BASE, "axes.unicode_minus": False,
    })

def _style_ax(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8, ls="--")
    ax.minorticks_off()

def r2_vs_yx(x, y):
    """R^2 relative to the fixed y=x line (Nash-Sutcliffe style):
    1 - SS_res/SS_tot, SS_res=sum((y-x)^2), SS_tot=sum((y-mean(y))^2)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

_setup_matplotlib()

IN_BASE  = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
OUT_BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
QUERY_RELEVANT = {0: 471602, 1: 822108, 2: 347885}
Q_LABELS = [
    'q0: "...house reshingled" (margin=1.62, seed=19)',
    'q1: "wave amplitude def." (margin=11.35, seed=19)',
    'q2: "where is hartwell ga" (margin=20.06, seed=19)',
]

C3 = {'optA': '#2196F3', 'optC': '#FF7811', 'true': '#E91E63'}

print("데이터 로드 중...")
step6 = pd.read_excel(f'{IN_BASE}/[seed19]step6_all_results.xlsx', sheet_name=None)

q_id = 2
dig_rank  = step6[f'q{q_id}_digital'].set_index('pid')
optA_rank = step6[f'q{q_id}_optA'].set_index('pid')
optC_rank = step6[f'q{q_id}_optC'].set_index('pid')

print("q2 단독 scatter + R2(y=x) 계산 중...")
fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))

df_dig, df_A, df_C = dig_rank, optA_rank, optC_rank

common_A = [p for p in df_dig.index if p in df_A.index]
xA = [int(df_dig.loc[p, 'rank_dig_f32']) for p in common_A]
yA = [int(df_A.loc[p, 'rank_optA'])      for p in common_A]
ax.scatter(xA, yA, c=C3['optA'], s=20, alpha=0.85, zorder=4, clip_on=False)

common_C = [p for p in df_dig.index if p in df_C.index]
xC = [int(df_dig.loc[p, 'rank_dig_f32']) for p in common_C]
yC = [int(df_C.loc[p, 'rank_optC'])      for p in common_C]
ax.scatter(xC, yC, c=C3['optC'], s=20, alpha=0.85, zorder=4, marker='D', clip_on=False)

r2_A = r2_vs_yx(xA, yA)
r2_C = r2_vs_yx(xC, yC)
print(f"R2(y=x)  Vth Comp(A): {r2_A:.4f}   No Comp(C): {r2_C:.4f}")

max_rank = max(xA + yA + xC + yC + [10])
lim = max_rank * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.3)
ax.set_xlabel('Digital rank', fontsize=18)
ax.set_ylabel('Analog rank', fontsize=18)
ax.tick_params(labelsize=17)
ax.set_title(f'{Q_LABELS[q_id]}\ncommon w/ A: {len(common_A)}  |  w/ C: {len(common_C)}', fontsize=12)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
_style_ax(ax)

fig.tight_layout()
out = f'{OUT_BASE}/[newq_margin_seed19]q2_scatter_r2.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"저장: {out}")

# --- R^2 텍스트박스만 별도 이미지로 ---
fig_r2, ax_r2 = plt.subplots(figsize=(3.6, 1.1))
ax_r2.axis('off')
r2_text = f"$R^2_{{y=x}}$  Vth Comp = {r2_A:.3f}\n$R^2_{{y=x}}$  No Comp = {r2_C:.3f}"
ax_r2.text(0.5, 0.5, r2_text, transform=ax_r2.transAxes, fontsize=13, fontweight='bold',
           va='center', ha='center', color='black',
           bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.6', alpha=0.9))
fig_r2.tight_layout()
r2_out = f'{OUT_BASE}/[newq_margin_seed19]q2_scatter_r2_textbox.png'
fig_r2.savefig(r2_out, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig_r2)
print(f"저장: {r2_out}")

# --- 범례만 별도 이미지로 ---
fig_leg, ax_leg = plt.subplots(figsize=(4.6, 0.9))
ax_leg.axis('off')
ax_leg.legend(handles=[
    mlines.Line2D([], [], color=C3['optA'], marker='o', ms=7, ls='None', label='Vth Comp'),
    mlines.Line2D([], [], color=C3['optC'], marker='D', ms=6, ls='None', label='No Comp'),
    mlines.Line2D([], [], color='black', ls='--', lw=0.8, alpha=0.5, label='y = x'),
], fontsize=13, loc='center', ncol=3, frameon=False)
fig_leg.tight_layout()
leg_out = f'{OUT_BASE}/[newq_margin_seed19]q2_scatter_r2_legend.png'
fig_leg.savefig(leg_out, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig_leg)
print(f"저장: {leg_out}")

# --- 범례(세로 배치, 흰 배경 + 검은 테두리 버전) ---
fig_leg2, ax_leg2 = plt.subplots(figsize=(2.0, 2.0))
ax_leg2.axis('off')
leg2 = ax_leg2.legend(handles=[
    mlines.Line2D([], [], color=C3['optA'], marker='o', ms=7, ls='None', label='Vth comp'),
    mlines.Line2D([], [], color=C3['optC'], marker='D', ms=6, ls='None', label='No comp'),
    mlines.Line2D([], [], color='black', ls='--', lw=0.8, alpha=0.5, label='y = x'),
], fontsize=13, loc='center', ncol=1, frameon=True)
leg2.get_frame().set_facecolor('white')
leg2.get_frame().set_edgecolor('black')
leg2.get_frame().set_linewidth(1.2)
fig_leg2.tight_layout()
leg_out2 = f'{OUT_BASE}/[newq_margin_seed19]q2_scatter_r2_legend_vertical_boxed.png'
fig_leg2.savefig(leg_out2, dpi=300, bbox_inches='tight')
plt.close(fig_leg2)
print(f"저장: {leg_out2}")
