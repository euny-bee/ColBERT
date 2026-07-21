"""
q2("where is hartwell ga", margin=20.06, seed=19)만 단독으로 candidate stacked bar 재생성.
기존 [vth_v3_fixedcell] 데이터로 그렸던 bar chart는 현재 q2 scatter(plot_seed19_q2_r2.py)가 쓰는
[seed19] 데이터셋과 다른 시뮬레이션 run이라 숫자가 어긋남 -> [seed19]step3_*_candidate_pids.xlsx로 다시 그림.
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

FONT_BASE = 13

def _setup_matplotlib():
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": FONT_BASE,
        "font.weight": "bold", "axes.unicode_minus": False,
    })

_setup_matplotlib()

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\06_newq_margin'
Q_LABEL = 'q2: "where is hartwell ga" (margin=20.06, dominant win)'

COLORS = {
    'all3':      '#4CAF50',
    'dig_A':     '#90CAF9',
    'dig_C':     '#FFCC80',
    'dig_only':  '#78909C',
    'A_only':    '#2196F3',
    'C_only':    '#FF9800',
    'AC_only':   '#9C27B0',
}

print("데이터 로드 중 ([seed19] candidate pids)...")
dig_sheets  = pd.read_excel(f'{BASE}/[seed19]step3_digital_candidate_pids.xlsx', sheet_name=None)
optA_sheets = pd.read_excel(f'{BASE}/[seed19]step3_optA_candidate_pids.xlsx',   sheet_name=None)
optC_sheets = pd.read_excel(f'{BASE}/[seed19]step3_optC_candidate_pids.xlsx',   sheet_name=None)

dig_pids = set(dig_sheets['q2']['pid'])
A_pids   = set(optA_sheets['q2']['pid'])
C_pids   = set(optC_sheets['q2']['pid'])

s_all3     = dig_pids & A_pids & C_pids
s_dig_A    = (dig_pids & A_pids) - C_pids
s_dig_C    = (dig_pids & C_pids) - A_pids
s_dig_only = dig_pids - A_pids - C_pids
s_A_only   = A_pids - dig_pids - C_pids
s_C_only   = C_pids - dig_pids - A_pids
s_AC_only  = (A_pids & C_pids) - dig_pids

jac_A = len(dig_pids & A_pids) / len(dig_pids | A_pids)
jac_C = len(dig_pids & C_pids) / len(dig_pids | C_pids)

print(f"|dig|={len(dig_pids)} |A|={len(A_pids)} |C|={len(C_pids)}")
print(f"dig∩A∩C={len(s_all3)} dig∩A\\C={len(s_dig_A)} dig∩C\\A={len(s_dig_C)} "
      f"dig only={len(s_dig_only)} A only={len(s_A_only)} C only={len(s_C_only)} A∩C\\dig={len(s_AC_only)}")
print(f"Jaccard(Dig↔A)={jac_A:.3f}  Jaccard(Dig↔C)={jac_C:.3f}")

W = 0.28

def draw_stacked(ax, x, segments, total):
    b = 0
    for val, col in segments:
        if val > 0:
            ax.bar(x, val, bottom=b, color=col, width=W)
            txt_col = 'black' if col in (COLORS['dig_A'], COLORS['dig_C']) else 'white'
            ax.text(x, b + val / 2, str(val), ha='center', va='center',
                    fontsize=9, color=txt_col, fontweight='bold')
            b += val
    ax.text(x, total + 2, str(total), ha='center', fontsize=11, fontweight='bold')

fig, ax = plt.subplots(1, 1, figsize=(4.6, 5))

draw_stacked(ax, 0, [(len(s_dig_only), COLORS['dig_only']), (len(s_dig_A), COLORS['dig_A']),
                     (len(s_dig_C),   COLORS['dig_C']),    (len(s_all3),  COLORS['all3'])], len(dig_pids))
draw_stacked(ax, 1, [(len(s_A_only),   COLORS['A_only']),  (len(s_AC_only), COLORS['AC_only']),
                     (len(s_dig_A),   COLORS['dig_A']),    (len(s_all3),  COLORS['all3'])], len(A_pids))
draw_stacked(ax, 2, [(len(s_C_only),   COLORS['C_only']),  (len(s_AC_only), COLORS['AC_only']),
                     (len(s_dig_C),   COLORS['dig_C']),    (len(s_all3),  COLORS['all3'])], len(C_pids))

ax.text(0.5, -0.09,
        f'Jaccard(Dig↔A)={jac_A:.3f}   Jaccard(Dig↔C)={jac_C:.3f}',
        ha='center', transform=ax.transAxes, fontsize=9, color='dimgray', style='italic')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Digital', 'OptionA', 'OptionC'], fontsize=11, fontweight='bold')
ax.set_ylabel('Candidate passages', fontsize=12)
ax.set_title(Q_LABEL, fontsize=11, fontweight='bold')
ax.set_ylim(0, max(len(dig_pids), len(A_pids), len(C_pids)) * 1.22)
ax.grid(axis='y', ls='--', alpha=0.3)
ax.legend(handles=[
    mpatches.Patch(color=COLORS['all3'],     label='dig ∩ A ∩ C'),
    mpatches.Patch(color=COLORS['dig_A'],    label='dig ∩ A \\ C'),
    mpatches.Patch(color=COLORS['dig_C'],    label='dig ∩ C \\ A'),
    mpatches.Patch(color=COLORS['dig_only'], label='dig only'),
    mpatches.Patch(color=COLORS['A_only'],   label='A only'),
    mpatches.Patch(color=COLORS['C_only'],   label='C only'),
    mpatches.Patch(color=COLORS['AC_only'],  label='A ∩ C \\ dig'),
], fontsize=8, loc='upper right')

fig.tight_layout()
out = f'{BASE}/[seed19]q2_candidate_bar.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"저장: {out}")
