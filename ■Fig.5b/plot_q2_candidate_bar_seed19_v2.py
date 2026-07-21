"""
q2("where is hartwell ga", margin=20.06, seed=19)만 단독으로 candidate stacked bar 재생성.
plot_q2_candidate_bar_seed19.py의 복사본 - x축 라벨(Digital/Vth comp/No comp)과
y축 라벨 굵기·크기(x축 tick 라벨과 통일)만 수정.
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
SMALL_THRESH   = 8      # 이 값 미만인 조각은 막대 안에 숫자를 못 넣으므로 바깥으로 뺌
LEADER_X       = W / 2 + 0.08
LABEL_SPACING  = 11     # 바깥 라벨끼리 겹치지 않도록 띄우는 간격

def draw_stacked(ax, x, segments, total):
    b = 0
    small_labels = []
    for val, col in segments:
        if val > 0:
            ax.bar(x, val, bottom=b, color=col, width=W)
            center = b + val / 2
            if val < SMALL_THRESH:
                small_labels.append(center)
            else:
                txt_col = 'black' if col in (COLORS['dig_A'], COLORS['dig_C']) else 'white'
                ax.text(x, center, str(val), ha='center', va='center',
                        fontsize=9 * 1.2, color=txt_col, fontweight='bold')
            b += val

    if small_labels:
        small_labels.sort()
        vals = []
        # re-collect (center, val) pairs matching small_labels order
        b2 = 0
        for val, col in segments:
            if val > 0:
                center = b2 + val / 2
                if val < SMALL_THRESH:
                    vals.append((center, val))
                b2 += val
        n = len(vals)
        mid = sum(c for c, _ in vals) / n
        start = max(mid - LABEL_SPACING * (n - 1) / 2, 4)
        for i, (center, val) in enumerate(vals):
            target_y = start + i * LABEL_SPACING
            ax.plot([x + W / 2, x + LEADER_X], [center, target_y],
                    color='black', lw=0.7, clip_on=False)
            ax.text(x + LEADER_X + 0.02, target_y, str(val), ha='left', va='center',
                    fontsize=10, color='black', fontweight='bold', clip_on=False)

    ax.text(x, total + 2, str(total), ha='center', fontsize=11, fontweight='bold')

fig, ax = plt.subplots(1, 1, figsize=(4.6 * 1.1, 5 * 0.8))

draw_stacked(ax, 0, [(len(s_dig_only), COLORS['dig_only']), (len(s_dig_A), COLORS['dig_A']),
                     (len(s_dig_C),   COLORS['dig_C']),    (len(s_all3),  COLORS['all3'])], len(dig_pids))
draw_stacked(ax, 1, [(len(s_A_only),   COLORS['A_only']),  (len(s_AC_only), COLORS['AC_only']),
                     (len(s_dig_A),   COLORS['dig_A']),    (len(s_all3),  COLORS['all3'])], len(A_pids))
draw_stacked(ax, 2, [(len(s_C_only),   COLORS['C_only']),  (len(s_AC_only), COLORS['AC_only']),
                     (len(s_dig_C),   COLORS['dig_C']),    (len(s_all3),  COLORS['all3'])], len(C_pids))

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Digital', 'Vth comp', 'No comp'], fontsize=11 * 1.2 * 1.2, fontweight='bold')
ax.set_ylabel('Candidate passages', fontsize=11 * 1.2 * 1.2, fontweight='bold')
ax.set_title(Q_LABEL, fontsize=11, fontweight='bold')
ax.set_ylim(0, max(len(dig_pids), len(A_pids), len(C_pids)) * 1.22)
ax.grid(axis='y', ls='--', alpha=0.3)
legend_items = [
    (len(s_all3),     COLORS['all3'],     'D ∩ VC ∩ NC'),
    (len(s_dig_A),    COLORS['dig_A'],    'D ∩ VC \\ NC'),
    (len(s_dig_C),    COLORS['dig_C'],    'D ∩ NC \\ VC'),
    (len(s_dig_only), COLORS['dig_only'], 'D only'),
    (len(s_A_only),   COLORS['A_only'],   'VC only'),
    (len(s_C_only),   COLORS['C_only'],   'NC only'),
    (len(s_AC_only),  COLORS['AC_only'],  'VC ∩ NC \\ D'),
]

fig.tight_layout()
out = f'{BASE}/[seed19]q2_candidate_bar_v2.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"저장: {out}")

# --- 범례만 별도 이미지로 ---
fig_leg, ax_leg = plt.subplots(figsize=(2.6, 1.6))
ax_leg.axis('off')
ax_leg.legend(handles=[mpatches.Patch(color=col, label=lbl) for cnt, col, lbl in legend_items if cnt > 0],
              fontsize=9, loc='center', frameon=True, edgecolor='black')
fig_leg.tight_layout()
leg_out = f'{BASE}/[seed19]q2_candidate_bar_v2_legend.png'
fig_leg.savefig(leg_out, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig_leg)
print(f"저장: {leg_out}")
