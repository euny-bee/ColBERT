"""
newq_margin2_top2/top5/top10 3개 폴더 결과를 모아, 경쟁문서 밀도(top2/top5/top10)에 따른
Digital / Vth Comp(Option A) / No Comp(Option C) true_pid rank 추이를 query별로 그린다.

  q0: qid 581521 "what can cause the right side of your back to hurt"  true_pid=493988  margin=1.76
  q1: qid 579133 "what blood stream messenger"                        true_pid=303786  margin=1.94
  q2: qid 690508 "what is a medical mc provider"                      true_pid=178693  margin=2.01

색상은 같은 논문 figure 파이프라인(newq_margin/plot_combined_split_newline_fixedcell.py)의
C3 팔레트를 그대로 재사용 (Digital=녹색, Vth Comp=파랑, No Comp=주황) -- 기존 figure들과의
시각적 일관성 유지.
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as fm

BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\08_newq_margin2'
FOLDERS = ['newq_margin2_top2', 'newq_margin2_top5', 'newq_margin2_top10']
X_LABELS = ['top2\n(1 forced)', 'top5\n(4 forced)', 'top10\n(10 forced)']

Q_LABELS = {
    'q0': 'q0: "...right side of your back hurt"\n(margin=1.76)',
    'q1': 'q1: "blood stream messenger"\n(margin=1.94)',
    'q2': 'q2: "medical mc provider"\n(margin=2.01)',
}

C3 = {'optA': '#2196F3', 'optC': '#FF7811', 'dig': '#4CAF50'}

FONT_BASE, FONT_AXIS, FONT_PANEL = 13, 14, 15

def _setup_matplotlib():
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    sans,
        "font.size":          FONT_BASE,
        "font.weight":        "bold",
        "axes.labelsize":     FONT_AXIS,
        "axes.labelweight":   "bold",
        "axes.titlesize":     FONT_PANEL,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    FONT_BASE,
        "ytick.labelsize":    FONT_BASE,
        "axes.unicode_minus": False,
    })

_setup_matplotlib()

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("데이터 로드 중...")
rows = []
for depth_idx, folder in enumerate(FOLDERS):
    df = pd.read_excel(f'{BASE}/{folder}/[vth_v3_fixedcell]step6_all_results.xlsx', sheet_name='summary')
    for _, r in df.iterrows():
        rows.append({
            'depth_idx': depth_idx,
            'query': r['query'],
            'rank_dig': r['rank_digital'],
            'rank_A':   r['rank_optA'],
            'rank_C':   r.get('rank_optC', np.nan),
            'n_A':      r['n_optA'],
            'n_C':      r['n_optC'],
        })
data = pd.DataFrame(rows)
print(data)

# ── Figure: 1x3 line plot (query별 패널) ─────────────────────────────────────
print("\nFigure 그리는 중...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for panel_idx, q_id in enumerate(['q0', 'q1', 'q2']):
    ax = axes[panel_idx]
    sub = data[data['query'] == q_id].sort_values('depth_idx')
    x = sub['depth_idx'].to_numpy()

    ax.plot(x, sub['rank_dig'], color=C3['dig'], marker='o', ms=8, lw=2, label='Digital', zorder=3)
    ax.plot(x, sub['rank_A'],   color=C3['optA'], marker='o', ms=8, lw=2, label='Vth Comp', zorder=4)

    # No Comp: 후보 탈락(NaN) 구간은 축 상단에 X 마커 + 점선으로 표시
    y_C = sub['rank_C'].to_numpy(dtype=float)
    n_C = sub['n_C'].to_numpy(dtype=float)
    valid = ~np.isnan(y_C)
    ax.plot(x[valid], y_C[valid], color=C3['optC'], marker='o', ms=8, lw=2, label='No Comp', zorder=5)

    if (~valid).any():
        y_top = np.nanmax([np.nanmax(y_C[valid]) if valid.any() else 1, np.nanmax(n_C)]) * 1.12
        for xi, is_drop, ni in zip(x, ~valid, n_C):
            if is_drop:
                ax.plot(xi, y_top, color=C3['optC'], marker='x', ms=14, mew=3, zorder=6, clip_on=False)
                ax.annotate('dropped', (xi, y_top), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=9, color=C3['optC'], fontweight='bold')
        # 마지막 유효 지점에서 탈락 지점까지 점선으로 연결 (끊김을 시각적으로 표현)
        last_valid_i = np.where(valid)[0]
        if len(last_valid_i):
            li = last_valid_i[-1]
            drop_i = np.where(~valid)[0]
            for di in drop_i:
                if di > li:
                    ax.plot([x[li], x[di]], [y_C[li], y_top], color=C3['optC'], lw=1.5, ls='--', zorder=4)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(X_LABELS)
    ax.set_ylabel('True passage rank (lower = better)')
    ax.set_title(Q_LABELS[q_id], fontsize=12)
    ax.invert_yaxis()
    ax.set_ylim(bottom=ax.get_ylim()[0] * 1.15)
    ax.grid(True, axis='y', ls='--', alpha=0.3, lw=0.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

fig.legend(handles=[
    mlines.Line2D([], [], color=C3['dig'],  marker='o', ms=7, lw=2, label='Digital'),
    mlines.Line2D([], [], color=C3['optA'], marker='o', ms=7, lw=2, label='Vth Comp'),
    mlines.Line2D([], [], color=C3['optC'], marker='o', ms=7, lw=2, label='No Comp'),
    mlines.Line2D([], [], color=C3['optC'], marker='x', ms=10, mew=2.5, ls='None', label='No Comp (candidate dropped)'),
], fontsize=10, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06), frameon=False)

fig.suptitle('True passage rank vs. competitor density (top2 -> top5 -> top10)', fontsize=14, fontweight='bold', y=1.05)
fig.tight_layout()
out_path = f'{BASE}/newq_margin2_rank_trend.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"저장: {out_path}")
print("완료!")
