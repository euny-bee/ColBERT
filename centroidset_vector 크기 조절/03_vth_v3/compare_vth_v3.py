"""
[vth_v3] 비교 시각화 — Vth [0,3]V 비율유지 (std=0.90)
  compare_ranking_all3: Digital / Option A / Option C 3-way rank 비교
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

BASE           = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\03_vth_v3'
QUERY_RELEVANT = {0: 7264308, 1: 7264266, 2: 7264253}
Q_LABELS = [
    'q0: "where does real insulin come from"',
    'q1: "where does name nora come from"',
    'q2: "where does most of the iron ore come from"',
]
TOP_K = 30

# =============================================================================
# 데이터 로드
# =============================================================================
print("데이터 로드 중...")
dig_sheets  = pd.read_excel(f'{BASE}/[vth_v3]step3_digital_candidate_pids.xlsx', sheet_name=None)
optA_sheets = pd.read_excel(f'{BASE}/[vth_v3]step3_optA_candidate_pids.xlsx',   sheet_name=None)
optC_sheets = pd.read_excel(f'{BASE}/[vth_v3]step3_optC_candidate_pids.xlsx',   sheet_name=None)
step6       = pd.read_excel(f'{BASE}/[vth_v3]step6_all_results.xlsx',            sheet_name=None)

dig_rank, optA_rank, optC_rank = {}, {}, {}
for q_id in [0, 1, 2]:
    dig_rank[q_id]  = step6[f'q{q_id}_digital'].set_index('pid')
    optA_rank[q_id] = step6[f'q{q_id}_optA'].set_index('pid')
    optC_rank[q_id] = step6[f'q{q_id}_optC'].set_index('pid')

dig_pids_dict, optA_pids_dict, optC_pids_dict = {}, {}, {}
for q_id in [0, 1, 2]:
    dig_pids_dict[q_id]  = set(dig_sheets[f'q{q_id}']['pid'])
    optA_pids_dict[q_id] = set(optA_sheets[f'q{q_id}']['pid'])
    optC_pids_dict[q_id] = set(optC_sheets[f'q{q_id}']['pid'])

COLORS = {
    'all3':      '#4CAF50',
    'dig_A':     '#90CAF9',
    'dig_C':     '#FFCC80',
    'dig_only':  '#78909C',
    'A_only':    '#2196F3',
    'C_only':    '#FF9800',
    'AC_only':   '#9C27B0',
    'true':      '#E91E63',
    'common':    '#4CAF50',
    'optA_only': '#2196F3',
    'optC_only': '#FF9800',
}

C3 = {
    'optA':   '#2196F3',
    'optC':   '#FF9800',
    'true':   '#E91E63',
    'dig':    '#4CAF50',
    'line_A': '#90CAF9',
    'line_C': '#FFCC80',
}

# =============================================================================
# 1. compare_candidates_all
# =============================================================================
print("\n[1/4] compare_candidates_all 그리는 중...")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle(
    "[vth_v3]  Digital vs OptionA & OptionC  —  Candidate Set Comparison  (nprobe=2)\n"
    "OptionA: 개별 Vth 보상  |  OptionC: M0/M3 독립 Vth ~ TruncGauss[0,3]V std=0.90, seed=42",
    fontsize=11, fontweight='bold', y=0.999
)

W = 0.28

def draw_stacked(ax, x, segments, total):
    b = 0
    for val, col in segments:
        if val > 0:
            ax.bar(x, val, bottom=b, color=col, width=W)
            txt_col = 'black' if col in (COLORS['dig_A'], COLORS['dig_C']) else 'white'
            ax.text(x, b + val/2, str(val), ha='center', va='center',
                    fontsize=8, color=txt_col, fontweight='bold')
            b += val
    ax.text(x, total + 1.5, str(total), ha='center', fontsize=10, fontweight='bold')

def draw_scatter(ax, dig_pids, other_pids, dig_df, other_df, label_other, q_id, other_color):
    true_pid    = QUERY_RELEVANT[q_id]
    inter       = dig_pids & other_pids
    dig_only    = dig_pids - other_pids
    other_only  = other_pids - dig_pids

    if inter:
        cp = list(inter)
        ax.scatter(dig_df.loc[cp, 'n_tokens'].values,
                   other_df.loc[cp, 'n_tokens'].values,
                   alpha=0.55, s=30, color=COLORS['all3'], label=f'Common ({len(cp)})')
    if dig_only:
        dp = list(dig_only)
        ax.scatter(dig_df.loc[dp, 'n_tokens'].values, np.zeros(len(dp)),
                   alpha=0.5, s=25, color=COLORS['dig_only'], marker='^',
                   label=f'Digital only ({len(dp)})')
    if other_only:
        op = list(other_only)
        ax.scatter(np.zeros(len(op)), other_df.loc[op, 'n_tokens'].values,
                   alpha=0.5, s=25, color=other_color, marker='s',
                   label=f'{label_other} only ({len(op)})')

    if true_pid in inter:
        tx = dig_df.loc[true_pid, 'n_tokens']
        ty = other_df.loc[true_pid, 'n_tokens']
        ax.scatter(tx, ty, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid}')
        ax.annotate(f'True\n({tx},{ty})', (tx, ty),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7.5, color=COLORS['true'], fontweight='bold')
    elif true_pid in dig_pids:
        tx = dig_df.loc[true_pid, 'n_tokens']
        ax.scatter(tx, 0, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid} (dig only)')
    elif true_pid in other_pids:
        ty = other_df.loc[true_pid, 'n_tokens']
        ax.scatter(0, ty, color=COLORS['true'], s=180, marker='*', zorder=5,
                   label=f'True pid {true_pid} ({label_other} only)')

    mx = max(ax.get_xlim()[1], ax.get_ylim()[1], 33)
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3)
    ax.set_xlabel('Digital n_tokens', fontsize=8)
    ax.set_ylabel(f'{label_other} n_tokens', fontsize=8)
    ax.set_title(f'q{q_id}: Digital vs {label_other} n_tokens', fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(ls='--', alpha=0.25)

for q_id in [0, 1, 2]:
    ax       = axes[0, q_id]
    dig_pids = dig_pids_dict[q_id]
    A_pids   = optA_pids_dict[q_id]
    C_pids   = optC_pids_dict[q_id]

    s_all3     = dig_pids & A_pids & C_pids
    s_dig_A    = (dig_pids & A_pids) - C_pids
    s_dig_C    = (dig_pids & C_pids) - A_pids
    s_dig_only = dig_pids - A_pids - C_pids
    s_A_only   = A_pids - dig_pids - C_pids
    s_C_only   = C_pids - dig_pids - A_pids
    s_AC_only  = (A_pids & C_pids) - dig_pids

    jac_A = len(dig_pids & A_pids) / len(dig_pids | A_pids)
    jac_C = len(dig_pids & C_pids) / len(dig_pids | C_pids)

    draw_stacked(ax, 0, [(len(s_dig_only),COLORS['dig_only']),(len(s_dig_A),COLORS['dig_A']),
                          (len(s_dig_C),COLORS['dig_C']),(len(s_all3),COLORS['all3'])], len(dig_pids))
    draw_stacked(ax, 1, [(len(s_A_only),COLORS['A_only']),(len(s_AC_only),COLORS['AC_only']),
                          (len(s_dig_A),COLORS['dig_A']),(len(s_all3),COLORS['all3'])], len(A_pids))
    draw_stacked(ax, 2, [(len(s_C_only),COLORS['C_only']),(len(s_AC_only),COLORS['AC_only']),
                          (len(s_dig_C),COLORS['dig_C']),(len(s_all3),COLORS['all3'])], len(C_pids))

    ax.text(0.5, -0.07,
            f'Jaccard(Dig↔A)={jac_A:.3f}   Jaccard(Dig↔C)={jac_C:.3f}',
            ha='center', transform=ax.transAxes, fontsize=7.5, color='dimgray', style='italic')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Digital', 'OptionA', 'OptionC'], fontsize=9, fontweight='bold')
    ax.set_ylabel('Candidate passages', fontsize=8)
    ax.set_title(Q_LABELS[q_id], fontsize=8.5, fontweight='bold')
    ax.set_ylim(0, max(len(dig_pids), len(A_pids), len(C_pids)) * 1.22)
    ax.grid(axis='y', ls='--', alpha=0.3)
    if q_id == 0:
        ax.legend(handles=[
            mpatches.Patch(color=COLORS['all3'],     label='dig ∩ A ∩ C'),
            mpatches.Patch(color=COLORS['dig_A'],    label='dig ∩ A \\ C'),
            mpatches.Patch(color=COLORS['dig_C'],    label='dig ∩ C \\ A'),
            mpatches.Patch(color=COLORS['dig_only'], label='dig only'),
            mpatches.Patch(color=COLORS['A_only'],   label='A only'),
            mpatches.Patch(color=COLORS['C_only'],   label='C only'),
            mpatches.Patch(color=COLORS['AC_only'],  label='A ∩ C \\ dig'),
        ], fontsize=7, loc='upper right')

    dig_df  = dig_sheets[f'q{q_id}'].set_index('pid')
    optA_df = optA_sheets[f'q{q_id}'].set_index('pid')
    optC_df = optC_sheets[f'q{q_id}'].set_index('pid')

    draw_scatter(axes[1, q_id], dig_pids, A_pids, dig_df, optA_df, 'OptionA', q_id, COLORS['A_only'])
    draw_scatter(axes[2, q_id], dig_pids, C_pids, dig_df, optC_df, 'OptionC', q_id, COLORS['C_only'])

plt.tight_layout()
out = f'{BASE}/[vth_v3]compare_candidates_all.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"  저장: {out}")
plt.close()

# =============================================================================
# 공통 헬퍼: rank scatter + bump chart
# =============================================================================
def plot_rank_comparison(df_x_dict, df_y_dict, x_rank_col, y_rank_col,
                         x_label, y_label, title, out_path, color_only):
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.99)

    for q_id in [0, 1, 2]:
        true_pid = QUERY_RELEVANT[q_id]
        df_x     = df_x_dict[q_id]
        df_y     = df_y_dict[q_id]

        ax = axes[0, q_id]
        common = [p for p in df_x.index if p in df_y.index]
        xr = [int(df_x.loc[p, x_rank_col]) for p in common]
        yr = [int(df_y.loc[p, y_rank_col]) for p in common]
        cols = [COLORS['true'] if p == true_pid else COLORS['common'] for p in common]
        szs  = [220 if p == true_pid else 25 for p in common]
        ax.scatter(xr, yr, c=cols, s=szs, alpha=0.6, zorder=3)

        if true_pid in df_x.index and true_pid in df_y.index:
            tx = int(df_x.loc[true_pid, x_rank_col])
            ty = int(df_y.loc[true_pid, y_rank_col])
            ax.scatter(tx, ty, c=COLORS['true'], s=250, marker='*', zorder=5)
            ax.annotate(f'True\n({x_label}:{tx}, {y_label}:{ty})', (tx, ty),
                        textcoords='offset points', xytext=(8, 4),
                        fontsize=7.5, color=COLORS['true'], fontweight='bold')

        mx = max(max(xr), max(yr)) + 5 if xr else 50
        ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3)
        ax.set_xlabel(f'{x_label} rank', fontsize=9)
        ax.set_ylabel(f'{y_label} rank', fontsize=9)
        ax.set_title(f'{Q_LABELS[q_id]}\ncommon: {len(common)}', fontsize=8.5, fontweight='bold')
        ax.grid(ls='--', alpha=0.25)
        ax.legend(handles=[
            mlines.Line2D([], [], color=COLORS['common'], marker='o', ls='None', ms=6, label='공통 후보'),
            mlines.Line2D([], [], color=COLORS['true'],   marker='*', ls='None', ms=10,
                          label=f'True pid ({true_pid})'),
        ], fontsize=8, loc='upper left')

        ax = axes[1, q_id]
        x_topk = df_x.reset_index().nsmallest(TOP_K, x_rank_col)['pid'].tolist()
        y_topk = df_y.reset_index().nsmallest(TOP_K, y_rank_col)['pid'].tolist()
        all_p  = list(dict.fromkeys(x_topk + y_topk))

        xrd = dict(zip(df_x.reset_index()['pid'], df_x[x_rank_col]))
        yrd = dict(zip(df_y.reset_index()['pid'], df_y[y_rank_col]))

        for pid in all_p:
            dr  = xrd.get(pid)
            yr_ = yrd.get(pid)
            is_true = pid == true_pid
            in_both = (dr is not None and dr <= TOP_K) and (yr_ is not None and yr_ <= TOP_K)
            x_only  = (dr is not None and dr <= TOP_K) and (yr_ is None or yr_ > TOP_K)
            col  = COLORS['true'] if is_true else (
                   COLORS['common'] if in_both else
                   COLORS['dig_only'] if x_only else color_only)
            lw   = 2.5 if is_true else 0.8
            alph = 1.0 if is_true else 0.45

            pts = []
            if dr  is not None and dr  <= TOP_K: pts.append((0, dr))
            if yr_ is not None and yr_ <= TOP_K: pts.append((1, yr_))
            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                        color=col, lw=lw, alpha=alph, zorder=2)
            for xp, yp in pts:
                ax.scatter(xp, yp, color=col, s=60 if is_true else 20,
                           marker='*' if is_true else 'o', zorder=4, alpha=alph)
            if is_true:
                if dr is not None and dr <= TOP_K:
                    ax.annotate(f'True (rank {dr})', (0, dr),
                                textcoords='offset points', xytext=(-65, 0),
                                fontsize=7.5, color=COLORS['true'], fontweight='bold', ha='left')
                if yr_ is not None and yr_ <= TOP_K:
                    ax.annotate(f'True (rank {yr_})', (1, yr_),
                                textcoords='offset points', xytext=(6, 0),
                                fontsize=7.5, color=COLORS['true'], fontweight='bold')

        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(TOP_K + 1, 0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([x_label, y_label], fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Rank (top-{TOP_K})', fontsize=9)
        ax.set_title(f'{Q_LABELS[q_id]}\nTop-{TOP_K} rank comparison', fontsize=8.5, fontweight='bold')
        ax.grid(axis='y', ls='--', alpha=0.2)
        ax.legend(handles=[
            mlines.Line2D([], [], color=COLORS['common'],  lw=2, label='In both top-k'),
            mlines.Line2D([], [], color=COLORS['dig_only'],lw=2, label=f'{x_label} only'),
            mlines.Line2D([], [], color=color_only,        lw=2, label=f'{y_label} only'),
            mlines.Line2D([], [], color=COLORS['true'],    lw=2.5, label='True document'),
        ], fontsize=7.5, loc='lower right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  저장: {out_path}")
    plt.close()

# =============================================================================
# 2. compare_ranking_L2_optA
# =============================================================================
print("\n[2/4] compare_ranking_L2_optA 그리는 중...")
plot_rank_comparison(
    dig_rank, optA_rank,
    x_rank_col='rank_dig_f32', y_rank_col='rank_optA',
    x_label='Digital L2', y_label='Option A',
    title=("[vth_v3]  Digital L2  vs  Option A  —  Ranking Comparison\n"
           "OptionA: M0/M3 개별 Vth 보상 → Vth 완전 상쇄"),
    out_path=f'{BASE}/[vth_v3]compare_ranking_L2_optA.png',
    color_only=COLORS['optA_only'],
)

# =============================================================================
# 3. compare_ranking_L2_optC
# =============================================================================
print("\n[3/4] compare_ranking_L2_optC 그리는 중...")
plot_rank_comparison(
    dig_rank, optC_rank,
    x_rank_col='rank_dig_f32', y_rank_col='rank_optC',
    x_label='Digital L2', y_label='Option C',
    title=("[vth_v3]  Digital L2  vs  Option C  —  Ranking Comparison\n"
           "OptionC: M0/M3 독립 Vth ~ TruncGauss[0,3]V std=0.90, seed=42"),
    out_path=f'{BASE}/[vth_v3]compare_ranking_L2_optC.png',
    color_only=COLORS['optC_only'],
)

# =============================================================================
# 4. compare_ranking_all3
# =============================================================================
print("\n[4/4] compare_ranking_all3 그리는 중...")

fig, axes = plt.subplots(2, 3, figsize=(17, 12))
fig.suptitle(
    "[vth_v3]  Digital L2  vs  Option A  vs  Option C  —  3-way Ranking Comparison\n"
    "OptionA: 개별 Vth 보상  |  OptionC: M0/M3 독립 Vth ~ TruncGauss[0,3]V std=0.90, seed=42",
    fontsize=11, fontweight='bold', y=0.99
)

for q_id in [0, 1, 2]:
    ax       = axes[0, q_id]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_rank[q_id]
    df_A     = optA_rank[q_id]
    df_C     = optC_rank[q_id]

    common_A = [p for p in df_dig.index if p in df_A.index]
    xA = [int(df_dig.loc[p, 'rank_dig_f32']) for p in common_A]
    yA = [int(df_A.loc[p, 'rank_optA'])      for p in common_A]
    cA = [C3['true'] if p == true_pid else C3['optA'] for p in common_A]
    sA = [200 if p == true_pid else 20 for p in common_A]
    ax.scatter(xA, yA, c=cA, s=sA, alpha=0.65, zorder=3, label='Option A')

    common_C = [p for p in df_dig.index if p in df_C.index]
    xC = [int(df_dig.loc[p, 'rank_dig_f32']) for p in common_C]
    yC = [int(df_C.loc[p, 'rank_optC'])      for p in common_C]
    cC = [C3['true'] if p == true_pid else C3['optC'] for p in common_C]
    sC = [200 if p == true_pid else 20 for p in common_C]
    ax.scatter(xC, yC, c=cC, s=sC, alpha=0.65, zorder=3, marker='D', label='Option C')

    for df_m, rank_col, lbl, off in [(df_A, 'rank_optA', 'A', (+8,+4)),
                                      (df_C, 'rank_optC', 'C', (+8,-12))]:
        if true_pid in df_m.index and true_pid in df_dig.index:
            tx = int(df_dig.loc[true_pid, 'rank_dig_f32'])
            ty = int(df_m.loc[true_pid, rank_col])
            ax.scatter(tx, ty, c=C3['true'], s=260, marker='*', zorder=6)
            ax.annotate(f'True ({lbl}): dig={tx}, rank={ty}', (tx, ty),
                        textcoords='offset points', xytext=off,
                        fontsize=7, color=C3['true'], fontweight='bold')

    mx = max(max(xA+xC, default=50), max(yA+yC, default=50)) + 5
    ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.3)
    ax.set_xlabel('Digital L2 rank', fontsize=9)
    ax.set_ylabel('Method rank', fontsize=9)
    ax.set_title(f'{Q_LABELS[q_id]}\ncommon w/ A: {len(common_A)}  |  w/ C: {len(common_C)}',
                 fontsize=8, fontweight='bold')
    ax.grid(ls='--', alpha=0.25)
    ax.legend(handles=[
        mlines.Line2D([], [], color=C3['optA'], marker='o',  ms=6, ls='None', label='Option A rank'),
        mlines.Line2D([], [], color=C3['optC'], marker='D',  ms=5, ls='None', label='Option C rank'),
        mlines.Line2D([], [], color='black', ls='--', lw=0.8, alpha=0.5, label='y = x'),
        mlines.Line2D([], [], color=C3['true'], marker='*',  ms=10, ls='None', label=f'True ({true_pid})'),
    ], fontsize=7.5, loc='upper left')

X_POS = {'dig': 0, 'optA': 1, 'optC': 2}

for q_id in [0, 1, 2]:
    ax       = axes[1, q_id]
    true_pid = QUERY_RELEVANT[q_id]
    df_dig   = dig_rank[q_id].reset_index()
    df_A     = optA_rank[q_id].reset_index()
    df_C     = optC_rank[q_id].reset_index()

    dig_topk  = df_dig.nsmallest(TOP_K, 'rank_dig_f32')['pid'].tolist()
    optA_topk = df_A.nsmallest(TOP_K,   'rank_optA')['pid'].tolist()
    optC_topk = df_C.nsmallest(TOP_K,   'rank_optC')['pid'].tolist()
    all_pids  = list(dict.fromkeys(dig_topk + optA_topk + optC_topk))

    dig_rd  = dict(zip(df_dig['pid'], df_dig['rank_dig_f32']))
    optA_rd = dict(zip(df_A['pid'],   df_A['rank_optA']))
    optC_rd = dict(zip(df_C['pid'],   df_C['rank_optC']))

    for pid in all_pids:
        dr = dig_rd.get(pid); ar = optA_rd.get(pid); cr = optC_rd.get(pid)
        in_dig = dr is not None and dr <= TOP_K
        in_A   = ar is not None and ar <= TOP_K
        in_C   = cr is not None and cr <= TOP_K
        is_true = pid == true_pid
        n_m     = sum([in_dig, in_A, in_C])

        col  = C3['true'] if is_true else (
               C3['dig']  if n_m >= 2 else
               C3['optA'] if in_A and not in_dig and not in_C else
               C3['optC'] if in_C and not in_dig and not in_A else C3['dig'])
        lw   = 2.5 if is_true else 0.7
        alph = 1.0 if is_true else 0.4

        pts = []
        if in_dig: pts.append((X_POS['dig'],  dr))
        if in_A:   pts.append((X_POS['optA'], ar))
        if in_C:   pts.append((X_POS['optC'], cr))

        for k in range(len(pts) - 1):
            ax.plot([pts[k][0], pts[k+1][0]], [pts[k][1], pts[k+1][1]],
                    color=col, lw=lw, alpha=alph, zorder=2)
        for xp, yp in pts:
            ax.scatter(xp, yp, color=col, s=80 if is_true else 18,
                       marker='*' if is_true else 'o', zorder=4, alpha=max(alph, 0.6))

        if is_true:
            for xp, yp, lbl in ([(X_POS['dig'],  dr,  f'L2:{dr}')]  if in_dig else []) + \
                                ([(X_POS['optA'], ar,  f'A:{ar}')]   if in_A   else []) + \
                                ([(X_POS['optC'], cr,  f'C:{cr}')]   if in_C   else []):
                off = (-45, 4) if xp == 0 else (6, 4)
                ax.annotate(f'True\n({lbl})', (xp, yp),
                            textcoords='offset points', xytext=off,
                            fontsize=7, color=C3['true'], fontweight='bold', ha='left')

    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(TOP_K + 1, 0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Digital L2', 'Option A', 'Option C'], fontsize=9, fontweight='bold')
    ax.set_ylabel(f'Rank (top-{TOP_K})', fontsize=9)
    ax.set_title(f'{Q_LABELS[q_id]}\nTop-{TOP_K} rank comparison', fontsize=8.5, fontweight='bold')
    ax.grid(axis='y', ls='--', alpha=0.2)
    ax.legend(handles=[
        mlines.Line2D([], [], color=C3['dig'],  lw=2,   label='In 2+ methods'),
        mlines.Line2D([], [], color=C3['optA'], lw=1.5, label='Option A only'),
        mlines.Line2D([], [], color=C3['optC'], lw=1.5, label='Option C only'),
        mlines.Line2D([], [], color=C3['true'], lw=2.5, label='True document'),
    ], fontsize=7.5, loc='lower right')

plt.tight_layout()
out = f'{BASE}/[vth_v3]compare_ranking_all3.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"  저장: {out}")
plt.close()

print("\n완료! 생성된 파일:")
for name in ['[vth_v3]compare_candidates_all.png',
             '[vth_v3]compare_ranking_L2_optA.png',
             '[vth_v3]compare_ranking_L2_optC.png',
             '[vth_v3]compare_ranking_all3.png']:
    print(f"  {BASE}/{name}")
