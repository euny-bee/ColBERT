"""
True document rank table — booktabs style (논문 형식)
rows: Digital / Vth comp / No comp
cols: q0, q1, q2  (rank / candidates)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE      = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\03_vth_v3'
FONT_BASE = 12

def _setup_matplotlib():
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  sans,
        "font.size":        FONT_BASE,
        "axes.unicode_minus": False,
    })

_setup_matplotlib()

# ── 데이터: rank / total_candidates ──────────────────────────────────────────
#          q0        q1       q2
raw = {
    'Digital':  [('1', '127'),  ('1', '70'),   ('1', '108')],
    'Vth comp': [('1', '127'),  ('1', '84'),   ('1', '85')],
    'No comp':  [('92', '197'), ('19', '195'), ('21', '195')],
}
methods = list(raw.keys())
queries = ['q0', 'q1', 'q2']

# 전치: rows=queries, cols=methods
cell_text = []
for qi, q in enumerate(queries):
    row = []
    for m in methods:
        rank, total = raw[m][qi]
        row.append(f'{rank}  ({total})')
    cell_text.append(row)

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 2.0))
ax.axis('off')

tbl = ax.table(
    cellText=cell_text,
    rowLabels=queries,
    colLabels=methods,
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(FONT_BASE)
tbl.scale(1, 1.8)

n_rows = len(queries)
n_cols = len(methods)

# ── Booktabs 스타일: 테두리/색 제거, 가로선만 ────────────────────────────────
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('none')
    cell.set_facecolor('none')
    cell.set_linewidth(0)

    if r == 0:  # column header
        cell.set_text_props(fontweight='bold', ha='center')
    if c == -1:  # row label
        cell.set_text_props(fontweight='bold', ha='right')

# 가로선 그리기: top rule, midrule, bottom rule
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

def _draw_hline(ax, tbl, row_above, lw, color='black'):
    """row_above 아래쪽에 가로선 그리기."""
    cells_in_row = [(r, c) for (r, c) in tbl.get_celld() if r == row_above and c >= 0]
    if not cells_in_row:
        return
    xs = []
    ys = []
    for (r, c) in cells_in_row:
        bb = tbl[r, c].get_window_extent(renderer)
        bb_ax = bb.transformed(ax.transData.inverted())
        xs.append(bb_ax.x0)
        xs.append(bb_ax.x1)
        ys.append(bb_ax.y0)
    x0, x1 = min(xs), max(xs)
    y = min(ys)
    ax.axhline(y=y, xmin=0, xmax=1, color=color, lw=lw,
               transform=ax.transData, clip_on=False,
               solid_capstyle='butt')

# 행 라벨 컬럼 포함한 전체 x 범위
all_cells = [(r, c) for (r, c) in tbl.get_celld() if r == 0]
xs_all = []
for (r, c) in tbl.get_celld():
    if r == 0:
        bb = tbl[r, c].get_window_extent(renderer)
        bb_ax = bb.transformed(ax.transData.inverted())
        xs_all += [bb_ax.x0, bb_ax.x1]

# 좌우 끝 좌표
x_left  = min(xs_all)
x_right = max(xs_all)

def _hline_full(ax, tbl, row, lw):
    cells = [(r, c) for (r, c) in tbl.get_celld() if r == row]
    ys = []
    xs = []
    for (r, c) in cells:
        bb = tbl[r, c].get_window_extent(renderer)
        bb_ax = bb.transformed(ax.transData.inverted())
        ys.append(bb_ax.y0)
        xs += [bb_ax.x0, bb_ax.x1]
    y = min(ys)
    x0, x1 = min(xs), max(xs)
    line = plt.Line2D([x0, x1], [y, y], transform=ax.transData,
                      color='black', lw=lw, clip_on=False)
    ax.add_line(line)

def _hline_top(ax, tbl, row, lw):
    cells = [(r, c) for (r, c) in tbl.get_celld() if r == row]
    ys = []
    xs = []
    for (r, c) in cells:
        bb = tbl[r, c].get_window_extent(renderer)
        bb_ax = bb.transformed(ax.transData.inverted())
        ys.append(bb_ax.y1)
        xs += [bb_ax.x0, bb_ax.x1]
    y = max(ys)
    x0, x1 = min(xs), max(xs)
    line = plt.Line2D([x0, x1], [y, y], transform=ax.transData,
                      color='black', lw=lw, clip_on=False)
    ax.add_line(line)

# top rule (헤더 위)
_hline_top(ax, tbl, 0, lw=1.5)
# mid rule (헤더 아래)
_hline_full(ax, tbl, 0, lw=1.0)
# bottom rule (마지막 행 아래)
_hline_full(ax, tbl, n_rows, lw=1.5)

fig.tight_layout()
out = f'{BASE}/[vth_v3]rank_table.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"저장: {out}")
