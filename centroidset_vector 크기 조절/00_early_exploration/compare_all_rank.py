"""
[original] / [clip] / [clip99] / [clip99.9] top-5 rank_order 비교
"""
import pandas as pd
import numpy as np

BASE      = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\01_clip99.9_baseline'
ORIG_BASE = f'{BASE}\clip99.9외'

centroid_cols = [f'centroid_{i}' for i in range(100)]

def load_order(path, sheet=None):
    """scores 시트 또는 단일 시트에서 top-k order 반환 (96, 100)"""
    try:
        df = pd.read_excel(path, sheet_name='rank_order', index_col=0)
        rank_cols = [f'rank_{i+1}' for i in range(100)]
        return df['query_id'].values, df.index.tolist(), df[rank_cols].values
    except Exception:
        # 단일 시트(scores만 있는 경우)
        df = pd.read_excel(path, index_col=0)
        scores = df[centroid_cols].values
        order  = np.argsort(-scores, axis=1)
        return df['query_id'].values, df.index.tolist(), order

versions = {
    'original' : f'{ORIG_BASE}/[original]query_centroid_ranking.xlsx',
    'clip'     : f'{ORIG_BASE}/[clip]query_centroid_ranking.xlsx',
    'clip99'   : f'{ORIG_BASE}/[clip99]query_centroid_ranking.xlsx',
    'clip99.9' : f'{BASE}/[clip99.9]query_centroid_ranking.xlsx',
}

data = {}
for name, path in versions.items():
    qids, tokens, order = load_order(path)
    data[name] = {'qids': qids, 'tokens': tokens, 'order': order}

query_ids = data['original']['qids']
tokens    = data['original']['tokens']
TOP = 5

# ==========================================================================
# 요약 테이블 (query별 × version별 일치율)
# ==========================================================================
print(f"{'='*70}")
print(f"top-{TOP} rank_order 비교  (vs original)")
print(f"{'='*70}")

ver_list = ['clip', 'clip99', 'clip99.9']
header = f"{'query':<8} {'n':>4}  " + "  ".join(f"{'['+v+']':^18}" for v in ver_list)
print(f"\n{header}")
print("-" * 70)

summary = {v: [] for v in ver_list}

for q_id in sorted(set(query_ids)):
    mask   = query_ids == q_id
    o_top  = data['original']['order'][mask, :TOP]

    row = f"query {q_id}  {mask.sum():>3}  "
    for v in ver_list:
        v_top   = data[v]['order'][mask, :TOP]
        matches = [len(set(o_top[i]) & set(v_top[i])) for i in range(mask.sum())]
        full    = sum(1 for m in matches if m == TOP)
        avg     = np.mean(matches)
        summary[v].extend(matches)
        row += f"  {full}/{mask.sum()} tok ({avg:.2f}/5)  "
    print(row)

# 전체 합계
print("-" * 70)
row = f"{'전체':<8} {'96':>4}  "
for v in ver_list:
    full = sum(1 for m in summary[v] if m == TOP)
    avg  = np.mean(summary[v])
    row += f"  {full}/96 tok ({avg:.2f}/5)    "
print(row)

# ==========================================================================
# 토큰별 상세 비교
# ==========================================================================
print(f"\n\n{'='*70}")
print("토큰별 상세 (top-5 centroid IDs)")
print(f"{'='*70}")

for q_id in sorted(set(query_ids)):
    mask  = query_ids == q_id
    idxs  = np.where(mask)[0]
    toks  = [tokens[i] for i in idxs]

    print(f"\n[Query {q_id}]")
    print(f"{'token':<10} {'original':^25} {'[clip]':^25} {'[clip99]':^25} {'[clip99.9]':^25}")
    print("-" * 105)

    for i, (idx, tok) in enumerate(zip(idxs, toks)):
        o5  = data['original']['order'][idx, :TOP].tolist()
        c5  = data['clip']['order'][idx, :TOP].tolist()
        c99 = data['clip99']['order'][idx, :TOP].tolist()
        c999= data['clip99.9']['order'][idx, :TOP].tolist()

        # original과 다른 경우 표시
        diff_c   = "" if set(o5)==set(c5)   else "*"
        diff_99  = "" if set(o5)==set(c99)  else "*"
        diff_999 = "" if set(o5)==set(c999) else "*"

        print(f"{tok:<10} {str(o5):<25} {str(c5)+diff_c:<25} {str(c99)+diff_99:<25} {str(c999)+diff_999:<25}")
