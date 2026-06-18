import pandas as pd, time

DATA = r'C:\Users\nmdl-khb\ColBERT\data\msmarco'
BASE = r'C:\Users\nmdl-khb\ColBERT\centroidset'

print('collection pid 수집 중...')
t0 = time.time()
collection_pids = set()
pid2idx = {}
with open(r'D:\msmarco\collection_1m_fair.tsv', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        pid = int(line.split('\t')[0])
        collection_pids.add(pid)
        pid2idx[pid] = idx

print(f'collection pid 수: {len(collection_pids):,}  ({time.time()-t0:.1f}s)')

qrels = pd.read_csv(f'{DATA}/qrels.dev.small.tsv', sep='\t', header=None, names=['qid','0','pid','rel'])
valid = qrels[qrels['pid'].isin(collection_pids)]
n_valid = valid['qid'].nunique()
print(f'전체 dev query: {qrels["qid"].nunique():,}')
print(f'정답 passage가 collection에 있는 query: {n_valid:,}')

# pid2idx 저장
import json
with open(f'{BASE}/collection_pid2idx.json', 'w') as f:
    json.dump(pid2idx, f)
print('collection_pid2idx.json 저장 완료')
