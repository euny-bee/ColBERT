# ColBERT Centroid Set 생성 작업 요약

## 목적

ColBERT retrieval의 centroid selection 단계(`C @ Q.T`)를 하드웨어에서 검증하기 위한
소규모 데이터셋 구성. 128차원 centroid set 100개와 query token embedding ~100개를 생성.

---

## 사용 데이터

| 항목 | 경로 | 내용 |
|------|------|------|
| Collection | `D:/msmarco/full/collection.tsv` | MS MARCO 8.8M passages |
| Queries | `C:/Users/nmdl-khb/ColBERT/data/msmarco/queries.dev.tsv` | 101,093개 |
| Qrels | `C:/Users/nmdl-khb/ColBERT/data/msmarco/qrels.dev.small.tsv` | 7,437 pairs |
| 모델 | `colbert-ir/colbertv2.0` | HuggingFace checkpoint |
| 실행 환경 | `C:/Users/nmdl-khb/miniconda3/envs/colbert` | Windows conda |

---

## 선택된 쿼리 및 정답 Passage

| qid | query | relevant pid |
|-----|-------|-------------|
| 1000000 | where does real insulin come from | 7264253 |
| 1000004 | where does name nora come from | 7264266 |
| 1000006 | where does most of the iron ore come from | 7264308 |

---

## 생성 파이프라인

### Step 1 — Passage 수집 (`make_centroidset.py`)
- 정답 passage 3개 (qrels 기준) + 랜덤 197개 (seed=42)
- **총 200개 passage**

### Step 2 — Document 인코딩
- `Checkpoint.docFromText()` (colbert-ir/colbertv2.0)
- 200개 passage → **12,919개 token embedding** (평균 ~64.6 tokens/passage)
- shape: (12919, 128), L2 정규화 완료 (norm ≈ 1.0)

### Step 3 — K-means (k=100)
- FAISS `Kmeans(dim=128, k=100, niter=20, seed=123)`
- 12,919개 token embedding → **100개 centroid** (L2 정규화)
- ColBERT 원본 공식(`2^floor(log2(16√n))`)으로는 1,024개가 나오나, 하드웨어 테스트 목적으로 k=100 직접 지정

### Step 4 — Query 인코딩
- `Checkpoint.queryFromText()` (동일 모델)
- 3개 쿼리 × 32 tokens = **96개 query token embedding**
- shape: (96, 128), L2 정규화 완료

### Step 5 — Centroid Selection 검증

정답 passage token 165개의 nearest centroid가 query top-n 선택에 포함되는 비율:

| top-n | hit rate |
|-------|---------|
| top-5 | 80.0% |
| top-10 | 80.0% |
| top-20 | 88.5% |
| top-30 | 89.7% |
| top-50 | 98.2% |

### Step 6 — IVF 구축 (`make_ivf.py`)

각 centroid에 어떤 token/passage가 배정됐는지 역방향 인덱스:
- centroid당 평균 **129.2개 token**, **17.7개 passage**
- 정답 passage별 배정된 centroid:
  - pid 7264253 (insulin) → centroid [29, 33, 40, 43, 59, 72, 73, 81]
  - pid 7264266 (nora) → centroid [16, 29, 30, 43, 59, 89, 93]
  - pid 7264308 (iron ore) → centroid [38, 43, 47, 61, 63, 93]

### Step 7 — 잔차 계산 (`make_residuals.py`)

각 token embedding에서 nearest centroid를 뺀 잔차:

**Float32 잔차:**
- mean abs residual: 0.0633
- max abs residual: 0.4680

**2-bit 양자화 (ColBERT 원본 방식):**
- bucket_cutoffs: [-0.0499, 0.0013, 0.0524]
- bucket_weights: [-0.0891, -0.0224, +0.0249, +0.0919]
- 양자화 오차 평균: 0.0229

### Step 8 — Query-Centroid Dot Product & Ranking (`make_query_centroid_ranking.py`)

Q (96×128) @ C.T (128×100) = scores (96×100):
- score 범위: [-0.40, 0.66]
- 각 query token마다 centroid 0~99번을 score 내림차순으로 rank 1~100 부여

> L2 정규화된 벡터끼리의 dot product = cosine similarity이므로,
> dot product 순위 = Euclidean distance 순위 (d = sqrt(2 - 2·score))

---

## 생성된 파일 목록

### 핵심 데이터

| 파일 | shape | 설명 |
|------|-------|------|
| `centroids_100x128.npy/.csv/.xlsx` | (100, 128) | centroid set, L2 정규화 |
| `query_embs_96x128.npy/.csv/.xlsx` | (96, 128) | query token embeddings, q0_t0~q2_t31 |
| `doc_embs_12919x128.npy/.csv/.xlsx` | (12919, 128) | document token embeddings, is_relevant 열 포함 |

### 인덱스

| 파일 | 설명 |
|------|------|
| `ivf_centroid2pid.xlsx` | centroid → passage 매핑 (구버전) |
| `ivf_centroid2token2pid.xlsx` | centroid → token → passage 매핑 (Sheet1: token_level / Sheet2: passage_level) |
| `ivf_centroid2pid_long.csv` | centroid-token-passage long format CSV |
| `doclens.json` | passage별 token 수 (200개) |

### 잔차

| 파일 | 설명 |
|------|------|
| `residuals_float32.csv/.xlsx` | float32 잔차 = doc_embs - nearest_centroid |
| `residuals_2bit.csv/.xlsx` | 2-bit 양자화 잔차 (버킷 중심값으로 역양자화) |
| `bucket_info.json` | 2-bit 버킷 경계/중심값 |

### 검색 시뮬레이션

| 파일 | 설명 |
|------|------|
| `query_centroid_ranking.xlsx` | Sheet1: scores(96×100) / Sheet2: ranks(96×100) / Sheet3: rank_order(q_t당 rank1=몇번 centroid) |
| `query_centroid_ranking_scores.csv` | dot product 점수 |
| `query_centroid_ranking_ranks.csv` | centroid별 rank |
| `centroid_selection_result.json` | centroid selection 검증 결과 (hit rate) |

### 스크립트

| 파일 | 역할 |
|------|------|
| `make_centroidset.py` | passage 수집 → 인코딩 → k-means → centroid/query embedding 생성 |
| `add_headers.py` | CSV/Excel 행/열 이름 추가 |
| `save_doc_embs.py` | doc_embs npy → CSV/Excel 변환 (is_relevant 열 포함) |
| `make_ivf.py` | centroid → token → passage IVF 구축 |
| `make_residuals.py` | float32 및 2-bit 잔차 계산 |
| `make_query_centroid_ranking.py` | query-centroid dot product & ranking |
