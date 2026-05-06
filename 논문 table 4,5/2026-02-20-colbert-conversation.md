# ColBERT 분석 대화 기록

> 날짜: 2026-02-20

---

## Q1. ColBERT 임베딩 연산량 계산

### 질문
> ColBERT 돌려봐서 embedding 돌리면 연산량이 나오는데, 연산량이 얼마인지 계산해줘.

### 답변

#### 기본 아키텍처 파라미터 (bert-base-uncased 기준)

| 파라미터 | 값 |
|---|---|
| Transformer 레이어 수 (L) | 12 |
| Hidden size (H) | 768 |
| Attention heads | 12 (head dim = 64) |
| FFN intermediate size (I) | 3072 |
| Query 시퀀스 길이 (s_q) | 32 (고정, MASK 패딩) |
| Document 시퀀스 길이 (s_d) | 최대 220 |
| 프로젝션 출력 차원 (d) | 128 |
| 모델 파라미터 수 | ~109.1M |

#### Transformer 레이어 1개당 FLOPs (시퀀스 길이 = s)

**Multi-Head Self-Attention:**

| 연산 | 수식 | FLOPs |
|---|---|---|
| Q, K, V 프로젝션 | 3 × 2sH² | 6sH² |
| Attention score (QK^T) | 2s²H | 2s²H |
| Attention × V | 2s²H | 2s²H |
| Output 프로젝션 | 2sH² | 2sH² |
| **소계** | | **8sH² + 4s²H** |

**Feed-Forward Network (FFN):**

| 연산 | 수식 | FLOPs |
|---|---|---|
| 1차 Linear (H→I) | 2sHI | 2sHI |
| 2차 Linear (I→H) | 2sIH | 2sIH |
| **소계** | | **4sHI** |

**레이어 1개 합계:**

```
FLOPs_layer = 8sH² + 4s²H + 4sHI
            = 8s(589,824) + 4s²(768) + 4s(2,359,296)
            = 14,155,776s + 3,072s²
```

#### 전체 모델 FLOPs

```
FLOPs_total = 12 × (14,155,776s + 3,072s²)     ← BERT 12 레이어
            + 2 × s × 768 × 128                  ← Linear projection (768→128)
            = 169,869,312s + 36,864s² + 196,608s
            = 170,065,920s + 36,864s²
```

#### Query 임베딩 (s = 32)

```
FLOPs = 170,065,920 × 32 + 36,864 × 32²
      = 5,442,109,440 + 37,748,736
      = 5,479,858,176
```

**≈ 5.48 GFLOPs / query**

#### Document 임베딩 (s = 220, 최대 길이)

```
FLOPs = 170,065,920 × 220 + 36,864 × 220²
      = 37,414,502,400 + 1,784,217,600
      = 39,198,720,000
```

**≈ 39.2 GFLOPs / document**

#### MaxSim 스코어링 (검색 시 query-document 유사도)

```
(32, 128) × (128, 220) 행렬곱 → 2 × 32 × 128 × 220 = 1,802,240 FLOPs
+ max 연산 + sum 연산 (무시 가능)
```

**≈ 1.8 MFLOPs / (query, document) 쌍** — 임베딩 대비 무시 가능한 수준

#### 요약 테이블

| 연산 | FLOPs | 비고 |
|---|---|---|
| **Query 1개 인코딩** | **~5.48 GFLOPs** | s=32 고정 (MASK 패딩) |
| **Document 1개 인코딩** | **~39.2 GFLOPs** | s=220 최대 기준 |
| **Document 1개 (평균 s≈128)** | **~22.4 GFLOPs** | 실제 평균 길이 기준 |
| MaxSim 스코어링 | ~1.8 MFLOPs | query-doc 쌍 당 |
| **컬렉션 N개 문서 인코딩** | **~39.2N GFLOPs** | 인덱싱 시 총 비용 |

> 참고: BERT-base의 전체 파라미터 ~109M개 중 ColBERT가 추가하는 것은 프로젝션 레이어의 98,304개(768×128)뿐이므로, 연산량의 99.9%는 BERT 인코더에서 발생합니다. self-attention의 s² 항은 짧은 시퀀스(32, 220)에서는 선형 항 대비 비중이 작습니다 (query 0.7%, document 4.6%).

---

## Q2. 임베딩 벡터 간 유사도/거리 계산 코드 위치

### 질문
> ColBERT Code 중에서 embedding vector 들끼리 dot product, cosine similarity, euclidean distance 계산하는 구문이 있으면 출처 밝혀서 찾아줘

### 답변

#### 1. Dot Product (`@` 연산자)

##### 1a. 핵심 스코어링 — `colbert_score`
`colbert/modeling/colbert.py:175`
```python
scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
```
- `(num_docs, doc_len, dim) @ (num_docs, dim, query_len)` → `(num_docs, doc_len, query_len)`
- Q, D 모두 L2 정규화 후 호출되므로 **cosine similarity와 동치**

##### 1b. Packed 변형 — `colbert_score_packed`
`colbert/modeling/colbert.py:195`
```python
scores = D_packed @ Q.to(dtype=D_packed.dtype).T
```
- `(total_doc_tokens, dim) @ (dim, query_len)` — 메모리 효율적 버전

##### 1c. In-batch negatives 학습 — `compute_ib_loss`
`colbert/modeling/colbert.py:69`
```python
scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)
```
- 배치 내 모든 query × 모든 document 교차 dot product (학습 시 negative sampling)

##### 1d. IVF 셀 선택 — `get_cells`
`colbert/search/candidate_generation.py:13`
```python
scores = (self.codec.centroids @ Q.T)
```
- `(num_centroids, dim) @ (dim, query_len)` — 검색 시 어느 centroid 셀을 탐색할지 결정

##### 1e. 임베딩 개별 스코어링 — `generate_candidate_scores`
`colbert/search/candidate_generation.py:43`
```python
return (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T
```

##### 1f. Centroid 할당 — `compress_into_codes`
`colbert/indexing/codecs/residual.py:215-217`
```python
indices = (self.centroids @ batch.T.cuda().half()).max(dim=0).indices
```
- 인덱싱 시 각 임베딩을 가장 가까운 centroid에 할당

#### 2. Cosine Similarity (L2 정규화 + dot product)

ColBERT는 `F.cosine_similarity()`를 직접 호출하지 않고, **L2 정규화 후 dot product**로 cosine similarity를 구현합니다.

##### 2a. Query 인코더 정규화
`colbert/modeling/colbert.py:93`
```python
return torch.nn.functional.normalize(Q, p=2, dim=2)
```

##### 2b. Document 인코더 정규화
`colbert/modeling/colbert.py:104`
```python
D = torch.nn.functional.normalize(D, p=2, dim=2)
```

##### 2c. 복원된 임베딩 정규화 (GPU)
`colbert/indexing/codecs/residual.py:271-273`
```python
D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()
```

##### 2d. 복원된 임베딩 정규화 (CPU)
`colbert/search/index_storage.py:185`
```python
D_packed = torch.nn.functional.normalize(D_packed.to(torch.float32), p=2, dim=-1)
```

##### 2e. Centroid 정규화
`colbert/indexing/collection_indexer.py:306`
```python
centroids = torch.nn.functional.normalize(centroids, dim=-1)
```

##### 2f. 토큰 간 Self-similarity (풀링용)
`colbert/modeling/checkpoint.py:34`
```python
similarities = torch.mm(passage_embeddings, passage_embeddings.t())
```
- passage 내 토큰 임베딩 간 pairwise cosine similarity 행렬 → 계층적 클러스터링에 사용

#### 3. Euclidean / L2 Distance

##### 3a. L2 기반 MaxSim 스코어링
`colbert/modeling/colbert.py:117-121`
```python
if self.colbert_config.similarity == 'l2':
    return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
```
- **음수 제곱 유클리드 거리** 기반 MaxSim — `config.similarity = 'l2'`로 설정 시 활성화
- `(batch, query_len, 1, dim) - (batch, 1, doc_len, dim)` → 모든 토큰 쌍의 제곱 L2 거리 계산

##### 3b. Ward 클러스터링용 유클리드 거리
`colbert/modeling/checkpoint.py:40`
```python
similarities = 1 - similarities.cpu().numpy()   # cosine → distance
Z = linkage(similarities, metric="euclidean", method="ward")
```

#### 4. MaxSim (ColBERT 핵심 연산)

##### 4a. Python/GPU 구현
`colbert/modeling/colbert.py:132-154`
```python
scores = scores_padded.max(1).values   # 각 query 토큰별 max over doc 토큰
return scores.sum(-1)                   # query 토큰 합산
```

##### 4b. C++ CPU 멀티스레드 구현
`colbert/modeling/segmented_maxsim.cpp:22-93`
- pthread 기반 병렬 MaxSim — CPU 검색 경로에서 사용

##### 4c. C++ 근사 MaxSim (후보 필터링)
`colbert/search/filter_pids.cpp:27-69`
- centroid 스코어 기반 근사 MaxSim으로 후보 문서를 사전 필터링

##### 4d. FLIPR TopK-MaxSim 변형
`colbert/modeling/colbert.py:139-152`
```python
A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)
```
- 전체 합산 대신 Top-K MaxSim 값만 합산하는 변형 (Baleen 멀티홉 검색용)

#### 요약

| 유형 | 파일 | 라인 | 용도 |
|---|---|---|---|
| **Dot product** | `modeling/colbert.py` | 175 | 핵심 query-doc 스코어링 |
| **Dot product** | `modeling/colbert.py` | 195 | packed 변형 |
| **Dot product** | `modeling/colbert.py` | 69 | in-batch negatives 학습 |
| **Dot product** | `search/candidate_generation.py` | 13 | IVF 셀 선택 |
| **Dot product** | `search/candidate_generation.py` | 43 | 개별 임베딩 스코어링 |
| **Dot product** | `indexing/codecs/residual.py` | 215 | centroid 할당 |
| **Cosine sim** | `modeling/colbert.py` | 93, 104 | Q/D L2 정규화 (dot→cosine) |
| **Cosine sim** | `modeling/checkpoint.py` | 34 | 토큰 self-similarity |
| **L2 distance** | `modeling/colbert.py` | 121 | L2 기반 MaxSim |
| **Euclidean** | `modeling/checkpoint.py` | 40 | Ward 클러스터링 |
| **MaxSim** | `modeling/colbert.py` | 132-154 | GPU MaxSim 집계 |
| **MaxSim** | `modeling/segmented_maxsim.cpp` | 22-93 | CPU C++ MaxSim |
| **MaxSim** | `search/filter_pids.cpp` | 27-69 | 근사 MaxSim 필터링 |

> **핵심 패턴**: ColBERT는 `F.cosine_similarity()`를 직접 사용하지 않습니다. 대신 Q와 D를 먼저 `F.normalize(p=2)`로 단위 벡터화한 뒤, `@` 연산자로 dot product를 수행하여 cosine similarity를 얻는 방식입니다. L2 distance는 `config.similarity='l2'` 설정 시에만 사용됩니다.

---

## Q3. ColBERT 전체 프로젝트 구조

### 질문
> ColBERT 전체 프로젝트 구조 한국어로 설명해줘

### 답변

```
ColBERT/
├── setup.py                  # 패키지 설치 설정 (v0.2.22)
├── server.py                 # Flask 기반 검색 API 서버
├── README.md                 # 프로젝트 문서
├── ROADMAP.md                # 개발 로드맵
├── LoTTE.md                  # LoTTE 벤치마크 설명
├── conda_env.yml             # GPU 환경 (CUDA 11.3, PyTorch 1.13.1)
├── conda_env_cpu.yml         # CPU 전용 환경
│
├── colbert/                  ★ 핵심 패키지
│   ├── __init__.py
│   ├── indexer.py            # 인덱싱 진입점 (Indexer 클래스)
│   ├── searcher.py           # 검색 진입점 (Searcher 클래스)
│   ├── trainer.py            # 학습 진입점 (Trainer 클래스)
│   ├── index.py              # 인덱스 메타데이터 관리
│   ├── index_updater.py      # 인덱스 증분 업데이트 (beta)
│   ├── parameters.py         # 전역 파라미터
│   │
│   ├── modeling/             ★ 모델 정의
│   │   ├── colbert.py        # ColBERT 모델 (query/doc 인코딩, 스코어링, MaxSim)
│   │   ├── base_colbert.py   # 기본 ColBERT 클래스 (체크포인트 로딩)
│   │   ├── hf_colbert.py     # HuggingFace 통합 (BERT→Linear 768→128)
│   │   ├── checkpoint.py     # 체크포인트 관리 (인코딩 유틸리티, 풀링)
│   │   ├── segmented_maxsim.cpp  # MaxSim C++ 구현 (CPU 멀티스레드)
│   │   │
│   │   ├── tokenization/     # 토크나이저
│   │   │   ├── query_tokenization.py   # 쿼리 토크나이저 ([CLS][Q]...[MASK] 패딩, 고정 32토큰)
│   │   │   ├── doc_tokenization.py     # 문서 토크나이저 ([CLS][D]..., 최대 220토큰)
│   │   │   └── utils.py                # 토크나이저 유틸리티
│   │   │
│   │   └── reranker/         # 리랭커 모델
│   │       ├── electra.py    # ELECTRA 기반 리랭커
│   │       └── tokenizer.py  # 리랭커용 토크나이저
│   │
│   ├── indexing/             ★ 인덱싱 파이프라인
│   │   ├── collection_indexer.py   # 컬렉션 인덱싱 메인 로직 (k-means, 인코딩)
│   │   ├── collection_encoder.py   # 문서 컬렉션 배치 인코딩
│   │   ├── index_saver.py          # 인덱스 디스크 저장
│   │   ├── index_manager.py        # 인덱스 병합/관리
│   │   ├── loaders.py              # 인덱스 로딩 유틸리티
│   │   ├── utils.py                # 인덱싱 유틸리티
│   │   │
│   │   └── codecs/           # 압축 코덱
│   │       ├── residual.py                     # 잔차 압축 코덱 (centroid + residual 양자화)
│   │       ├── residual_embeddings.py          # 잔차 임베딩 저장/로딩
│   │       ├── residual_embeddings_strided.py  # 스트라이드 접근
│   │       ├── decompress_residuals.cpp        # 잔차 복원 C++ 구현
│   │       └── packbits.cpp                    # 비트 패킹 C++ 구현
│   │
│   ├── search/               ★ 검색 파이프라인
│   │   ├── candidate_generation.py  # 후보 생성 (centroid→셀 선택→후보 문서)
│   │   ├── index_storage.py         # 인덱스 스토리지 (스코어링, 필터링)
│   │   ├── index_loader.py          # 인덱스 로딩
│   │   ├── strided_tensor.py        # 가변 길이 텐서 처리
│   │   ├── strided_tensor_core.py   # StridedTensor 핵심 구현
│   │   ├── filter_pids.cpp          # 근사 MaxSim 후보 필터링 C++ 구현
│   │   ├── decompress_residuals.cpp # 잔차 복원 C++ (검색용)
│   │   └── segmented_lookup.cpp     # 세그먼트 룩업 C++ 구현
│   │
│   ├── training/             ★ 학습 파이프라인
│   │   ├── training.py       # 학습 루프 (DDP 분산 학습 지원)
│   │   ├── eager_batcher.py  # Eager 배처 (메모리에 전체 로딩)
│   │   ├── lazy_batcher.py   # Lazy 배처 (필요 시 로딩)
│   │   ├── rerank_batcher.py # 리랭킹용 배처
│   │   └── utils.py          # 학습 유틸리티
│   │
│   ├── data/                 # 데이터 추상화
│   │   ├── collection.py     # Collection 클래스 (문서 컬렉션 관리)
│   │   ├── queries.py        # Queries 클래스 (쿼리 관리)
│   │   ├── ranking.py        # Ranking 클래스 (랭킹 결과)
│   │   ├── dataset.py        # Dataset 클래스 (학습 데이터)
│   │   └── examples.py       # Examples 클래스 (학습 예제)
│   │
│   ├── infra/                # 인프라/설정
│   │   ├── config/
│   │   │   ├── config.py         # ColBERTConfig (통합 설정 클래스)
│   │   │   ├── settings.py       # 설정 기본값 (dim=128, doc_maxlen=220, query_maxlen=32 등)
│   │   │   ├── core_config.py    # 핵심 설정 (RunSettings)
│   │   │   └── base_config.py    # 설정 기본 클래스
│   │   ├── run.py            # Run 컨텍스트 관리
│   │   ├── launcher.py       # 분산 학습 런처
│   │   ├── provenance.py     # 실험 추적
│   │   └── utilities/        # 데이터 전처리 유틸리티
│   │
│   ├── distillation/         # 지식 증류
│   │   ├── scorer.py         # 증류용 스코어러
│   │   └── ranking_scorer.py # 랭킹 기반 스코어러
│   │
│   ├── evaluation/           # 평가
│   │   ├── metrics.py        # 평가 메트릭 (MRR, Recall 등)
│   │   ├── loaders.py        # 평가 데이터 로더
│   │   └── load_model.py     # 모델 로딩 유틸리티
│   │
│   ├── utils/                # 공통 유틸리티
│   │   ├── amp.py            # Mixed precision 학습
│   │   ├── distributed.py    # 분산 학습 유틸리티
│   │   ├── coalesce.py       # 인덱스 병합
│   │   └── ...
│   │
│   └── tests/                # 테스트
│       ├── e2e_test.py           # E2E 테스트 (SQuAD)
│       ├── index_coalesce_test.py# 인덱스 병합 테스트
│       ├── index_updater_test.py # 인덱스 업데이트 테스트
│       └── tokenizers_test.py    # 토크나이저 테스트
│
├── baleen/                   # Baleen 멀티홉 검색 프레임워크
│   ├── engine.py             # Baleen 엔진 (반복적 검색-압축)
│   ├── hop_searcher.py       # 멀티홉 검색기 (FLIPR 변형 사용)
│   ├── condenser/            # 컨텍스트 압축기
│   │   ├── condense.py       # 압축 로직
│   │   ├── model.py          # 압축 모델
│   │   └── tokenization.py   # 압축용 토크나이저
│   └── utils/                # Baleen 유틸리티
│
├── utility/                  # 외부 유틸리티 (벤치마크, 전처리)
│   ├── evaluate/             # 벤치마크 평가 (MS MARCO, LoTTE, EM)
│   ├── preprocess/           # 데이터 전처리 (문서→패시지 분할)
│   ├── rankings/             # 랭킹 결과 처리 (병합, 분할, 튜닝)
│   ├── supervision/          # 학습 데이터 생성 (self-training, triples)
│   └── utils/                # DPR 변환, QA 로더 등
│
├── data/                     # 샘플 데이터
└── docs/                     # 문서 (Sphinx 설정)
```

#### 핵심 데이터 흐름

```
[학습]  Trainer → training.py → ColBERT.forward() → Q/D 인코딩 → dot product → 손실 계산
                                                                        ↓
[인덱싱] Indexer → collection_indexer → collection_encoder → BERT+Linear → L2 norm
                                              ↓
                                    k-means centroid 학습 (FAISS)
                                              ↓
                                    residual 압축 (centroid code + 잔차 양자화)
                                              ↓
                                    디스크 저장 (.codes, .residuals, centroids)
                                                                        ↓
[검색]  Searcher → candidate_generation → centroid @ Q.T (셀 선택)
                                              ↓
                        index_storage → 후보 필터링 (근사 MaxSim)
                                              ↓
                        잔차 복원 → L2 norm → dot product → MaxSim → 최종 랭킹
```

#### C++ 확장 모듈 (성능 최적화)

| 파일 | 역할 |
|---|---|
| `colbert/modeling/segmented_maxsim.cpp` | CPU MaxSim 연산 (pthread 병렬) |
| `colbert/search/filter_pids.cpp` | 근사 MaxSim 후보 필터링 |
| `colbert/indexing/codecs/decompress_residuals.cpp` | 잔차 복원 (인덱싱) |
| `colbert/search/decompress_residuals.cpp` | 잔차 복원 (검색) |
| `colbert/indexing/codecs/packbits.cpp` | 비트 패킹/언패킹 |
| `colbert/search/segmented_lookup.cpp` | 세그먼트 임베딩 룩업 |

#### 주요 설정 기본값 (`colbert/infra/config/settings.py`)

| 설정 | 기본값 | 설명 |
|---|---|---|
| `dim` | 128 | 최종 임베딩 차원 |
| `doc_maxlen` | 220 | 문서 최대 토큰 수 |
| `query_maxlen` | 32 | 쿼리 최대 토큰 수 |
| `similarity` | `"cosine"` | 유사도 함수 (`"cosine"` 또는 `"l2"`) |
| `nbits` | 1 | 잔차 양자화 비트 수 |
| `kmeans_niters` | 4 | k-means 반복 횟수 |
| `bsize` | 32 | 학습 배치 크기 |
| `lr` | 3e-6 | 학습률 |
