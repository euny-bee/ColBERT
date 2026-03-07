# ColBERT 구조 및 연산량 분석

> 분석 일자: 2026-02-20

---

## 1. 임베딩 연산량 (FLOPs) 계산

### 1.1 기본 아키텍처 파라미터 (bert-base-uncased)

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

### 1.2 Transformer 레이어 1개당 FLOPs

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

### 1.3 전체 모델 FLOPs 공식

```
FLOPs_total = 12 × (14,155,776s + 3,072s²)     ← BERT 12 레이어
            + 2 × s × 768 × 128                  ← Linear projection (768→128)
            = 169,869,312s + 36,864s² + 196,608s
            = 170,065,920s + 36,864s²
```

### 1.4 실제 연산량

**Query (s = 32):**
```
FLOPs = 170,065,920 × 32 + 36,864 × 32²
      = 5,442,109,440 + 37,748,736
      = 5,479,858,176
      ≈ 5.48 GFLOPs
```

**Document (s = 220, 최대 길이):**
```
FLOPs = 170,065,920 × 220 + 36,864 × 220²
      = 37,414,502,400 + 1,784,217,600
      = 39,198,720,000
      ≈ 39.2 GFLOPs
```

**MaxSim 스코어링 (query-document 쌍):**
```
(32, 128) × (128, 220) = 2 × 32 × 128 × 220 = 1,802,240 FLOPs ≈ 1.8 MFLOPs
```

### 1.5 요약

| 연산 | FLOPs | 비고 |
|---|---|---|
| **Query 1개 인코딩** | **~5.48 GFLOPs** | s=32 고정 (MASK 패딩) |
| **Document 1개 인코딩** | **~39.2 GFLOPs** | s=220 최대 기준 |
| **Document 1개 (평균 s≈128)** | **~22.4 GFLOPs** | 실제 평균 길이 기준 |
| MaxSim 스코어링 | ~1.8 MFLOPs | query-doc 쌍 당 |
| **컬렉션 N개 문서 인코딩** | **~39.2N GFLOPs** | 인덱싱 시 총 비용 |

> 연산량의 99.9%는 BERT 인코더에서 발생. self-attention의 s² 항은 짧은 시퀀스(32, 220)에서는 선형 항 대비 비중이 작다 (query 0.7%, document 4.6%).

---

## 2. 유사도/거리 계산 코드 위치

### 2.1 Dot Product (`@` 연산자)

| 위치 | 코드 | 용도 |
|---|---|---|
| `colbert/modeling/colbert.py:175` | `D_padded @ Q.permute(0,2,1)` | 핵심 query-doc 스코어링 |
| `colbert/modeling/colbert.py:195` | `D_packed @ Q.T` | packed 변형 |
| `colbert/modeling/colbert.py:69` | `D.unsqueeze(0) @ Q.permute(0,2,1).unsqueeze(1)` | in-batch negatives 학습 |
| `colbert/search/candidate_generation.py:13` | `self.codec.centroids @ Q.T` | IVF 셀 선택 |
| `colbert/search/candidate_generation.py:43` | `Q.unsqueeze(0) @ E.unsqueeze(2)` | 개별 임베딩 스코어링 |
| `colbert/indexing/codecs/residual.py:215-217` | `self.centroids @ batch.T` | centroid 할당 |

### 2.2 Cosine Similarity (L2 정규화 + dot product)

ColBERT는 `F.cosine_similarity()`를 직접 호출하지 않고, L2 정규화 후 dot product로 cosine similarity를 구현한다.

| 위치 | 코드 | 용도 |
|---|---|---|
| `colbert/modeling/colbert.py:93` | `F.normalize(Q, p=2, dim=2)` | Query 인코더 정규화 |
| `colbert/modeling/colbert.py:104` | `F.normalize(D, p=2, dim=2)` | Document 인코더 정규화 |
| `colbert/indexing/codecs/residual.py:271-273` | `F.normalize(centroids_, p=2, dim=-1)` | 복원된 임베딩 정규화 (GPU) |
| `colbert/search/index_storage.py:185` | `F.normalize(D_packed, p=2, dim=-1)` | 복원된 임베딩 정규화 (CPU) |
| `colbert/indexing/collection_indexer.py:306` | `F.normalize(centroids, dim=-1)` | Centroid 정규화 |
| `colbert/modeling/checkpoint.py:34` | `torch.mm(embs, embs.t())` | 토큰 self-similarity (풀링용) |

### 2.3 Euclidean / L2 Distance

| 위치 | 코드 | 용도 |
|---|---|---|
| `colbert/modeling/colbert.py:117-121` | `(-1.0 * ((Q.unsqueeze(2)-D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)` | 음수 제곱 L2 거리 MaxSim (`config.similarity='l2'`) |
| `colbert/modeling/checkpoint.py:40` | `linkage(1-cosine_sim, metric="euclidean", method="ward")` | Ward 계층적 클러스터링 |

### 2.4 MaxSim (ColBERT 핵심 연산)

| 위치 | 구현 | 용도 |
|---|---|---|
| `colbert/modeling/colbert.py:132-154` | `scores_padded.max(1).values` → `.sum(-1)` | GPU MaxSim 집계 (Python) |
| `colbert/modeling/segmented_maxsim.cpp:22-93` | pthread 병렬 max + sum | CPU MaxSim (C++ 멀티스레드) |
| `colbert/search/filter_pids.cpp:27-69` | centroid score 기반 근사 MaxSim | 후보 문서 사전 필터링 |
| `colbert/modeling/colbert.py:139-152` | `topk(K1).values.sum(-1)` | FLIPR TopK-MaxSim 변형 (Baleen) |

---

## 3. 프로젝트 구조

```
ColBERT/
├── colbert/                  ★ 핵심 패키지
│   ├── modeling/             모델 정의 (ColBERT, HF 통합, 토크나이저, 리랭커)
│   │   ├── colbert.py        ColBERT 모델 (query/doc 인코딩, 스코어링, MaxSim)
│   │   ├── base_colbert.py   기본 클래스 (체크포인트 로딩)
│   │   ├── hf_colbert.py     HuggingFace 통합 (BERT→Linear 768→128)
│   │   ├── checkpoint.py     체크포인트 관리, 인코딩 유틸리티, 풀링
│   │   ├── segmented_maxsim.cpp  MaxSim C++ 구현
│   │   ├── tokenization/     Query/Doc 토크나이저
│   │   └── reranker/         ELECTRA 리랭커
│   │
│   ├── indexing/             인덱싱 파이프라인
│   │   ├── collection_indexer.py   메인 로직 (k-means, 인코딩)
│   │   ├── collection_encoder.py   배치 인코딩
│   │   ├── index_saver.py          디스크 저장
│   │   └── codecs/           잔차 압축 코덱 (residual.py, C++ 확장)
│   │
│   ├── search/               검색 파이프라인
│   │   ├── candidate_generation.py  후보 생성 (centroid→셀→문서)
│   │   ├── index_storage.py         스코어링, 필터링
│   │   ├── strided_tensor.py        가변 길이 텐서
│   │   └── *.cpp                    C++ 최적화 (filter, decompress, lookup)
│   │
│   ├── training/             학습 (DDP 분산 학습, eager/lazy/rerank 배처)
│   ├── data/                 데이터 추상화 (Collection, Queries, Ranking, Dataset)
│   ├── infra/                설정 (ColBERTConfig), 런처, 실험 추적
│   ├── distillation/         지식 증류
│   ├── evaluation/           평가 메트릭
│   ├── utils/                공통 유틸리티
│   └── tests/                테스트 (E2E, 인덱스, 토크나이저)
│
├── baleen/                   멀티홉 검색 프레임워크 (FLIPR, 컨텍스트 압축)
├── utility/                  벤치마크 평가, 전처리, 랭킹, 학습 데이터 생성
├── server.py                 Flask 검색 API 서버
└── setup.py                  패키지 설치 (v0.2.22)
```

### 핵심 데이터 흐름

```
[인덱싱] Indexer → BERT+Linear → L2 norm → k-means → residual 압축 → 디스크 저장
[검색]   Searcher → centroid@Q.T (셀선택) → 후보 필터링 → 잔차복원 → MaxSim → 랭킹
[학습]   Trainer → Q/D 인코딩 → dot product → CrossEntropy 손실
```

### C++ 확장 모듈

| 파일 | 역할 |
|---|---|
| `colbert/modeling/segmented_maxsim.cpp` | CPU MaxSim 연산 (pthread 병렬) |
| `colbert/search/filter_pids.cpp` | 근사 MaxSim 후보 필터링 |
| `colbert/indexing/codecs/decompress_residuals.cpp` | 잔차 복원 (인덱싱) |
| `colbert/search/decompress_residuals.cpp` | 잔차 복원 (검색) |
| `colbert/indexing/codecs/packbits.cpp` | 비트 패킹/언패킹 |
| `colbert/search/segmented_lookup.cpp` | 세그먼트 임베딩 룩업 |

### 주요 설정 기본값 (colbert/infra/config/settings.py)

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




