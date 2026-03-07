# ColBERT 프로젝트 컨텍스트

## 개요
Stanford Future Data Lab에서 개발한 신경 정보 검색 라이브러리. Late Interaction (MaxSim) 메커니즘으로 대규모 문서 컬렉션에서 효율적인 검색을 수행한다.

- **버전**: 0.2.22
- **기본 모델**: bert-base-uncased (12 layers, hidden=768, FFN=3072)
- **임베딩 차원**: 128 (768→128 Linear projection, bias 없음)
- **쿼리 최대 길이**: 32 토큰 (MASK 패딩)
- **문서 최대 길이**: 220 토큰

## 핵심 아키텍처

### 임베딩 연산량 (FLOPs)
- **Query 1개**: ~5.48 GFLOPs (s=32)
- **Document 1개**: ~39.2 GFLOPs (s=220 최대), ~22.4 GFLOPs (평균 s≈128)
- **FLOPs 공식**: `170,065,920 × s + 36,864 × s²` (BERT 12층 + Linear projection)
- 연산량의 99.9%는 BERT 인코더에서 발생

### 유사도 계산 방식
ColBERT는 `F.cosine_similarity()`를 직접 사용하지 않는다. Q와 D를 `F.normalize(p=2)`로 단위 벡터화한 뒤, `@` 연산자로 dot product를 수행하여 cosine similarity를 얻는 방식이다.

주요 유사도/거리 계산 위치:
- **Dot product (핵심 스코어링)**: `colbert/modeling/colbert.py:175` — `D_padded @ Q.permute(0,2,1)`
- **L2 정규화 (Query)**: `colbert/modeling/colbert.py:93`
- **L2 정규화 (Doc)**: `colbert/modeling/colbert.py:104`
- **L2 distance MaxSim**: `colbert/modeling/colbert.py:121` — `config.similarity='l2'` 설정 시
- **MaxSim (GPU)**: `colbert/modeling/colbert.py:132-154`
- **MaxSim (CPU C++)**: `colbert/modeling/segmented_maxsim.cpp:22-93`
- **Centroid dot product**: `colbert/search/candidate_generation.py:13`

### 데이터 흐름
```
[인덱싱] Indexer → BERT+Linear → L2 norm → k-means → residual 압축 → 디스크 저장
[검색]   Searcher → centroid@Q.T (셀선택) → 후보 필터링 → 잔차복원 → MaxSim → 랭킹
[학습]   Trainer → Q/D 인코딩 → dot product → CrossEntropy 손실
```

## 프로젝트 구조 요약
- `colbert/modeling/` — 모델 정의 (ColBERT, HF 통합, 토크나이저)
- `colbert/indexing/` — 인덱싱 파이프라인 (인코딩, k-means, 잔차 압축)
- `colbert/search/` — 검색 파이프라인 (후보 생성, 스코어링, C++ 최적화)
- `colbert/training/` — 학습 (DDP 분산 학습, 배처)
- `colbert/infra/` — 설정 (ColBERTConfig), 런처, 실험 추적
- `colbert/data/` — 데이터 추상화 (Collection, Queries, Ranking)
- `baleen/` — 멀티홉 검색 프레임워크 (FLIPR 변형)
- `utility/` — 벤치마크 평가, 전처리, 랭킹 유틸리티
