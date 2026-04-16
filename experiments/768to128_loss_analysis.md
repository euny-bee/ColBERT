# 768-dim float32 → 128-dim float16 벡터 손실 분석

> Branch: `experiment/768-32to128-16`
> 최종 업데이트: 2026-04-16

---

## 실험 목적

BERT encoder 출력 **(768-dim, float32)** 이 ColBERT에서 **(128-dim, float16)** 으로 압축될 때 발생하는 손실을 분석한다.

- **차원 축소 손실**: 768 → 128 (Linear projection)
- **Precision 손실**: float32 → float16
- **검색 성능 영향**: Recall, MRR, nDCG 지표 비교

---

## 비교 대상

| 표기 | 모델 | 차원 | dtype | 비고 |
|------|------|------|-------|------|
| **A** | msmarco-bert-base-dot-v5 | 768 | float32 | MS MARCO 학습, CLS 토큰 |
| **B** | colbertv2.0 projection | 128 | float32 | Linear projection + mean pooling + L2 norm |
| **C** | colbertv2.0 projection | 128 | float16 | B.half() |

> A를 raw BERT가 아닌 `msmarco-bert-base-dot-v5`로 선택한 이유: raw BERT와 비교하면 차원 축소와 학습 도메인 차이가 혼재되어 불공정한 비교가 됨. MS MARCO로 같은 학습 데이터를 쓴 768-dim 모델을 baseline으로 사용.

> B/C에서 mean pooling을 쓴 이유: 실제 ColBERT는 MaxSim 방식이나, 단일 벡터 비교를 위해 mean pooling 사용. 따라서 B/C 성능은 실제 ColBERT보다 낮게 측정됨.

---

## 데이터셋

- **BEIR TREC-COVID**
- Corpus: 171,332개 passages
- Queries: 50개
- Qrels: test.tsv

---

## Phase 1: 벡터 수준 손실 분석

**스크립트**: `phase1_vector_analysis.py`
**샘플**: 2,000개 passages
**출력**: `phase1_results.png`

### Phase 1-1: 차원 축소 손실 (A 768 → B 128)

| 분석 항목 | 결과 |
|-----------|------|
| PCA Top-32 explained variance | 64.9% |
| PCA Top-64 explained variance | 81.6% |
| PCA Top-128 explained variance | 92.5% |
| Cosine sim Spearman (A' vs B) | 0.67 |

- 128-dim PCA로 원래 분산의 92.5% 보존 가능 (이론적 상한)
- ColBERT linear projection(B)과 PCA(A')의 pairwise cosine sim 순위 상관 = 0.67 → 두 공간은 다른 방향으로 정보를 압축함

### Phase 1-2: Precision 손실 (B float32 → C float16)

| 분석 항목 | 결과 |
|-----------|------|
| MSE per vector (mean) | 3.37e-10 |
| Max abs error per vector (mean) | 6.68e-05 |
| Cosine similarity B vs C (mean) | 1.000000 |

**결론: float32 → float16 변환은 사실상 손실 없음.**

---

## Phase 2+3: 검색 성능 비교 및 순위 보존율

**스크립트**: `phase2_retrieval_comparison.py`
**출력**: `phase2_results.png`
**임베딩 캐시**: `emb_cache/` (A_corpus.pt, A_query.pt, B_corpus.pt, B_query.pt)

### Phase 2: 검색 성능 (Retrieval Metrics)

| Metric | A (768 f32) | B (128 f32) | C (128 f16) | B-A | C-A |
|--------|------------|------------|------------|-----|-----|
| nDCG@10 | 0.4814 | 0.2208 | 0.2210 | -0.2606 | -0.2604 |
| MRR@10 | 0.6957 | 0.4882 | 0.4882 | -0.2076 | -0.2076 |
| Recall@5 | 0.0071 | 0.0032 | 0.0032 | -0.0039 | -0.0039 |
| Recall@10 | 0.0130 | 0.0051 | 0.0051 | -0.0079 | -0.0079 |
| Recall@50 | 0.0525 | 0.0159 | 0.0159 | -0.0366 | -0.0366 |
| Recall@100 | 0.0917 | 0.0241 | 0.0240 | -0.0676 | -0.0677 |

> B와 C가 동일한 이유: float16 변환으로 인한 성능 차이가 전혀 없음 (Phase 1 결과와 일치).

### Phase 3: 순위 보존율 (Ranking Preservation)

| 항목 | A vs B | A vs C | B vs C |
|------|--------|--------|--------|
| Spearman rank corr (top-100 mean) | 0.255 | 0.256 | 1.000 |
| Top-10 overlap (mean) | 2.34/10 | 2.34/10 | 10.00/10 |

- A vs B: Spearman 0.255 → 순위가 약하게만 일치 (768과 128이 다른 문서를 선호)
- B vs C: Spearman 1.000, Top-10 overlap 10/10 → 완전히 동일 (float16 손실 = 0 재확인)

### 쿼리별 nDCG@10 A→B 변화

**가장 손실 큰 쿼리 (A→B nDCG@10 하락):**
- [27] -0.8669 | what is known about those infected with Covid-19 but are asymptomatic?
- [37] -0.7992 | What is the result of phylogenetic analysis of SARS-CoV-2 genome sequence?
- [3]  -0.7425 | will SARS-CoV2 infected people develop immunity? Is cross protection possible?

**가장 이득 큰 쿼리 (A→B nDCG@10 상승):**
- [28] +0.1534 | what evidence is there for the value of hydroxychloroquine in treating Covid-19?
- [33] +0.1463 | What vaccine candidates are being tested for Covid-19?
- [26] +0.1050 | what are the initial symptoms of Covid-19?

---

## Phase 2 Visualization

**스크립트**: `phase2_visualization.py`
**출력**: `phase2_visualization.png`

- t-SNE: A(768-dim), B(128-dim) 벡터 공간 시각화 (relevant/non-relevant 색상 구분)
- Score distribution: A와 B를 하나의 그래프에 오버레이
- Per-query nDCG@10 scatter: A vs B
- Relevant document rank change histogram: A→B 순위 변화
- Discriminability: 쿼리별 관련/비관련 점수 격차

---

## 주요 결론

1. **float32 → float16은 손실 없음**: MSE ~3e-10, cosine sim = 1.0000, Spearman B vs C = 1.000
2. **768 → 128 차원 축소가 실질적 손실 원인**: nDCG@10 -0.26, top-10 overlap 2.34/10
3. **단, mean pooling 방식 한계 있음**: ColBERT는 본래 MaxSim 방식이므로 B/C 성능이 실제보다 낮게 측정됨
4. **쿼리마다 손실 편차 크다**: 최대 -0.87 ~ +0.15 범위

---

## 스크립트 목록

| 파일 | 역할 |
|------|------|
| `phase1_vector_analysis.py` | A/B/C 임베딩 추출 및 벡터 수준 손실 분석 |
| `phase2_retrieval_comparison.py` | 검색 성능 비교 + 순위 보존율 분석 + 캐시 저장 |
| `phase2_visualization.py` | t-SNE, score distribution, discriminability 시각화 |

## 출력 파일

| 파일 | 설명 |
|------|------|
| `phase1_results.png` | Phase 1 분석 결과 (6개 subplot) |
| `phase2_results.png` | Phase 2+3 검색 성능 비교 (4개 subplot) |
| `phase2_visualization.png` | 벡터 공간 + 검색 결과 시각화 (8개 subplot) |
| `emb_cache/A_corpus.pt` | A 모델 corpus 임베딩 캐시 |
| `emb_cache/A_query.pt` | A 모델 query 임베딩 캐시 |
| `emb_cache/B_corpus.pt` | B 모델 corpus 임베딩 캐시 |
| `emb_cache/B_query.pt` | B 모델 query 임베딩 캐시 |
