# ColBERTv2 Analog Pipeline 진행 요약

> 작성일: 2026-06-02  
> 작업 폴더: `C:\Users\nmdl-khb\ColBERT\centroidset`

---

## 1. 목표

ColBERTv2의 Top-k 문서 추출 파이프라인을 **아날로그 회로**로 대체했을 때,
디지털 소프트웨어 대비 retrieval 성능이 얼마나 유지되는지 검증

---

## 2. 핵심 지표

**비교 대상 4가지 방법:**

| 방법 | 설명 |
|------|------|
| Software f32 | dot product + 원본 벡터 (압축 없음) |
| Software 2bit | dot product + 2bit 압축 벡터 |
| Analog f32 | 아날로그 전류 + 원본 벡터 |
| Analog 2bit | 아날로그 전류 + 2bit 압축 벡터 (실제 하드웨어) |

---

### ① MRR@10 (Mean Reciprocal Rank) — 핵심 지표

```
MRR@10 = (1/N) × Σᵢ (1/rankᵢ)   단, rankᵢ > 10이면 0
```

- 정답이 rank 1 → 1.0 / rank 2 → 0.5 / rank 5 → 0.2 / rank 11+ → 0
- **Success@k보다 민감** — rank 1과 rank 5의 차이를 포착
- MS MARCO 표준 지표 (ColBERTv2 논문 Table 4)

---

### ② Success@k (= Recall@k)

```
Success@k = (정답이 rank ≤ k인 쿼리 수) / (전체 쿼리 수) × 100%
```

- k = **1, 5, 10, 50** 측정
- MS MARCO는 query당 정답 1개 → R@k = Success@k 동일
- ColBERTv2 논문: LoTTE·OpenQA에서 Success@5 사용

---

### ③ R@k (Recall@k)

```
R@k = (top-k 안에 포함된 정답 수) / (전체 정답 수)
```

- k = **50, 1000** 측정
- MS MARCO 단일 정답 구조에서는 Success@k와 동일
- ColBERTv2 논문 Table 4: R@50, R@1k 보고

---

### ④ nDCG@10 (normalized Discounted Cumulative Gain)

```
nDCG@10 = DCG@10 / IDCG@10
DCG@k   = Σᵢ₌₁ᵏ relᵢ / log₂(i+1)
```

- 순위가 높을수록 가중치 부여
- BEIR 벤치마크 표준 지표 (ColBERTv2 논문 Table 5a)
- MS MARCO는 relevance가 0/1 이진 → Success@k와 유사하나 순위 가중치 반영

---

### 현재 결과 (3 queries, 200 documents)

| 지표 | Software f32 | Software 2bit | Analog f32 | Analog 2bit |
|------|-------------|--------------|-----------|------------|
| MRR@10 | **1.000** | **1.000** | **1.000** | **1.000** |
| Success@1 | **100%** | **100%** | **100%** | **100%** |
| Success@5 | **100%** | **100%** | **100%** | **100%** |
| Success@10 | **100%** | **100%** | **100%** | **100%** |
| R@50 | **100%** | **100%** | **100%** | **100%** |
| nDCG@10 | **1.000** | **1.000** | **1.000** | **1.000** |

→ 현재 규모(3 queries, 200 docs)에서는 모든 지표 동일 (margin이 너무 커서 오차 영향 없음)

---

### 목표 (스케일업 후: 500 queries, 50K docs)

- 4가지 방법의 모든 지표 비교
- margin 구간별 (0~1, 1~2, 2~5, 5+) 지표 분포 분석
- Software vs Analog 간 차이가 통계적으로 유의미한지 확인

---

## 3. 데이터셋

| 항목 | 내용 |
|------|------|
| Queries | MS MARCO dev 3개 (`qrels.dev.small.tsv`) |
| Documents | MS MARCO passage 200개 (12,919 token embeddings) |
| Centroids | 100개 (k-means) |
| Embedding dim | 128차원, L2-normalized |

### Query ↔ 정답 Document 매핑 (MaxSim으로 확인)

| Query ID | 질문 | 정답 pid |
|----------|------|---------|
| q0 | where does real insulin come from | 7264308 |
| q1 | where does name nora come from | 7264266 |
| q2 | where does most of the iron ore come from | 7264253 |

---

## 4. ColBERTv2 Top-k 추출 파이프라인

```
Step 1  Query 토큰 32개 인코딩
Step 2  각 토큰마다 가까운 centroid nprobe=2개 선택
Step 3  Inverted list 조회 → 후보 pid 수집
Step 4  (Step 3에서 pid 직접 획득 — 이미 완료)
Step 5  후보 문서 벡터 복원
Step 6  MaxSim 계산 → 최종 랭킹
```

### Step별 핵심 파일

| Step | 파일 | 내용 |
|------|------|------|
| 1 | `query_embs_96x128.xlsx` | 3 queries × 32 tokens × 128 dims |
| 2 | `query_centroid_ranking.xlsx` | 96 tokens × 100 centroids dot product |
| 2 (analog) | `step2_analog_centroid.xlsx` | 아날로그 전류 기반 centroid 선택 |
| 3 | `step3_candidate_pids.xlsx` | query별 후보 pid (digital) |
| 3 (analog) | `step3_analog_candidate_pids.xlsx` | query별 후보 pid (analog) |
| 6 | `step6_all_results.xlsx` | 4가지 방법 랭킹 결과 |

---

## 5. 문서 벡터 구조 확인

ColBERTv2 압축 파이프라인이 파일로 구현되어 있음:

```
원본 벡터 v (128차원, L2-norm=1)
    ↓  k-means
가장 가까운 centroid C_t 선택
    ↓
residual r = v - C_t              ← residuals_float32.xlsx
    ↓  2-bit quantization
r̃: 4가지 값만 [-0.089, -0.022, +0.025, +0.092]  ← residuals_2bit.csv
    ↓
저장: centroid index t + r̃
    ↓  복원
v̂ = C_t + r̃                      ← doc_embs_12919x128.xlsx (float32 기준 복원)
```

**검증 결과:**
- `doc_embs` = `C_t + r_float32` (최대 오차: 0.00000001) ✓
- `C_t + r_2bit` 최대 오차: 0.379 (2bit 압축 손실 존재)

---

## 6. 아날로그 회로 모델

**Option A (Vth compensation):**

```
IDS = f(|V2 - V1|)    ← Vth 상쇄
```

VGS = |V2-V1| + Vth → logistic 함수에서 Vth 항 소거

**die4 파라미터:**

| 파라미터 | 값 |
|---------|---|
| L | 6.112020 |
| K | 2.731949 |
| B | -10.713911 |
| Vth | 0.151045 V |
| VDS | 1.7 V |

**디지털 vs 아날로그 대응:**

| 디지털 | 아날로그 |
|--------|---------|
| dot product `Σₖ Qₖ·Dₖ` | Match Line 전류 `Σₖ IDS(\|Qₖ-Dₖ\|)` |
| 높을수록 유사 | **낮을수록** 유사 |
| argmax | argmin |

---

## 7. Digital vs Analog 후보 집합 비교 (nprobe=2)

| Query | Digital 후보 | Analog 후보 | 교집합 | Jaccard |
|-------|------------|-----------|--------|---------|
| q0 | 127개 | 124개 | 124개 | **0.976** |
| q1 | 70개 | 82개 | 64개 | 0.727 |
| q2 | 108개 | 85개 | 85개 | 0.787 |

- 정답 pid는 3개 query 모두 교집합 안에 포함 ✓

---

## 8. 최종 랭킹 결과 (4가지 방법 비교)

| 방법 | 설명 |
|------|------|
| digital f32 | dot product + C_t+r_float32 벡터 (원본) |
| digital 2bit | dot product + C_t+r_2bit 벡터 (ColBERTv2 압축) |
| analog f32 | 아날로그 전류 + C_t+r_float32 벡터 |
| analog 2bit | 아날로그 전류 + C_t+r_2bit 벡터 (실제 하드웨어 시나리오) |

### Rank 결과

| Query | digital f32 | digital 2bit | analog f32 | analog 2bit |
|-------|------------|-------------|-----------|------------|
| q0 | **1**/127 | **1**/127 | **1**/124 | **1**/124 |
| q1 | **1**/70  | **1**/70  | **1**/82  | **1**/82  |
| q2 | **1**/108 | **1**/108 | **1**/85  | **1**/85  |

4가지 방법 모두 정답 문서 rank 1 → **차이 없음**

### Score 비교 (q0 기준)

| 방법 | Score / Current |
|------|----------------|
| digital f32 | 22.5419 |
| digital 2bit | 16.6461 |
| analog f32 | 164.015 uA |
| analog 2bit | 181.981 uA |

---

## 9. 차이가 없는 이유 분석

### Score Margin (정답 - 2등)

| Query | 정답 score | 2등 score | margin |
|-------|-----------|---------|--------|
| q0 | 22.54 | 10.48 | **12.06** |
| q1 | 25.45 | 12.42 | **13.04** |
| q2 | 24.93 | 13.91 | **11.02** |

- Margin이 압도적으로 큼 → 아날로그 오차(~5%)로는 순위 역전 불가
- 현재 3개 쿼리가 "where does X come from" 형태의 매우 쉬운 쿼리
- Document 200개로 경쟁이 충분하지 않음

---

## 10. 다음 단계: 스케일업 계획

### 목표
500개 query, 50K document에서 digital vs analog 지표 비교

### 추천 규모

| 항목 | 현재 | 목표 | 근거 |
|------|------|------|------|
| Queries | 3개 | **500개** | MS MARCO dev 활용 |
| Documents | 200개 | **50,000개** | msmarco_1m 서브셋 |
| Centroids | 100개 | **32,768개 (2^15)** | 16×√3.4M 권장값 |

### 전체 플랜

**Phase 1. 데이터 준비**
1. `qrels.dev.small.tsv`에서 query 500개 선정 (정답 passage 포함 조건)
2. 50K document pool 구성 (정답 500개 필수 + 나머지 49,500개 추출)
3. `colbert-ir/colbertv2.0` 모델로 query embedding 500개 생성

**Phase 2. 인덱스 구조 구성**
4. 50K docs 토큰 3.4M개로 k-means → centroid 32K개 구축
5. `centroid_id → [token_ids, pids]` IVF 구축

**Phase 3. Digital + Analog 전체 실행 (500 queries × 4가지)**
6. Digital f32 / Digital 2bit: dot product → MaxSim → rank
7. Analog f32 / Analog 2bit: `IDS=f(|V2-V1|)` → MinCurrent → rank

**Phase 4. 지표 계산 및 분석**

**(4-1) 전체 500 queries 기준 지표 — 메인**

| 지표 | Software f32 | Software 2bit | Analog f32 | Analog 2bit |
|------|-------------|--------------|-----------|------------|
| MRR@10 | ? | ? | ? | ? |
| Success@1 | ? | ? | ? | ? |
| Success@5 | ? | ? | ? | ? |
| Success@10 | ? | ? | ? | ? |
| Success@50 | ? | ? | ? | ? |
| R@50 | ? | ? | ? | ? |
| R@1000 | ? | ? | ? | ? |
| nDCG@10 | ? | ? | ? | ? |

**(4-2) Margin 구간별 breakdown — 추가 분석**

| margin 구간 | 쿼리 수 | SW f32 Success@1 | Analog 2bit Success@1 | 차이 |
|------------|--------|-----------------|----------------------|------|
| < 1 (매우 어려움) | ~50개 | ?% | ?% | ?%p |
| 1 ~ 2 (어려움) | ~50개 | ?% | ?% | ?%p |
| 2 ~ 5 (보통) | ~150개 | ?% | ?% | ?%p |
| 5+ (쉬움) | ~250개 | ?% | ?% | ?%p |
| **전체** | **500개** | **?%** | **?%** | **?%p** |

→ margin이 작은 구간일수록 analog 오차가 순위에 영향 → 차이 발생 예상

**(4-3) 개별 케이스 분석**
- digital rank ≠ analog rank인 쿼리 집중 분석

### 예상 소요 시간

| 단계 | 예상 시간 |
|------|---------|
| Query embedding 생성 (500개) | ~10분 |
| Centroid 32K 구축 (k-means) | ~15분 |
| IVF 구축 (50K docs) | ~5분 |
| Digital pipeline (500 queries) | ~5분 |
| Analog pipeline (500 queries) | ~25분 |
| **전체** | **~60분** |

### 보유 인프라 (msmarco_1m)

| 항목 | 내용 |
|------|------|
| 총 passage | 1,100,000개 |
| 총 token embeddings | 74,356,430개 |
| 기존 Centroids | 131,072개 (1.1M 기준, 50K에는 부적합) |
| 평균 토큰/passage | 67.6개 |
| RAM | 128GB (여유 109GB) |

---

## 11. 생성 파일 목록

| 파일 | 내용 |
|------|------|
| `query_embs_96x128.xlsx` | Query token embeddings (3 queries) |
| `step3_candidate_pids.xlsx` | Digital 후보 pid (query별 시트) |
| `step3_analog_candidate_pids.xlsx` | Analog 후보 pid (query별 시트) |
| `step2_analog_centroid.xlsx` | 아날로그 centroid 전류값 |
| `step6_maxsim_results.xlsx` | Digital MaxSim 결과 (f32, 2bit) |
| `step6_all_results.xlsx` | 4가지 방법 전체 비교 결과 |
| `compare_candidates.png` | Digital vs Analog 후보 집합 비교 그래프 |
| `step3_ivf_lookup.py` | Step 3 구현 스크립트 |
| `step6_maxsim.py` | Step 6 digital MaxSim 스크립트 |
| `step_analog.py` | Analog pipeline 전체 스크립트 |
| `compare_candidates.py` | 후보 집합 비교 시각화 스크립트 |
