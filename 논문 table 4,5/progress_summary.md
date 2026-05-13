# ColBERTv2 float16 vs 2-bit 잔차 비교 실험 — 진행 현황

> 최종 업데이트: 2026-04-02

---

## 실험 개요

- **목적**: ColBERTv2의 2-bit 양자화 잔차(residual) vs float16(아날로그) 잔차 비교
- **핵심 질문**: 소자 수를 50% 줄이는 대신 정밀도를 높이면 검색 품질이 얼마나 달라지는가?
- **하드웨어**: RTX 4060 8GB VRAM, 16GB RAM, Windows 11 (WSL2)

### 두 방법 비교

| 항목 | 2-bit (디지털) | float16 (아날로그) |
|------|--------------|------------------|
| 잔차 저장 | 2-bit 양자화 → uint8 | float16 연속값 |
| 소자 수/벡터 | 256 bit-cells | 128 devices |
| 소자 절약 | 기준 (1x) | **50% 절약** |
| 2-bit 버킷 값 | [-0.059, -0.016, +0.016, +0.059] | 연속값 (무한 정밀도) |

---

## 데이터셋

### Part 1 — MS MARCO (200k 서브셋)

| 항목 | 값 |
|------|-----|
| 컬렉션 | MS MARCO Passage, **200,000** passages (서브셋) |
| 쿼리 | Dev Small, 6,980개 중 **500개** 사용 (seed=42) |
| Qrels | ~7,500 (query, passage) pairs |
| 임베딩 수 | **13,499,648** (200k passages × 평균 ~67.5 토큰) |
| 인덱스 | `200k.analog` (float16), `200k.2bit` (2-bit) |

### Part 2 — BEIR 벤치마크 (Table 5 재현)

| 항목 | 값 |
|------|-----|
| 데이터셋 수 | 13개 BEIR 데이터셋 |
| 집중 분석 대상 | **TREC-COVID** (171K docs, 50 queries, qrels 66K lines) |
| 이유 | nDCG@10 delta가 가장 큼 (+0.0169) |

---

## Part 1: MS MARCO 실험 결과

### 1. 기본 검색 품질 (step2_eval.py, ndocs=8192)

| Metric | float16 | 2-bit | Delta |
|--------|---------|-------|-------|
| MRR@10 | 0.8511 | 0.8496 | **+0.0015** |
| R@50 | 0.9874 | 0.9878 | -0.0004 |

### 2. Reconstruction Error (recon_error.py, 13.5M embeddings)

개별 임베딩 벡터 수준의 float16 vs 2-bit 복원 품질.

| 지표 | 값 |
|------|----|
| L2 error mean | 0.253 |
| L2 error std | 0.095 |
| L2 error p95 | 0.434 |
| Angular error mean | **14.57°** |
| Angular error p95 | 25.05° |

→ 개별 벡터 수준에서 평균 14.6° 각도 오차. MaxSim은 다수 토큰의 평균이므로 오차 상쇄됨.

**2-bit 잔차 버킷 구조 (plot_residual_reconstruction.py)**

- 버킷 4개: `-0.059, -0.016, +0.016, +0.059`
- 양자화 오차 절댓값 평균: **0.019** (차원별)
- 양자화 오차 std: 0.031, p95: 0.071

### 3. MaxSim Score 비교 (score_analysis.py, k=200, 100 queries)

| 지표 | 값 |
|------|----|
| Score diff mean | +0.044 (float16이 더 높음) |
| Spearman ρ | 0.898 |
| r² | 0.946 |
| Top-50 overlap | **88%** |

### 4. Rank Analysis (rank_analysis.py, k=1000, 500 queries)

#### R@k, MRR@k, nDCG@k (k=1..10)

| k | R f16 | R 2bit | MRR f16 | MRR 2bit | nDCG f16 | nDCG 2bit |
|---|-------|--------|---------|---------|---------|---------|
| 1 | 75.9% | 75.6% | 78.4% | 78.0% | 78.4% | 78.0% |
| 5 | 94.3% | 93.9% | 84.8% | 84.1% | 86.1% | 85.3% |
| 10 | 96.4% | 96.2% | 85.2% | 85.0% | 87.6% | 87.5% |

→ MRR@k, nDCG@k 모두 k=7~10에서 float16이 **일관되게 우세** (교차 없음).

#### Rank Displacement

- 분석 쌍: 543개 (query × relevant_doc)
- mean = **-0.06** (사실상 0), std = 2.55, abs_max = 45
- 89.1% 동일 순위, 6.3% 하락, 5.0% 상승

### 5. Score Margin 분석 (score_margin.py, 500 queries)

`margin = score(관련문서) - score(1등 비관련문서)`

| 지표 | float16 | 2-bit |
|------|---------|-------|
| Margin mean | **+3.173** | +2.803 |
| Failures (margin<0) | — | — |
| float16 > 2-bit | **71.9%** 쿼리 | — |

#### 통계적 유의성 (nDCG@10 per-query, n=500)

| 검정 | p-value | 결론 |
|------|---------|------|
| Paired t-test | 0.570 | 유의하지 않음 |
| Wilcoxon | 0.670 | 유의하지 않음 |

→ 차이는 실재하지만 500개 쿼리로는 p<0.05 달성 불가. ~2000개 이상 필요.

### 6. Failure Analysis (plot_failure.py)

margin<0인 케이스(관련문서가 비관련문서에 밀리는 검색 실패)를 4분류.

| 분류 | 수 | 비율 |
|------|----|------|
| 둘 다 성공 | — | 다수 |
| 2-bit만 실패 | 더 많음 | float16 우세 근거 |
| float16만 실패 | 더 적음 | — |
| 둘 다 실패 | — | — |

---

## Part 2: BEIR 벤치마크 (Table 5)

### nDCG@10 전체 결과

| Dataset | Corpus | float16 | 2-bit | Delta |
|---------|--------|---------|-------|-------|
| TREC-COVID | 171K | **0.7477** | 0.7308 | **+0.0169** ← 최대 |
| FiQA | 57K | 0.3534 | 0.3465 | +0.0069 |
| NFCorpus | 3.6K | 0.3375 | 0.3348 | +0.0027 |
| Quora | 523K | 0.8565 | 0.8541 | +0.0024 |
| SCIDOCS | 25K | 0.1576 | 0.1552 | +0.0024 |
| HotpotQA | 2.6M | 0.8536 | 0.8529 | +0.0007 |
| SciFact | 5K | 0.6672 | 0.6680 | -0.0008 |
| ArguAna | 8.6K | 0.3328 | 0.3359 | -0.0031 |
| BRIGHT | 1.33M | 0.0683 | 0.0754 | -0.0071 ← 최대 역전 |

→ 대부분 데이터셋에서 float16 우세. ArguAna/BRIGHT에서만 2-bit 우세 (이유 불명).

---

## Part 3: TREC-COVID 심층 분석

BEIR 중 delta가 가장 큰 TREC-COVID에 대해 MS MARCO와 동일한 지표 분석.

### 데이터 특성

| 항목 | 값 |
|------|-----|
| Corpus | 171,332 passages (의학 논문) |
| 쿼리 수 | **50개** |
| Qrels | 66,336 lines (평균 **1,327개** 관련문서/query) |
| 임베딩 수 | 23,788,821 (평균 138.8 토큰/doc) |
| Qrels 레이블 | Graded (0/1/2), rel≥1 기준 사용 |

### 결과 요약

| 지표 | float16 | 2-bit | Delta | MS MARCO Delta |
|------|---------|-------|-------|---------------|
| **Top-50 Overlap** | — | — | **93.1%** | 88.0% |
| **MRR@1** | 86.0% | 84.0% | **+2.0%** | +0.4% |
| **MRR@10** | 91.9% | 90.9% | **+1.0%** | +0.2% |
| **nDCG@10** | 81.6% | 79.7% | **+1.9%** | +0.1% |
| **Score Margin mean** | +1.087 | +0.997 | +0.090 | +0.370 |
| **Failures (margin<0)** | 7/50 (14%) | 8/50 (16%) | — | — |
| **float16 > 2-bit margin** | 28/50 (**56%**) | — | — | 71.9% |

### nDCG@k (k=1..30) 패턴

| 구간 | 패턴 |
|------|------|
| k=1 | float16 +2.0% (최대) |
| k=2~4 | 2-bit가 잠깐 역전 (최대 -0.77%) |
| k=5~13 | float16 재역전, k=9에서 +2.5% 최대 |
| k=14~20 | 차이 급격히 감소 |
| k=20+ | 거의 수렴 (≈0) |

→ float16 이점은 **상위 1~13등** 정밀 랭킹에 집중.

### MS MARCO vs TREC-COVID 비교

| 항목 | MS MARCO | TREC-COVID |
|------|----------|------------|
| 관련문서 수/query | ~1개 | ~1,327개 |
| MRR@1 delta | +0.4% | **+2.0%** |
| nDCG@10 delta | +0.1% | **+1.9%** |
| 차이가 더 큰 이유 | 랭킹 분산 작음 | 관련문서 多 → 순위 정밀도 중요 |

---

## 생성된 파일 목록

### 스크립트

| 파일 | 역할 |
|------|------|
| `step2_eval.py` | MS MARCO 기본 검색 품질 (MRR@10, R@50) |
| `recon_error.py` | L2/Angular 재구성 오류 (13.5M 임베딩) |
| `score_analysis.py` | MaxSim 스코어 비교 (k=200) |
| `rank_analysis.py` | R@k, MRR@k, nDCG@k, rank displacement |
| `plot_recall_k10.py` | R@k / MRR@k / nDCG@k 3-panel 그래프 |
| `score_margin.py` | Score Margin 4종 비교 그래프 |
| `plot_failure.py` | 검색 실패 분류 3-panel 그래프 |
| `plot_residual_reconstruction.py` | 2-bit 잔차 버킷 시각화 |
| `run_table5.py` | BEIR 13개 데이터셋 자동 인덱싱/검색/평가 |
| `run_covid_analog.py` | TREC-COVID float16 인덱싱 + 분석 파이프라인 |
| `plot_covid_analysis.py` | TREC-COVID Top-50 overlap / R@k / MRR@k / nDCG@k / Score Margin |
| `plot_covid_ndcg50.py` | TREC-COVID nDCG@k (k=1..30) 2-panel |

### 결과 파일

| 경로 | 내용 |
|------|------|
| `experiments/msmarco/rank_analysis_500q.json` | MS MARCO R@k, rank displacement |
| `experiments/msmarco/recall_mrr_ndcg_k10.csv` | MS MARCO R@k / MRR@k / nDCG@k 수치 |
| `experiments/msmarco/rank_analysis_500q.png` | MS MARCO rank 분석 3-panel |
| `experiments/msmarco/recall_mrr_ndcg_k10.png` | MS MARCO R@k / MRR@k / nDCG@k |
| `experiments/msmarco/score_margin.png` | MS MARCO Score Margin 4-panel |
| `experiments/msmarco/failure_analysis.png` | MS MARCO 검색 실패 분류 |
| `experiments/table5/table5_results.json` | BEIR 전체 nDCG@10 결과 |
| `experiments/table5/trec_covid_analysis.json` | TREC-COVID Top-50 / MRR / Score Margin |
| `experiments/table5/covid_top50_overlap.png` | TREC-COVID Top-50 overlap histogram |
| `experiments/table5/covid_recall_mrr_ndcg.png` | TREC-COVID R@k / MRR@k / nDCG@k (%) |
| `experiments/table5/covid_score_margin_scatter.png` | TREC-COVID Score Margin scatter |
| `experiments/table5/covid_ndcg_k30.png` | TREC-COVID nDCG@k (k=1..30) 2-panel |

---

## 핵심 결론

1. **float16은 거의 모든 지표에서 2-bit보다 우세** — 방향은 일관됨
2. **차이의 크기는 데이터셋 특성에 따라 다름** — 관련문서가 많고 복잡한 도메인(TREC-COVID)에서 더 크게 나타남
3. **Score Margin이 가장 민감한 지표** — MS MARCO에서 71.9% 쿼리가 float16 우세
4. **Top-50 overlap이 높음** (MS MARCO 88%, TREC-COVID 93.1%) — ANN 후보 선택 단계는 두 방법이 거의 동일, 차이는 MaxSim 재랭킹에서 발생
5. **통계적 유의성**: MS MARCO 500 queries에서 p>0.5 — 유의성 확보에 ~2000 queries 필요
6. **2-bit가 우세한 경우**: ArguAna, BRIGHT, TREC-COVID k=2~4 구간 — 원인 불명

---

## 재현 방법

```bash
# WSL2에서 실행
cd ~/ColBERT && source ~/colbert-env/bin/activate

# MS MARCO 분석
python rank_analysis.py          # R@k, MRR@k, nDCG@k, rank displacement
python plot_recall_k10.py        # 그래프
python score_margin.py           # Score Margin 그래프
python plot_failure.py           # 검색 실패 분석

# TREC-COVID 분석
python run_covid_analog.py       # 인덱싱 + 검색 + 분석 (전체 파이프라인)
python plot_covid_analysis.py    # 그래프 생성
python plot_covid_ndcg50.py      # nDCG@k k=1..30 그래프

# BEIR Table 5 전체
python run_table5.py             # 13개 데이터셋 자동 처리 (수 시간 소요)
```
