# ColBERTv2 스케일업 실험 요약

> 작성일: 2026-06-05  
> 작업 폴더: `C:\Users\nmdl-khb\ColBERT\centroidset`  
> 참조: [progress_summary.md](progress_summary.md) (소규모 실험 기반)

---

## 1. 목표

소규모 실험(3 queries, 200 docs)에서 digital/analog 차이가 없었던 원인이  
"쉬운 쿼리 + 적은 문서"였으므로, 규모를 키워 의미있는 비교를 시도

---

## 2. 실험 규모

| 항목 | 계획 | 실제 | 이유 |
|------|------|------|------|
| Queries | 500개 | **255개** | collection_1m_fair.tsv에 정답 passage가 있는 query가 6,980개 중 255개뿐 |
| Documents | 50,000개 | **50,000개** | 정답 254개 필수 포함 + 랜덤 49,746개 |
| Centroids | 32,768개 | **32,768개** | 16×√3.4M 권장값 (2^15) |
| nprobe | 2 | **2** | 논문 기본값 (ColBERTv2 paper) |
| Embedding dim | 128 | **128** | ColBERTv2 기본값 |

---

## 3. 데이터 준비 (Phase 1)

**Query 선정 조건**: `qrels.dev.small.tsv`에서 정답 passage가 `collection_1m_fair.tsv`에 존재하는 query만

```
전체 dev query: 6,980개
→ 정답 passage가 collection에 있는 query: 255개 (3.7%)
```

**Document Pool**:
- 정답 passage 254개 (1개 중복) 필수 포함
- 랜덤 49,746개 추가
- 총 50,000개, 토큰 임베딩 3,378,234개

**Query Embedding 생성**:
- `colbert-ir/colbertv2.0` 모델 (HuggingFace)
- 출력: 255 × 32 × 128 벡터, L2-norm=1.0

---

## 4. 인덱스 구조 (Phase 2)

**Centroid 구축 (k-means, FAISS GPU)**:
- 입력: 3.38M 토큰 벡터 × 128차원
- 결과: 32,768개 centroid, 20 iterations, ~35초

**IVF 구축**:
- centroid당 평균 103.1 토큰 (~1.5 passages)
- 전체 32,768개 centroid 활성화
- ~70초 소요

---

## 5. 파이프라인 실행 (Phase 3)

**2 GPU 병렬 실행 전략**:
```
GPU0: digital(255 queries) → analog(q128~254)
GPU1: analog(q0~127)
총 소요: ~13분
```

**4가지 방법**:

| 방법 | Step 2 | Step 6 | doc 벡터 |
|------|--------|--------|---------|
| dig_f32 | dot product argmax | MaxSim | C_t + r_float16 |
| dig_2bt | dot product argmax | MaxSim | C_t + r_2bit |
| ana_f32 | IDS 전류 argmin | MinCurrent | C_t + r_float16 |
| ana_2bt | IDS 전류 argmin | MinCurrent | C_t + r_2bit |

**2bit 양자화 방식**:
- 32K centroid 기준 residual 계산
- 분포 기반 4-bucket: `[-0.147, -0.063, -0.002, +0.088]`

---

## 6. 결과 (Phase 4)

### [6-1] 전체 지표 (N=255, nprobe=2)

| 지표 | dig_f32 | dig_2bt | ana_f32 | ana_2bt |
|------|---------|---------|---------|---------|
| MRR@10 | 0.0 | 0.0 | 0.0 | 0.0 |
| Success@1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Success@5 | 0.0% | 0.0% | 0.0% | 0.0% |
| Success@10 | 0.0% | 0.0% | 0.0% | 0.0% |
| Success@50 | 0.0% | 0.0% | 0.0% | 0.0% |
| R@50 | 0.0% | 0.0% | 0.0% | 0.0% |
| **R@1000** | **2.75%** | **2.75%** | **2.75%** | **2.35%** |
| nDCG@10 | 0.0 | 0.0 | 0.0 | 0.0 |
| 후보 내 정답 | 7/255 | 7/255 | 8/255 | 8/255 |

### [6-2] Digital vs Analog 차이

| | 수 | 비율 |
|-|----|------|
| 둘 다 정답 찾음 | 7 | 2.7% |
| **Analog만 찾음** | **1** | **0.4%** |
| Digital만 찾음 | 0 | 0.0% |
| 둘 다 못 찾음 | 247 | 96.9% |

- Analog만 찾은 쿼리: "how long after pap smear will i receive results" (rank 427)

---

## 7. 핵심 발견: nprobe=2 스케일 문제

```
소규모 (100 centroids, 200 docs):
  nprobe=2 → 2개/100 = 2% centroid 커버 → 정상 작동

대규모 (32K centroids, 50K docs):
  nprobe=2 → 2개/32,768 = 0.006% centroid 커버 → 너무 낮음
```

**이론적 recall 분석:**

| nprobe | 이론 recall | 후보 수/query |
|--------|------------|-------------|
| 2 | ~8% | ~97개 |
| 8 | ~29% | ~390개 |
| 16 | ~50% | ~780개 |
| 32 | ~75% | ~1,561개 |
| 64 | ~94% | ~3,123개 |

**실제 recall**: nprobe=2에서 2.7~3.1% (이론보다 낮음)

**중요**: analog step2는 nprobe에 관계없이 32K centroid 전부 계산  
→ nprobe를 올려도 analog 연산 시간 변화 없음

---

## 8. 결론

nprobe=2에서의 비교는 **의미있는 결과를 도출하기 어려움**:
- 전체 recall이 너무 낮아(~3%) 대부분 쿼리에서 정답 미검색
- Digital/analog 차이 측정 불가 (비교 대상 쿼리가 7~8개뿐)

---

## 9. 다음 단계 제안

**Option A**: nprobe 증가 (권장)
- nprobe=32~64 → recall 75~94%
- analog step2 속도 변화 없음
- 의미있는 digital vs analog 비교 가능

**Option B**: centroid 수 감소
- 32K → 2,048개로 줄이면 nprobe=2가 0.1% 커버
- 재indexing 필요 (Phase 2 재실행)

---

## 10. 생성 파일 목록

| 파일 | 내용 |
|------|------|
| `phase1_data_prep.py` | Query 선정 + doc pool + embedding 생성 |
| `phase2_index_build.py` | Centroid k-means + IVF 구축 |
| `phase3_digital.py` | Digital pipeline (GPU0) |
| `phase3_analog.py` | Analog pipeline (GPU 지정, query range) |
| `phase3_gpu0.py` | GPU0 wrapper (digital → analog 순차 실행) |
| `phase4_metrics.py` | 전체 지표 + margin breakdown + 케이스 분석 |
| `check_collection.py` | collection pid→idx 매핑 구축 |
| `scale_query_embs_255x32x128.pt` | 255개 query embedding |
| `scale_query_meta.csv` | query 메타 (qid, text, true_pid, true_passage_idx) |
| `scale_doc_pool_50k.csv` | 50K passage index 목록 |
| `scale_centroids_32k.npy` | 32K centroid 벡터 |
| `scale_all_vectors.npy` | 3.38M 토큰 벡터 (50K docs) |
| `scale_assignments.npy` | 각 토큰의 centroid 할당 |
| `scale_ivf_pids/tokidxs.npy` | IVF 구조 |
| `phase3_combined.csv` | digital + analog 결과 합본 |
| `phase4_metrics.csv` | 전체 지표 테이블 |
| `phase4_margin_breakdown.csv` | margin 구간별 지표 |
| `collection_pid2idx.json` | MS MARCO pid → passage index 매핑 |
