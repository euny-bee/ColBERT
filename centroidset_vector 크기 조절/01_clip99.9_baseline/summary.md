# Vector 크기 조절 및 아날로그 파이프라인 요약

## 1. 배경 및 목표

ColBERT MaxSim 연산을 아날로그 소자로 재현하는 실험에서,  
기존 임베딩 벡터 값의 범위가 **-0.3 ~ 0.3** 수준으로 작아 아날로그 회로의 동적 범위를 충분히 활용하지 못하는 문제가 있었음.  
이를 **-1 ~ 1** 범위로 확장하면서 ranking 순서 및 centroid-doc 관계를 최대한 보존하는 것이 목표.

---

## 2. 원본 파일 구조

| 파일 | Shape | 내용 |
|------|-------|------|
| `centroids_100x128.xlsx` | (100, 128) | k-means centroid 벡터 |
| `doc_embs_12919x128.xlsx` | (12919, 129) | doc 토큰 임베딩 + `is_relevant` 컬럼 |
| `query_embs_96x128.xlsx` | (96, 128) | query 토큰 임베딩 (3 queries × 32 tokens) |
| `ivf_centroid2pid.xlsx` | (1773, 2) | centroid → pid 매핑 (IVF 인덱스) |
| `query_centroid_ranking.xlsx` | (96, 101) | query 토큰 × centroid dot product 점수 |

**원본 값 범위:**
- 모든 벡터: L2 norm = **1.0** (unit vector)
- centroids max_abs: 0.3402
- doc_embs dim max_abs: 0.4260 (이상치 1개)
- query_embs max_abs: 0.3396

---

## 3. 스케일링 방법 비교

### 핵심 원칙
- centroid는 doc_embs에서 k-means로 만든 것이므로 **동일한 scale factor** 적용 필요
- query와 doc에도 동일 factor 적용 → dot product가 k² 배 → ranking 보존

### 검토한 방법

| 방법 | Scale Factor | doc ±1 | centroid ±1 | query ±1 | Ranking 보존 |
|------|------------|--------|-----------|--------|------------|
| Global scale | 1/max_abs(doc) = 2.347 | O | ±0.80 | ±0.80 | 완벽 |
| Clip (max_abs centroid 기준) | 1/0.3402 = 2.940 | O (117개 clip) | O | O | 토큰 1개 재배정 (pid 레벨 동일) |
| **Clip99** | 1/global_99th = 4.423 | O | O | O | top-5 중 14개 차이 |
| **Clip99.9** | 1/global_99.9th = 3.483 | O | O | O | top-5 중 1개 차이 |

### 각 방법의 clip 비율

| 방법 | Threshold | doc clip 비율 | centroid clip 비율 | query clip 비율 |
|------|----------|------------|-----------------|--------------|
| Global | 0.4260 | 0% | 0% | 0% |
| Clip | 0.3402 | 0.0071% | 0% | 0% |
| Clip99 | 0.2261 | 1.00% | 1.09% | 1.03% |
| Clip99.9 | 0.2871 | 0.10% | 0.16% | 0.11% |

---

## 4. 최종 선택: `[clip99.9]`

**선택 이유:**
- 세 파일 모두 ±1.0 범위 달성
- original 대비 top-2 centroid 선택 완전 일치 (top-5도 95/96 일치)
- ivf_centroid2pid pid 레벨 차이: 22쌍 (원본 1773쌍 대비 1.2%)
- ranking 보존 수준이 가장 우수한 clip 버전

**Scale Factor:** `1 / 0.2871 = 3.483`  
**Global 99.9th percentile** (doc_dims + centroids + query 전체 기준)

---

## 5. 생성된 파일 구조

```
centroidset_vector 크기 조절/
│
├── [clip99.9]centroids_100x128.xlsx          ← 최종 사용
├── [clip99.9]doc_embs_12919x128.xlsx         ← 최종 사용
├── [clip99.9]query_embs_96x128.xlsx          ← 최종 사용
├── [clip99.9]ivf_centroid2pid.xlsx           ← 탭: long_format, wide_format
├── [clip99.9]query_centroid_ranking.xlsx     ← 탭: scores, ranks, rank_order
│
├── [clip99.9]step2_digital_centroid.xlsx     ← Step2 digital centroid 선택
├── [clip99.9]step2_optA_centroid.xlsx        ← Step2 Option A 전류 기반
├── [clip99.9]step2_optC_centroid.xlsx        ← Step2 Option C Vth mismatch
├── [clip99.9]step3_digital_candidate_pids.xlsx
├── [clip99.9]step3_optA_candidate_pids.xlsx
├── [clip99.9]step3_optC_candidate_pids.xlsx
├── [clip99.9]step6_all_results.xlsx          ← 최종 ranking 결과
├── [clip99.9]compare_ranking_all3.png        ← 비교 플롯
│
└── clip99.9외/                               ← 참조용 보관
    ├── [original]*.xlsx                      ← 원본 (unscaled)
    ├── [global]*.xlsx                        ← Global scale 버전
    ├── [clip]*.xlsx                          ← max_abs(centroid) 기준 clip
    └── [clip99]*.xlsx                        ← 99th percentile clip
```

---

## 6. 아날로그 파이프라인 결과 (clip99.9 기준)

### 회로 모델

| 옵션 | VGS | Vth |
|------|-----|-----|
| Option A | \|V_q - V_d\| + Vth | 고정 0.151045V |
| Option C | \|V_q - V_d\| | TruncGaussian(mean=0, std=0.15, range=[0,0.5]V), seed=42 |

### Step 2: Centroid 선택 (nprobe=2)
- Digital: `query_centroid_ranking` rank_order top-2
- Option A/C: 128차원 전류 합산 → MinCurrent top-2

### Step 3: 후보 pid 수집

| Query | Digital | Option A | Option C |
|-------|---------|---------|---------|
| q0 | 127개 | 127개 | 148개 |
| q1 | 70개 | 84개 | 88개 |
| q2 | 108개 | 85개 | 132개 |

### Step 6: 최종 Ranking (정답 pid 순위)

| Query | 정답 pid | Digital f32 | Option A | Option C |
|-------|---------|------------|---------|---------|
| q0: "where does real insulin come from" | 7264308 | **1**/127 | **1**/127 | **1**/148 |
| q1: "where does name nora come from" | 7264266 | **1**/70 | **1**/84 | **1**/88 |
| q2: "where does most of the iron ore come from" | 7264253 | **1**/108 | **1**/85 | **1**/132 |

**세 가지 방법 모두 3개 쿼리에서 정답을 rank 1로 찾음.**

---

## 7. 스크립트 목록

| 파일 | 역할 |
|------|------|
| `scale_and_rebuild.py` | Global / Clip 두 버전 임베딩 생성 및 ivf, qcr 재계산 |
| `scale_percentile.py` | Clip99 / Clip99.9 버전 생성 및 original 비교 |
| `rebuild_sheets.py` | [clip99.9] ivf, qcr 파일에 멀티시트 추가 |
| `pipeline_clip99.9.py` | Step 2, 3, 6 전체 파이프라인 (clip99.9 기준) |
| `compare_ranking_clip99.9.py` | 3-way ranking 비교 플롯 생성 |
| `compare_all_rank.py` | 4개 버전 top-5 rank_order 비교 |
| `plot_dist_99.9.py` | [clip99.9] 값 분포 히스토그램 |
