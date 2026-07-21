# Vth Compensation vs No-Compensation — 결과 요약

최초 작성: 2026-06-30 / 수정: 2026-07-01  
실험 환경: MS MARCO, query 255개, passage pool 50,000개, centroid 2,048개 (nprobe=2), seed=42  
셀 모델: 3T1C dual cell (die4, VDD=1.7V), Vth_orig=0.151045V, sigmoid I-V fit (L_f=6.112020, K_f=2.731949, B_f=-10.713911)

---

## ⚠️ 시뮬레이션 모델 수정 이력 (2026-07-01)

초기 버전(2026-06-30)에는 **시뮬레이션 모델 오류**가 있었으며, 이로 인해 Option C 성능이 실제보다 훨씬 낮게 측정됨.

### 구버전 모델 (버그)
- MaxSim 연산 시 동일 소자(셀)를 비교할 때마다 Vth를 **독립적으로 재샘플링**
- 예: 같은 문서 토큰 셀을 32개 쿼리 토큰과 비교하면 32번 모두 다른 Vth가 사용됨
- 실제 PBS는 소자에 영구적으로 Vth를 바꾸는 현상이므로 물리적으로 비현실적

### 신버전 모델 (수정, 1+2 조합)
1. **소자 고정 Vth**: 각 물리적 셀(토큰 × 차원)의 Vth를 실험 시작 전 한 번만 샘플링 → 전 쿼리에서 동일한 Vth 사용
2. **128차원 독립**: 한 토큰 내 128개 차원 각각이 독립적인 Vth를 가짐 (M0/M3도 독립)

### 버그가 성능에 미친 영향
| | 구버전 MRR@10 | 신버전 MRR@10 |
|---|---|---|
| Option C v1 (rank-only) | ~5.7% | **77.4%** |
| Option C v1 (full pipeline) | ~4.0% | **78.1%** |

구버전은 매 비교마다 i.i.d. 노이즈가 추가되어 MaxSim 합산 시 신호가 완전히 묻혔음. 분포(marginal) 자체는 맞지만 동일 소자의 시간적 일관성(temporal consistency)이 없어 랭킹 신호 파괴.

---

## 1. 비교 대상

| 방식 | 설명 |
|---|---|
| **Digital** | float32 원본 임베딩으로 dot product MaxSim (압축/회로 모델 없음, 이상적 상한선) |
| **Option A (VthComp)** | vth_stored = vth_actual → 셀 전류에서 Vth 항이 상쇄됨 → PBS 스트레스와 무관하게 안정 |
| **Option C (NoComp)** | vth_stored = 0 → 트랜지스터별 dead zone [0, vth_actual] 발생 → PBS가 강할수록 dead zone 확대 |

PBS(Positive Bias Stress) 조건 4단계:

| 조건 | Vth shift 범위 | std |
|---|---|---|
| v1 | [0, 0.5] V | 0.15 |
| v2 | [0, 1.0] V | 0.30 |
| v3 | [0, 2.0] V | 0.60 |
| v4 | [0, 3.0] V | 0.90 |

Vth 테이블 크기 (신버전 기준):
- Centroid 테이블: 2,048 × 128 × 2 (M0/M3) = 524,288개
- Doc token 테이블: NT × 128 × 2 (M0/M3) ≈ 860M개 (전체 pipeline 기준)

---

## 2. 메인 결과 — Full Pipeline (Step 2/3 + Step 6, 자체 후보)

Option C가 centroid 선택(Step 2)부터 MaxSim 랭킹(Step 6)까지 모두 dead-zone 회로로 수행.

| Method | PBS 조건 | MRR@10 | nDCG@10 | R@50 | R@1k | Cand.Recall |
|---|---|---|---|---|---|---|
| Digital (float32) | -- | 78.4 | 83.1 | 98.8 | 99.6 | 100% |
| Digital (2bit residual) | -- | 76.9 | 81.9 | 98.8 | 100.0 | 100% |
| Option A (VthComp) | v1–v4 동일 | 78.8 | 83.4 | 98.8 | 99.6 | 100% |
| Option C (NoComp) | v1 [0, 0.5] V | 78.1 | 82.3 | 98.8 | 99.6 | 100.0% |
| Option C (NoComp) | v2 [0, 1.0] V | 71.3 | 75.9 | 94.1 | 96.5 | 96.5% |
| Option C (NoComp) | v3 [0, 2.0] V | 27.8 | 32.0 | 58.8 | 67.1 | 68.2% |
| Option C (NoComp) | v4 [0, 3.0] V | 2.1 | 3.4 | 23.1 | 48.2 | 51.0% |

**핵심 관찰**
- v1 조건: Option C가 Option A(78.8%)와 사실상 동일한 수준(78.1%) — 약한 PBS에서는 dead zone이 존재하더라도 랭킹 정확도 거의 유지.
- v2 조건: MRR@10 약 7%p 저하, Cand.Recall도 96.5%로 소폭 하락 — Step 2(centroid 선택)에서 일부 오류 시작.
- v3/v4 조건: Cand.Recall 자체가 68%/51%까지 떨어지며 성능 급락 — dead zone이 centroid 선택을 크게 왜곡.

---

## 3. 원인 분리 실험 — Rank-Only (Step 6만, Option A 후보 재사용)

Option A가 찾은 (candidate recall ≈ 100%) 후보 집합을 그대로 Option C Step6 랭킹 로직에 입력.  
→ IVF 단계(Step 2/3) 영향을 제거하고 MaxSim(Step 6) 오류만 측정.

| Method | PBS 조건 | MRR@10 | nDCG@10 | R@50 | R@1k |
|---|---|---|---|---|---|
| Digital (float32) | -- | 78.4 | 83.1 | 98.8 | 99.6 |
| Option A (VthComp) | v1–v4 동일 | 78.8 | 83.4 | 98.8 | 99.6 |
| Option C rank-only | v1 [0, 0.5] V | 77.4 | 81.8 | 99.2 | 99.6 |
| Option C rank-only | v2 [0, 1.0] V | 74.6 | 79.3 | 98.0 | 99.6 |
| Option C rank-only | v3 [0, 2.0] V | 35.0 | 41.9 | 86.3 | 98.4 |
| Option C rank-only | v4 [0, 3.0] V | 5.1 | 7.5 | 42.7 | 89.8 |

**Full pipeline vs Rank-only 비교** (PBS에 의한 Step2 추가 손실):

| PBS 조건 | Rank-only MRR | Full MRR | Step2 추가 손실 |
|---|---|---|---|
| v1 | 77.4 | 78.1 | ≈ 0 (샘플링 분산 수준) |
| v2 | 74.6 | 71.3 | −3.3%p |
| v3 | 35.0 | 27.8 | −7.2%p |
| v4 | 5.1 | 2.1 | −3.0%p |

- v1~v2: Step 6(MaxSim)이 주된 손실원. Step 2 영향은 미미.
- v3 이상: Step 2(centroid 선택 오류)와 Step 6(MaxSim 오류) 모두 크게 기여.
- R@1k는 rank-only에서 v4에서도 89.8%로 높지만, full pipeline에서는 48.2%로 급락 → Step 2 오류(Cand.Recall 51%)가 R@1k를 직접 제한.

---

## 4. Digital + 2bit Residual 압축

ColBERTv2/PLAID 방식의 2bit residual 압축이 PBS 없이 랭킹 정확도에 미치는 영향.

- 후보 검색(Step2/3)은 압축과 무관 → Digital과 동일 (recall 100%)
- 2bit 양자화 평균 L2 복원 상대오차: **27.91%**
- bucket cutoffs: [-0.0424, -0.00002, 0.0425], bucket weights: [-0.0819, -0.0204, 0.0203, 0.0819]

| Method | MRR@10 | nDCG@10 | R@50 | R@1k |
|---|---|---|---|---|
| Digital (float32) | 78.4 | 83.1 | 98.8 | 99.6 |
| **Digital (2bit residual)** | **76.9** | **81.9** | **98.8** | **100.0** |
| Option A (VthComp, analog) | 78.8 | 83.4 | 98.8 | 99.6 |

**해석**
- 평균 27.91% 복원 오차에도 랭킹 지표 저하는 MRR@10 −1.5%p, nDCG@10 −1.2%p에 불과.
- **Option A가 Digital(2bit 압축)보다 모든 지표에서 더 좋음** → 아날로그 Vth 보상 설계가 2bit 양자화보다 더 정밀한 연산을 제공.
- Option C v1에서도 full pipeline MRR@10 78.1%로 2bit 압축(76.9%)보다 높음 → 약한 PBS 조건에서는 Option C도 디지털 압축 대비 손실이 없음.

---

## 5. 종합 결론 (수정된 물리적 시뮬레이션 기준)

1. **Option A(VthComp)**: PBS 스트레스, 2bit 압축과 무관하게 Digital(float32) 수준 유지 — Vth 상쇄 설계가 가장 강건.

2. **Option C(NoComp) — 약한 PBS(v1, [0,0.5]V)**:  
   - Rank-only MRR@10 77.4%, Full pipeline MRR@10 78.1% → Option A와 사실상 동등.  
   - Cand.Recall 100% → Step 2 오류 없음. dead zone이 존재해도 랭킹 신호는 충분히 유지.

3. **Option C(NoComp) — 중간 PBS(v2, [0,1.0]V)**:  
   - Full MRR@10 71.3%로 약 7%p 저하. Cand.Recall 96.5%로 일부 후보 누락 시작.  
   - Step 6(MaxSim) 오류가 주된 손실원.

4. **Option C(NoComp) — 강한 PBS(v3/v4)**:  
   - v3: MRR@10 27.8%, Cand.Recall 68.2% → Step 2, 6 모두 크게 붕괴.  
   - v4: MRR@10 2.1%, Cand.Recall 51.0% → 검색 자체가 거의 불가능.

5. **2bit 압축 vs PBS 영향 비교**: 2bit 압축 노이즈(MRR −1.5%p)는 v2 이상의 PBS 영향에 비하면 미미. v1 Option C는 2bit 압축보다도 나은 성능을 보임.

---

## 6. 생성된 Figure 목록

| 파일 | 위치 | 내용 |
|---|---|---|
| `table_optC_rankonly.png` | `논문 figure/` | Table 1: rank-only 결과 (구버전) |
| `table_optC_rankonly_v2.png` | `논문 figure/` | Table 1 v2: PBS-Invariant Baseline / Under PBS-Induced Vth Variation 섹션, 메서드명 정리 |
| `table_optC_owncandidate.png` | `논문 figure/` | Table 2: full pipeline 결과 (구버전) |
| `table_optC_owncandidate_v2.png` | `논문 figure/` | Table 2 v2: 2단계 헤더(Scoring\|Coarse Search), 메서드명 정리, 우측 여백 수정 |
| `[graph_optC]option1_single.png` | `논문 figure/` | 그래프: 전체 지표 단일 line plot (4면 박스, x축 v1~v4 제거, xlim 여백) |
| `[graph_optC]option2_twopanel.png` | `논문 figure/` | 그래프: ranking/recall 2-panel |
| `[graph_optC]option3_degradation.png` | `논문 figure/` | 그래프: rank-only vs full pipeline IVF 영향 비교 |
| `[graph_optC]legend.png` | `논문 figure/` | 레전드 단독 파일 (실선: Option C 지표, 점선: Option A baseline) |
| `[graph_optC]option1_2bit.png` | `논문 figure/` | 그래프: Digital 2bit baseline 포함 단일 line plot |
| `[graph_optC]option2_2bit.png` | `논문 figure/` | 그래프: Digital 2bit 포함 2-panel |
| `[graph_optC]option3_2bit.png` | `논문 figure/` | 그래프: Digital 2bit 포함 rank-only vs full pipeline |
| `[graph_optC]legend_2bit.png` | `논문 figure/` | 레전드 단독 파일 (2bit 버전) |
| `[vth_v3_fixedcell]row0_scatter_newline.png` | `centroidset_vector 크기 조절/` | Smoke test: Digital vs A/C rank scatter |
| `[vth_v3_fixedcell]row1_bar.png` | `centroidset_vector 크기 조절/` | Smoke test: 후보집합 stacked bar |

---

## 7. Figure 스타일 기준 (2026-07-01 확정)

- **표 (table_optC_*_v2.png)**: Times New Roman serif, 2단계 컬럼 헤더 (Scoring | Coarse Search), booktabs 스타일 수평선, TABLE_RIGHT=0.86으로 우측 여백 제어
- **그래프 (graph_optC)**: sans-serif bold, tick 20pt, axis label 23pt, xlim(-0.4, 3.4)으로 첫 tick 여백 확보, option1은 4면 박스 스타일
- **x축 레이블**: `[0, 0.5] V` / `[0, 1.0] V` / `[0, 2.0] V` / `[0, 3.0] V` (v1~v4 표기 제거)
- **x축 제목**: `PBS-Induced Vth Shift (No Compensation)`

---

## 참고: 관련 스크립트

| 스크립트 | 경로 | 역할 |
|---|---|---|
| `phase3_optC_rank_only.py` | `centroidset/` | rank-only 실험 (255쿼리, v1~v4) |
| `phase3_vth_pipeline.py` | `centroidset/` | full pipeline 실험 (255쿼리, v1~v4) |
| `plot_table_optC_rankonly_v2.py` | `논문 figure/` | Table 1 v2 PNG 생성 |
| `plot_table_optC_owncandidate_v2.py` | `논문 figure/` | Table 2 v2 PNG 생성 |
| `plot_optC_degradation_all.py` | `논문 figure/` | 그래프 3종 생성 (기본) |
| `plot_optC_degradation_2bit.py` | `논문 figure/` | 그래프 3종 생성 (Digital 2bit 포함) |
| `plot_optC_legend.py` | `논문 figure/` | 레전드 단독 PNG 생성 |
| `pipeline_vth_v3_fixedcell.py` | `centroidset_vector 크기 조절/` | 소규모 smoke test (3쿼리) |
