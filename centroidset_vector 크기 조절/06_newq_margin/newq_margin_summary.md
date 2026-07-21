# newq_margin 계열 실험 요약 — margin 기반 3-query, No Comp vs Vth Comp 분석

## 1. 배경 및 목표

기존 `[clip99.9]`/`[vth_v3_fixedcell]` 3-query(q0/q1/q2) 스터디에서는 no-comp(Option C)도 항상 true passage를
rank 1로 잡는 문제가 있었음. 원인은 query와 true passage 간 maxsim 절댓값만으로 query를 선정했기 때문에,
실제 "2등 경쟁 문서"와의 margin(격차)을 고려하지 않았던 것.

이를 개선하기 위해 **"1등(true passage) vs 2등(경쟁 문서) MaxSim margin"** 기준으로 새 3-query를 선정하고,
경쟁 문서를 얼마나 pool에 포함시키느냐에 따라 Digital/Vth Comp(Option A)/No Comp(Option C) 순위가 어떻게
달라지는지 단계적으로 실험함.

## 2. 확정된 3-query (margin 기준)

254개 MS MARCO 후보 query 중, 500개 공유 distractor pool 대비 margin이 최소/중간/최대인 query를 선정:

| q_id | qid | query text | true_pid | margin (1등-2등 maxsim 차) |
|---|---|---|---|---|
| q0 | 329114 | "how much to have your house reshingled" | 471602 | **1.62** (경쟁 치열) |
| q1 | 498398 | "simple definition wave amplitude" | 822108 | **11.35** (중간) |
| q2 | 984178 | "where is hartwell ga" | 347885 | **20.06** (압도적 1등) |

margin = maxsim(query, true_pid) − max(maxsim(query, distractor) for distractor in 500개 공유 pool)

## 3. 데이터 파이프라인 공통 구조

각 실험 폴더는 독립적으로 다음을 포함:
- `[original]`/`[clip99.9]` query_embs(96×128) / centroids(100×128) / doc_embs(N×128) — raw + 99.9 percentile clip+scale, Excel
- `[clip99.9]ivf_centroid2pid.xlsx`, `query_centroid_ranking.xlsx`
- `[vth_v3_fixedcell]step2/3/6_*.xlsx` — Vth mismatch pipeline 결과
- `[vth_v3_fixedcell]row0_scatter_newline.png`, `row1_bar.png`

**Vth 모델** (전 실험 공통, 변경 없음): `Vth = 0.151045V + TruncGaussian(std=0.90V)`, range `[0,3]V` shift,
물리 셀(centroid/문서토큰 × 128dim) 단위로 고정 샘플링 후 모든 비교에서 재사용.
Option A(Vth Comp)는 M0/M3 게이트-소스가 매칭되어 Vth mismatch가 구조적으로 상쇄되고,
Option C(No Comp)는 상쇄 없이 그대로 반영됨.

## 4. 실험별 구성 및 결과

| 폴더 | Pool 크기 | 경쟁문서 포함 방식 |
|---|---|---|
| `newq_margin` | 200 | q0~q2 각각 rank2 경쟁문서 1개씩 강제 포함 |
| `newq_margin_top10` | 200 | q0~q2 각각 top10(총 30개) 경쟁문서 강제 포함 |
| `newq_margin_500` | 500 | q0의 top50 경쟁문서만 강제 포함 |
| `newq_margin_500_top100` | 500 | q0의 top100 경쟁문서 강제 포함 |
| `newq_margin_1000` | 1000 | q0의 top100 경쟁문서 강제 포함 |

### Rank 결과 (Digital / Vth Comp(A) / No Comp(C), true_pid 기준)

**newq_margin (200, rank2만)**

| q_id | Digital | Vth Comp | No Comp |
|---|---|---|---|
| q0 (margin 1.62) | 1/54 | 1/167 | **3/194** |
| q1 (margin 11.35) | 1/30 | 1/155 | 1/178 |
| q2 (margin 20.06) | 1/134 | 1/184 | 1/185 |

**newq_margin_top10 (200, top10×3)**

| q_id | Digital | Vth Comp | No Comp |
|---|---|---|---|
| q0 | 1/64 | 1/168 | 1/181 |
| q1 | 1/28 | 1/173 | 1/184 |
| q2 | 1/82 | 1/146 | **후보 자체에서 탈락** |

**newq_margin_500 (500, q0 top50)**

| q_id | Digital | Vth Comp | No Comp |
|---|---|---|---|
| q0 | 1/416 | 1/442 | **2/488** |
| q1 | 1/84 | 1/453 | 1/445 |
| q2 | 1/211 | 1/462 | 1/486 |

**newq_margin_500_top100 (500, q0 top100)**

| q_id | Digital | Vth Comp | No Comp |
|---|---|---|---|
| q0 | **2/420** | **2/489** | **2/476** |
| q1 | 1/250 | 1/404 | 1/443 |
| q2 | 1/307 | 1/443 | 1/470 |

**newq_margin_1000 (1000, q0 top100)**

| q_id | Digital | Vth Comp | No Comp |
|---|---|---|---|
| q0 | **2/628** | **2/966** | **2/982** |
| q1 | 1/228 | 1/870 | 1/924 |
| q2 | 1/837 | 1/889 | 1/876 |

## 5. 핵심 발견

1. **경쟁문서 밀도가 낮을 때(rank2 1개)**: No Comp만 rank 3으로 밀리고 Digital/Vth Comp는 rank 1 유지
   → Vth mismatch 보상 유무의 순수한 효과가 가장 뚜렷하게 드러남.
2. **경쟁문서를 top10, top50으로 늘리면**: No Comp가 더 크게 흔들리거나(q2 후보 탈락) 여전히 A/C 간 격차가 관찰됨.
3. **top100까지 촘촘히 채우면(500/1000 docs)**: Digital 자체가 rank 2로 흔들림 — 이 지점부터는
   "Vth mismatch로 인한 오류"가 아니라 **"진짜 근소한 차이라 99.9-percentile clip 같은 전처리 수준의
   미세한 변화로도 순위가 갈리는 영역"**에 들어간 것으로 해석됨. Digital/A/C 모두 동일하게 rank 2를 유지.
4. **pool을 500→1000으로 늘려도 q0는 여전히 rank 2 고정**: 이미 500개일 때 "진짜 2등"을 찾아냈고,
   1000개로 늘려도 더 강한 경쟁자가 새로 나타나지 않음.
5. q1(margin 11.35), q2(margin 20.06)는 별도로 근접 경쟁문서를 강화하지 않아 pool 크기와 무관하게
   대체로 rank 1을 유지함 — margin이 작을수록, 그리고 그 근접 경쟁자가 실제로 pool에 있을수록 취약해짐.

## 6. 진행 중 — Vth 랜덤시드 스윕 (`newq_margin/seed_sweep_optC.py`)

`newq_margin`(200 docs, rank2 1개 포함) pool을 고정한 채, Vth 샘플링 SEED만 0~39로 40회 바꿔가며
Option C(No Comp)의 true passage rank 분포를 측정 중. Digital/Option A는 Vth 랜덤과 무관하므로 1회만 계산.
목적: pool 구성을 바꾸지 않고 "노이즈 운"만으로 A vs C 간격이 얼마나 벌어질 수 있는지 통계적으로 확인.
결과는 `newq_margin/seed_sweep_optC_results.csv`에 저장 예정 (완료 후 본 문서에 결과 추가 필요).
