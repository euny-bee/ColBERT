# TOPS/W 분석 요약 (analog AIMDC vs GPU)

작성일: 2026-07-09
작업 폴더: `ColBERT\TOPSW\` (에너지/GPU 계산 스크립트, 최종 표 이미지 전부 이 폴더로 이동됨).
Xyce 회로 넷리스트(.cir/.sub)와 device-physics 검증 플롯은 device 특성 분석 작업 전반에 공용으로 쓰여서
원래 위치(`xyce_ColBERT용소자\260505공정 결과\260505공정 결과\`)에 그대로 있음.

---

## 1. 목표

ColBERT 스타일 검색(coarse search + MaxSim scoring)을 수행하는 analog in-memory distance computing(AIMDC) 6T2C(dual 3T1C) 셀의 실제 에너지를, **Xyce SPICE 트랜지언트 시뮬레이션**으로 구해서 GPU(RTX 3070Ti/3080Ti 실측, A100/H100 추정)와 TOPS/W(에너지 효율) 비교.

원칙: 이론적 FLOP 추정이 아니라 **실측/실제 물리 시뮬레이션** 기반으로 계산 (GPU는 nvidia-smi 실측, 셀은 Xyce 실측).

---

## 2. 회로/에너지 방법론

### 2-1. 3단계 동작
- **Phase1 (precharge)**: SN을 VML=0V로 리셋. Vth 무관, ~수십ns, 에너지 무시 가능(0.0072fJ).
- **Phase2 (store, write)**: bitline에 ∓V1(=1.0V) 인가해서 Vth를 보상하며 문서 값 저장. Option A만 수행 (Vth compensation), Option C는 스킵.
- **Phase3 (search)**: 커플링 캐패시터로 쿼리 값 V2를 주입해서 M0/M3 전류로 유사도 판정. Option A/C 둘 다 수행.

### 2-2. 핵심 발견 -- write가 이상식(-V1+Vth)과 다른 이유
- 실제 IGZO 소자는 **hard-threshold가 아니라 smooth logistic I-V**(SS=240mV/decade) — Vgs=Vth에서도 이미 22nA 흐름.
- M0가 자기참조(diode-connected 유사) 루프라, SN=0V(Phase1 리셋값)에서 시작하면 **Vth가 클수록 subthreshold 병목**에 걸려 목표 전압까지 못 감.
- 실측: Vth=0.03V→SN=-0.9987V(거의 완전 보상) / Vth=3.03V→SN=-0.0098V(거의 보상 안 됨).
- **Vth=3.03V만 이중안정성(bistable)** 확인 (SN=0V 시작→-0.0098V, SN=4V 시작→+1.6831V로 분기). 실제 칩은 항상 0V에서 시작하므로 이 bistable 분기는 실사용과 무관.
- Square-law(이상적 MOSFET) 모델로 재현 시도 → 수치적으로 불안정, 결론 보류.
- Sharptail(Vgs=Vth에서 전류를 1e-14A로 낮춘) 모델 → 일부 Vth에서 이상식에 더 가까워지지만, **SN=0V 시작(실제 조건)에서는 여전히 대부분 갇힘** → subthreshold 개선만으로는 근본 해결 안 됨, V1을 높이거나 다른 reset 방식이 필요.

### 2-3. Write 에너지
- 방법: 전체 Vth 공통 고정 write-time(worst-case Vth=1.03V 기준 settling×1.2 = 3200.1us) 사용 — 실제 칩은 셀마다 Vth를 미리 모르므로.
- PBS(Positive Bias Stress) Vth 변이 분포(TruncNorm)로 가중평균:
  - v2 [0,1]V: 1,379.6 fJ/cell
  - v3 [0,2]V: 1,128.0 fJ/cell
  - **v4 [0,3]V (채택): 923.46 fJ/cell** (Option A). Option C는 Phase2 자체가 없어 0.0072fJ/cell (Phase1만).
- Vth 낮을수록 write 에너지가 **더 큼** (직관과 반대) — Vth 낮으면 M1이 잘 켜져서 SN이 목표(-1V 근처)까지 크게 움직여 전하를 많이 옮기기 때문. Vth 높으면 subthreshold에 갇혀 거의 안 움직여 에너지도 작음.

### 2-4. Search 에너지
- 방법: 1us sensing window(전류가 거의 즉시 정상상태 도달 후 유지되는 성질 확인 — 즉 에너지=P×T_sense로 단순화됨).
- **버그 발견/수정**: 초기 diff별 에너지가 비단조적으로 나왔는데, 원인은 Xyce 적응형 타임스텝이 diff마다 다른 시점(0.53~0.96us)에서 샘플을 끊어서 생긴 아티팩트. **스냅샷(0.5us 지점) 전류 × 고정전압 × T_sense로 재계산**하여 해결 → 매끈하고 단조적인 결과로 수정됨.
- q2 실제 (Q-C) diff 분포(1,228,800 샘플)로 가중평균:
  - **Option A: 656.90 fJ/eval, Option C: 607.78 fJ/eval**

---

## 3. 스케일 (두 가지)

| | 코드명 | Coarse centroid | 후보(candidate) 수 | 비고 |
|---|---|---|---|---|
| (i) 프로토타입 | seed19 | 100 | A=184, C=193 | Xyce 소자 검증에 쓴 소규모 subset |
| (ii) 시스템레벨 | scaleup(2048) | 2,048 | A=10,988(avg), C(v4)=10,292(avg) | **Table 2(retrieval quality) 실제 스케일과 일치** |

- 32,768-centroid 시도(`scaleup_summary.md`)는 nprobe=2로 recall이 ~3%까지 붕괴해서 폐기됨 → 실제 Table 2는 2,048-centroid, nprobe=2로 만들어진 것 확인 (`vth_comp_vs_nocomp_summary.md`).
- 시스템레벨 후보 수는 `phase3_vth_pipeline.py` 실행 결과(`phase3_vth_results.csv`)에서 직접 추출.

### 연산 횟수 (쿼리 1개 = 32 토큰)
- Coarse search: 32 × N_centroid
- Scoring: 32 × N_candidate
- Option A/C 코어 셀 에너지/쿼리 = (coarse_ops + scoring_ops) × E_search_per_eval

---

## 4. GPU 실측/추정

### 4-1. 실측 (RTX 3070Ti, RTX 3080Ti, nvidia-smi 폴링 + PyTorch)
- **batch=1(비배치)은 CUDA 커널 launch 오버헤드가 지배적**이라 GPU가 인위적으로 나쁘게 나옴 (mJ 단위) — 표에서 제외.
- **batch=256 채택**: 여러 쿼리를 하나의 커널 호출로 묶어 오버헤드 상각. 실제 서비스의 표준적인 처리 방식과 일치.
- batch=1/32/256 모두 측정해서 Table 5로 정리 (배치 커질수록 GPU 에너지 급감, analog 대비 격차도 같이 줄어듦).
- **단위 버그 발견/수정**: 최초 표에 GPU 에너지를 μJ로 잘못 표기(실제는 mJ, 1000배 차이) → 정정.
- A100/H100은 실측 불가(장비 없음) → **TDP 비율 기반 추정치**로 명확히 라벨링 (RTX3070Ti 실측 utilization 비율 × 해당 GPU TDP, 메모리대역폭 비로 latency 추정).

### 4-2. 핵심 수치 (batch=256, Option A 기준, "Cell만")
| | 프로토타입 | 시스템레벨 |
|---|---|---|
| Analog (cell only) | 5.97 nJ | 274.03 nJ |
| RTX 3070Ti (실측) | 200.6 μJ (33,602×) | 3,949.7 μJ (14,414×) |
| A100 (추정) | 82.5 μJ (13,820×) | 1,624.4 μJ (5,928×) |
| H100 (추정) | 87.9 μJ (14,721×) | 1,730.2 μJ (6,314×) |

---

## 5. 주변회로(periphery) 에너지 -- 참고자료 기반 추가 분석

출처: `G:\내 드라이브\CEY\ColBERT\from공유\260708\energy_calcul_revised_v2.xlsx` (Revised_v2 시트, 알파벳 인식 ACU 참고 계산) — "Cell만 187,372× → 전체(주변회로 포함) 576×"로 격차가 크게 줄어드는 패턴을 그대로 우리 시스템에 적용.

### 5-1. 인용 파라미터
- Driver(WWL/WBL/SBL/RWL 공통): 0.45mW (그대로 인용)
- TIA: 9mW @ 5ns → **우리 sensing window(1us)에 맞춰 200배 낮춰 45uW로 스케일** (analog 회로는 대역폭-전력이 대략 비례한다는 설계 경향 근거)
- ML buffer: 42mW @ 5ns → **동일하게 200배 낮춰 210uW**
- ADC: 1.5pJ/conversion (8-bit, 45nm) — 그대로 인용

### 5-2. Write-phase driver 에너지 (one-time, 코어 테이블 기준)
Option A의 write-time(3200.1us, subthreshold 병목 때문에 매우 김) 때문에 driver 에너지가 코어 셀 에너지를 압도 (프로토타입 328.3μJ, 시스템레벨 3.13mJ) — 다만 이건 인덱스 빌드 1회성 비용.

### 5-3. Search 시나리오 누적 (batch=256, Table 4/4v2, 논문 figure 폴더)
| 시나리오 | 프로토타입 에너지 | vs GPU | 시스템레벨 에너지 | vs GPU |
|---|---|---|---|---|
| Cell only | 5.97 nJ | 33,602× | 274.03 nJ | 14,414× |
| + TIA | 414.9 nJ | 484× | 19,045.9 nJ | 207× |
| + ADC | 428.6 nJ | 468× | 19,671.6 nJ | 201× |
| + Driver | 6,361.4 nJ | 31.5× | 209,233.2 nJ | 18.9× |
| **+ ML buffer (최종)** | **8,269.8 nJ** | **24.3×** | **296,835.1 nJ** | **13.3×** |

**Driver가 주변회로 중 가장 큰 비중** (TIA/ML buffer는 200배 스케일 후 상대적으로 작음).

### 5-4. Batch size별 (Table 6)
| 시나리오 | P, batch=1 | P, batch=32 | P, batch=256 | S, batch=1 | S, batch=32 | S, batch=256 |
|---|---|---|---|---|---|---|
| Cell only | 402,776× | 57,526× | 33,602× | 42,965× | 15,946× | 14,414× |
| + ML buffer(최종) | 291× | 42× | 24.3× | 40× | 15× | 13.3× |

### 5-5. Cell+ADC+Driver만 (TIA/ML buffer 제외, batch별)
| | batch=1 | batch=32 | batch=256 |
|---|---|---|---|
| 프로토타입 | 404.0× | 57.7× | 33.7× |
| 시스템레벨 | 61.8× | 22.9× | 20.7× |

A100/H100(추정) 대비 (batch=256, Cell+ADC+Driver): 프로토타입 13.9~14.8×, 시스템레벨 8.5~9.1×.

**결론**: 주변회로를 현실적으로 다 넣어도(Driver 지배적) analog가 GPU 대비 **13~34배(RTX3070Ti)**, **5.5~15배(A100/H100 추정)** 유리 — Cell만 비교했을 때(수만 배)보다는 훨씬 보수적이지만 여전히 일관되게 유리.

---

## 6. 생성된 주요 파일

### TOPS/W 에너지/GPU 계산 스크립트 + 표 이미지 (`ColBERT\TOPSW\`, 전부 이 폴더로 이동됨)
- `calc_energy_phase12_v2.py`, `calc_energy_phase3_clean.py` — 최종 채택 에너지 계산 스크립트
- `gpu_measure.py`, `gpu_measure_batched.py`, `gpu_measure_2048*.py` — GPU 실측 스크립트 (+ 결과 csv)
- `periphery_energy_model_v2.py`, `periphery_scenario_table.py` — 주변회로 모델
- `energy_analysis_decision_log.md` — 세부 결정사항(축1~8) 로그
- `table_tops_w_energy.png` (Table 3) — Analog vs GPU, 두 스케일, batch=256
- `table_periphery_scenario.png` (Table 4, 원본 2열) / `table_periphery_scenario_v2.png` (Table 4v2, Latency/Power 추가)
- `table_batch_comparison.png` (Table 5), `table_periphery_batch_combined.png` (Table 6)

### Xyce 넷리스트 (원래 위치 그대로, `xyce_ColBERT용소자\260505공정 결과\260505공정 결과\`)
- `phase12_dual_die4.cir`, `phase123_dual_die4.cir`, `phase13_optionC_die4.cir` — write/search 트랜지언트
- `phase2_only_sn4V_die4.cir`, `igzo_die4_sharptail.sub`, `phase12_dual_squarelaw.cir` — 이상식 비교/bistability 진단

---

## 6-1. (추가, 2026-07-15) Cell+ADC+DAC (final) -- 논문 Fig.5c 채택 수치

섹션 5의 "+ML buffer(최종)"는 TIA/driver/ML buffer까지 전부 포함한 가장 보수적인 시나리오였음. 이후 별도로,
**ADC+DAC 두 변환 블록만** (TIA·driver·ML buffer 제외) 포함한 축소 시나리오를 따로 확정해서 논문
Fig.5c(`fig_adc_dac_gpu_vs_analog_bar.py`)에 최종 채택함. TIA/driver/ML buffer를 뺀 이유는 이 두 변환 블록이
analog 코어 주위에 반드시 필요한 최소 주변회로이기 때문 (자세한 근거는 `table_periphery_adc_dac.py` 참고).

### 시나리오 누적 (Option A, 시스템레벨 2,048 centroids, batch=256 GPU 기준)
| 시나리오 | Energy/query | vs. GPU | TOPS/W |
|---|---|---|---|
| Cell only | 274.03 nJ | 14,414× | 107.07 |
| + ADC | 899.76 nJ | 4,390× | (섹션 6-1 표 참고) |
| **+ DAC (final, 채택)** | **901.89 nJ** | **4,379×** | **118.41** |
| GPU (RTX 3070Ti, 실측) | 3,949,707.8 nJ | 1× | 0.0270 |

- ADC: 1.5 pJ/conversion, 8-bit 45nm (Andrulis et al., as-cited), N_rows_total × N_TOKENS(32) 이벤트
- DAC: 0.52 pJ/conversion, 8-bit (Hong & Lee 2007, 65 fJ/step×8, Saberi et al. TCAS-I 2011 경유 인용), N_COLS × N_TOKENS(32) 이벤트 — row 수와 무관해서 프로토타입/시스템레벨 두 스케일 모두 동일하게 +2.13 nJ
- TOPS/W = FLOPs/query ÷ Energy/query, FLOPs/query = N_rows_total × N_TOKENS × 256 (128-dim 내적 = mult 128 + add 128) — 시스템레벨 106,790,912 FLOPs/query. GPU도 동일 FLOPs 정의를 쓰므로 vs-GPU 비율은 nJ 기준과 TOPS/W 기준이 완전히 동일 (FLOPs가 비율에서 상쇄)
- **최종 결론: Analog가 GPU 대비 4,379× 더 에너지 효율적** (Cell+ADC+DAC 기준)

### 관련 파일
- `table_periphery_adc_dac.py` (Table 4-ADC/DAC) — nJ 버전, `table_periphery_adc_dac_topsw.py` — TOPS/W 버전
- `fig_adc_dac_gpu_vs_analog_bar.py` — 논문 Fig.5c용 바 차트 (Energy/query, Energy efficiency 두 버전 + simple x라벨 버전), 출력: `fig_adc_dac_energy_bar.png`, `fig_adc_dac_topsw_bar.png`, `fig_adc_dac_topsw_bar_simple.png`
- `table_periphery_adc_dac_combined.py` — 위 두 표(nJ/TOPS-W)를 하나로 합치고, Cell+ADC+DAC=Energy/query→TOPS/W 유도 도식을 상단에 추가한 supplement
- `fig_energy_derivation_schematic.py` — 위 combined의 유도 도식 부분만 표 없이 단독 figure로 분리
- `table_periphery_adc_dac_stacked.py` — Prototype/System-level을 좌우 대신 위아래로 쌓은 좁은 레이아웃 버전 (내용은 `table_periphery_adc_dac.py`와 동일)

---

## 9. (추가, 2026-07-21) Write 에너지 상각(amortization) 분석

섹션 6-1의 헤드라인(901.89nJ, 4,379×)은 **write(인덱스 빌드) 에너지를 제외한 search-only** 수치. "write까지 합치면 전체 수치가 어떻게 되는가"를 별도로 분석함.

### 9-1. 핵심 결론 — TOPS/W엔 보통 write를 안 넣는다
TOPS/W는 추론(search) 동작 지표이지 프로그래밍 지표가 아니라서, 학계/칩 논문(ISSCC 등 analog CIM)에서는 write/programming 에너지를 **TOPS/W에 섞지 않고 별도의 one-time 수치로 보고**하는 게 표준 관행. 이유: (1) write는 매 쿼리가 아니라 인덱스 빌드 시 1회만 발생, (2) 섞으려면 "몇 쿼리에 걸쳐 상각할지(N_queries_lifetime)" 가정이 필요한데 이게 논문마다 달라 비교 가능성이 깨짐. → **결론: 메인 Fig.5c 헤드라인은 write 제외 4,379×를 유지하고, write 상각 버전은 참고용 sensitivity 분석으로 별도 보관.**

### 9-2. Write 총량 (index build, 1회성, Option A)
periphery_energy_model_v2.py 기준, write는 **centroid(coarse table)만** 포함 (candidate 문서는 write 대상이 아님 — `N_cells = N_coarse × N_COLS`):
| | cell만 | + driver(WWL+WBL) | **합계** |
|---|---|---|---|
| Prototype (100 centroids) | 11.82 nJ | 328,330.26 nJ | **328,342.08 nJ** |
| System-level (2,048 centroids) | 242.08 nJ | 3,133,537.92 nJ | **3,133,780.00 nJ** |

driver(write-time 3200.1us 때문에 매우 김)가 write 비용의 **99.99%**를 차지 — cell 자체는 무시할 수준.

### 9-3. 상각(255 쿼리 가정)
- N_queries=255는 System-level Table 1 벤치마크의 실제 쿼리 수(`scale_query_embs_255x32x128.pt`)에서 가져옴
- Prototype scale의 실제 이력은 3-query(q0/q1/q2) subset뿐이지만, 이건 디바이스 검증용이지 배포 시나리오가 아니라서 **양쪽 스케일 모두 N_queries=255로 통일** (3으로 나누면 write가 압도적으로 커져 의미 없는 숫자가 됨 — 약 109,447 nJ/query)
- Write/query (amortized) = Write_total / 255:
  - Prototype: **1,287.62 nJ/query**
  - System-level: **12,289.33 nJ/query**

### 9-4. 상각 후 수치 (두 스코프)
| 스코프 | Prototype 최종(+DAC/+MLbuf) | System-level 최종 |
|---|---|---|
| 헤드라인(Cell+ADC+DAC)만 + write | 1,309.35 nJ, **153.2×**, 1.777 TOPS/W | 13,191.22 nJ, **299.4×**, 8.096 TOPS/W |
| 전체 periphery(+TIA+driver+MLbuf) + write (scope-consistent) | 9,557.46 nJ, **21.0×**, 0.2434 TOPS/W | 309,124.45 nJ, **12.8×**, 0.3454 TOPS/W |

"헤드라인만+write"는 search 쪽 driver가 빠져있어 write(driver 포함)와 scope가 안 맞음 — "전체 periphery+write" 버전이 apples-to-apples.

### 관련 파일
- `table_periphery_adc_dac_with_write.py` — 헤드라인 scope, 두 스케일
- `table_periphery_full_with_write.py` — 전체 periphery scope, 두 스케일 (scope-consistent)
- `table_adc_dac_with_write_systemlevel.py` — 헤드라인 scope, system-level만. Cell→+ADC→+DAC 누적 사슬과 write(1회성 flat 추가)를 시각적으로 분리한 레이아웃(초기 버전은 write가 매 행 반복돼 누적처럼 보이는 문제가 있어서 재설계함)

---

## 10. (추가, 2026-07-21) Node/Precision 정규화 캐비어트

리뷰 피드백: TOPS/W로 GPU와 비교할 때 **node(공정 노드) / precision(연산 정밀도) / scope(측정 범위)** 3가지를 명시해야 함. Scope는 섹션 5~9의 Cell→+ADC→+DAC→+periphery 단계별 표들이 이미 다루고 있음. Node/precision은 별도 확인 필요했음.

### 10-1. 확인 결과
- **Node**: ADC 45nm(Andrulis et al. 인용), DAC는 인용 논문 자체 공정, analog cell은 IGZO TFT(표준 CMOS 노드 아님), GPU는 Samsung 8nm — **전부 다른 공정, 정규화 안 함**
- **Precision**: `gpu_measure.py`/`gpu_measure_2048.py` 확인 결과 GPU 연산은 **FP32**(`.astype(np.float32)`만 쓰고 fp16/int8 캐스팅 없음), analog는 ADC/DAC **8-bit** — **정밀도도 정규화 안 함** (GPU가 4배 더 높은 정밀도로 계산)
- 45nm→8nm로 스케일링 추정은 시도하지 않음: ADC는 디지털 로직과 달리 열잡음(kT/C)·소자 미스매치로 정밀도가 결정되는 회로라 공정 미세화 이득이 훨씬 적고(디지털처럼 세제곱 스케일링 적용 불가), 정량적 스케일링 공식을 쓰면 리뷰어 반박 리스크가 커서 **"정규화 안 함"을 캡션에 명시하는 방식으로 처리**

### 10-2. 캡션에 추가한 문구 (3개 파일)
> "Process node not normalized: ADC (45 nm), DAC (as-cited process), analog cell (IGZO TFT, not a standard CMOS node), and GPU (Samsung 8 nm) are each taken as-is from their own source. Precision not matched either: GPU measured in FP32 (PyTorch default, no fp16/int8 cast); analog uses 8-bit ADC/DAC quantization."

적용된 파일: `fig_adc_dac_gpu_vs_analog_bar.py`, `table_periphery_adc_dac.py`, `table_periphery_adc_dac_combined.py`
(write 포함 표 3개는 아직 미적용 — 필요 시 추가)

---

## 7. 미해결 / 다음 단계
- K(subthreshold slope)를 키운 "이상적 케이스" 재현 실험 (아직 미착수)
- Scoring 단계 문서 토큰 저장(write) 비용은 현재 스코프 밖 (coarse table만 반영)

---

## 8. 세부 보충 사항

### 8-1. Analog latency 산출 근거 (64us)
- 가정: 같은 쿼리 토큰이면 centroid/candidate 여러 개를 **동시에(병렬로)** 읽고, 쿼리 토큰 32개만 **순차적으로** 처리.
- 순차 이벤트 수 = 32(coarse search) + 32(scoring) = 64회, 이벤트당 1us(sensing) → **64us/query**, Option A/C 동일(둘 다 32+32 구조).
- 완전 순차(셀 1개씩) 가정 시 9,088~9,376회 × 1us = 9~9.4ms로 계산되어, 이 병렬성 가정이 결과에 결정적 — array word-line/bit-line 드라이버 개수 등 실제 구현 제약에 따라 달라질 수 있음(미검증 가정).

### 8-2. GPU 워크로드 정의 (재현용)
```python
coarse = Q @ C.T            # (batch*32, 128) x (128, N_centroid)
_ = coarse.topk(2, dim=1)   # nprobe=2 흉내
score = Q @ D.T             # (batch*32, 128) x (128, N_candidate)
maxsim = score.view(batch, 32, -1).max(dim=2).values.sum()   # per-query MaxSim
```
- Q: 실제 `[clip99.9]query_embs_96x128.xlsx`(프로토타입) / `scale_query_embs_255x32x128.pt`(시스템레벨)에서 로드.
- C: 실제 centroid 데이터. D(candidate 문서 임베딩)는 실측 통계(평균/표준편차)와 일치하는 합성(synthetic) 데이터 — 실제 문서 벡터 전수 로드는 안 함.
- 전력은 `pynvml`로 20ms 간격 폴링, latency는 `time.perf_counter()`로 측정, energy = avg_power × latency.

### 8-3. Table 5 원자료 (GPU 실측, RTX 3070Ti, Option A 기준)
| batch | 프로토타입 E/query | 프로토타입 latency | 프로토타입 power | 시스템레벨 E/query | 시스템레벨 latency | 시스템레벨 power |
|---|---|---|---|---|---|---|
| 1 | 2,404.5 μJ | 93.77 μs | 25.6 W | 11,773.5 μJ | 118.83 μs | 99.1 W |
| 32 | 343.4 μJ | 3.28 μs | 104.9 W | 4,369.5 μJ | 18.94 μs | 230.7 W |
| 256 | 200.6 μJ | 0.96 μs | 209.4 W | 3,949.7 μJ | 15.50 μs | 254.9 W |

(batch가 커질수록 latency/query는 줄어들지만 GPU가 더 바빠져서 평균 전력은 오히려 올라감 — 그래도 에너지(=전력×시간)는 순감소.)

### 8-4. 파일 정리 이력
- 2026-07-09: TOPS/W 관련 스크립트/표/요약을 `논문 figure\`, `xyce_ColBERT용소자\...\260505공정 결과\`에서 전부 `ColBERT\TOPSW\`로 이동 (Xyce .cir/.sub 회로 파일만 원 위치 유지).

### 8-5. 참고자료 원본 구조
`energy_calcul_revised_v2.xlsx`에는 `Revised_v2`(이번에 인용한 시트) 외에 `AIMDC`, `CNN`, `Peri` 시트도 존재 (다른 참고/원본 계산으로 추정, 이번 분석에는 미사용).

---

## 11. (추가, 2026-07-21) 그림 포맷/레이아웃 조정 로그

수치/방법론 변경은 없음. Fig.5a-c 논문용 그림들의 포맷을 서로 맞추는 작업.

### 11-1. `table_periphery_adc_dac_stacked.py`
- 행 간격(ROW_H)·블록 간격(BLOCK_GAP)·헤더 여백을 기존의 70% 수준으로 축소 (사용자 요청: "위아래 간격을 원래의 70%가 되게").
- 축소 과정에서 섹션 타이틀 밑줄이 글자 디센더에 붙는 문제 발생 → 그 밑줄 간격만 원래 값(0.018)으로 되돌림.
- `bbox_inches="tight"`가 축(axes) 사각형 전체를 항상 포함하는 matplotlib 특성 때문에 캡션 아래 여백이 남는 문제 → PIL로 실제 잉크 bbox까지 자동 크롭하는 후처리 단계 추가.

### 11-2. `table_periphery_full_with_write.py`
- Prototype/System-level 비교 표에서 Prototype 블록 제거, System-level만 남기고 "System-level scale (2,048 centroids + 255 queries + 10,988 documents)"를 표 제목으로 승격.
- 블록이 하나로 줄면서 전체 figure 높이가 줄었고, 그 결과 "+ Write (amortized, 255q)" 2줄 헤더가 위쪽 구분선과 겹침 → 헤더 위/아래 여백을 늘려서 해결.

### 11-3. `plot_mrr_final.py` (`■Fig.5a, table1/`)
- 파란색(Vth Compensation 라인)을 `#2196F3` → `#1565C0`으로 변경, `plot_combined_final.py`/`plot_r50_final.py`의 파란색과 통일.
- x/y축 눈금·축 제목 폰트를 23/20pt에서 20%, 이어서 10% 축소 후, 눈금 폰트를 축 제목과 동일한 16.56pt로 통일.
- 주황색(No Comp) 마커를 원(`o`) → 다이아몬드(`D`)로 변경.
- 범례 텍스트 축약: "Vth Compensation\n(This Work)"/"No Compensation" → "Vth comp (this work)"/"No comp" (맨 앞 글자만 대문자).
- 범례 테두리 두께 1.5 → 1.2pt (`■Fig.5b/plot_seed19_q2_r2_1565C0.py`의 boxed 범례와 동일 값으로 통일).

### 11-4. `fig_full_periphery_with_write_gpu_vs_analog_bar.py`
- `make_figure()`에 `figsize`, `spine_lw`, `xlim_margin` 파라미터 추가 (기존 호출은 기본값으로 동작 그대로 유지).
- `fig_full_periphery_with_write_topsw_bar_linear_ticks0p1.png`: 플롯 박스(축 사각형) 크기를 `[graph_optC]mrr_final.png`와 동일하게(5.03×3.88 in) 맞추기 위해 figsize를 (6.062, 5.924)로 역산(반복 측정으로 solve). x/y축 폰트도 mrr_final과 동일한 16.56pt로 통일. 테두리 두께도 mrr_final 기본값에 맞춰 1.2 → 0.8pt로 축소.
- `fig_full_periphery_with_write_energy_bar_linear.png`: 테두리 1.2 → 0.8pt. x축 눈금 폰트를 y축 눈금(24pt)에 맞춤. 막대 위 숫자 라벨(예: "309,126.58 nJ") 폰트를 10% 확대(23.76→26.136pt)했는데, 커진 텍스트가 GPU 막대 중앙 정렬 상태로 왼쪽 축 테두리를 침범 → 막대는 그대로 두고 플롯 좌우 여백(xlim_margin)을 0.5 → 0.62로 넓혀서 해결.

### 11-5. Git
- `feature/xyce` 브랜치에 TOPSW 폴더를 제외한 나머지(Vth 보상 파이프라인, xyce 회로 시뮬레이션, Fig.4d/5a/5b 스크립트) 커밋 후 push.
- `feature/xyce`에서 `feature/topsw` 브랜치를 새로 만들어 TOPSW 폴더만 별도로 커밋, `-u origin feature/topsw`로 push (신규 브랜치).
