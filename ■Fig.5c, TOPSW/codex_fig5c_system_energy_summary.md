# Codex 작업 요약 — Figure 5c 시스템 에너지 분석

작성일: 2026-07-20

이 문서는 Figure 5c의 시스템 수준 에너지 분석을 위해 확정한 조건, 계산식, 표와 그래프 파일, Supplementary Table 문구, Figure caption 및 논문 본문 초안을 정리한 것이다.

## 1. 확정된 시스템 조건

| 항목 | 값 |
|---|---:|
| Centroids | 2,048 |
| Candidate documents | 10,988 |
| Queries | 255 |
| Query tokens | 32 |
| Total rows | 2,048 + 10,988 = 13,036 |
| Row–token evaluations/query | 13,036 × 32 = 417,152 |
| Column–token events/query | 128 × 32 = 4,096 |
| Operations/query | 106,790,912 FLOPs |

## 2. 부품별 에너지 계산

| Component | 계산 | Energy/query |
|---|---:|---:|
| Cell search | 656.90 fJ/eval × 417,152 evals | 274.03 nJ |
| Index write | 3,133,780 nJ ÷ 255 queries | 12,289.33 nJ |
| TIA | 45 pJ/eval × 417,152 evals | 18,771.84 nJ |
| ADC | 1.5 pJ/conversion × 417,152 conversions | 625.73 nJ |
| DAC | 0.52 pJ/conversion × 4,096 conversions | 2.13 nJ |
| RWL/SBL driver | 450 pJ × (417,152 + 4,096) events | 189,561.60 nJ |
| ML buffer | 210 pJ/eval × 417,152 evals | 87,601.92 nJ |

전체 index-write energy는 write 구간의 순간 전력을 시간에 따라 적분해 얻은 값이다.

`E_write,total = integral P(t)dt = integral V(t)I(t)dt = 3,133,780 nJ`

`amortized over 255 queries`는 write가 query마다 다시 발생한다는 뜻이 아니다. 한 번 발생한 전체 write energy를 255개 query에 나누어 query당 비용으로 환산했다는 뜻이다. 더 쉬운 본문 표현으로는 `the one-time index-write energy distributed across the 255-query workload`를 사용할 수 있다.

## 3. 누적 에너지 표

| Scenario | Incremental search energy | Write energy | Cumulative total energy/query |
|---|---:|---:|---:|
| Cell only | 274.03 nJ | 12,289.33 nJ | 12,563.36 nJ |
| + TIA | 18,771.84 nJ | shared | 31,335.20 nJ |
| + ADC | 625.73 nJ | shared | 31,960.93 nJ |
| + DAC | 2.13 nJ | shared | 31,963.06 nJ |
| + Driver | 189,561.60 nJ | shared | 221,524.66 nJ |
| + ML buffer | 87,601.92 nJ | shared | 309,126.58 nJ |

표 해석 시 다음 사항에 유의한다.

- `Search/query` 값은 누적값이 아니라 해당 부품이 추가하는 incremental energy이다.
- 누적값은 `Cumulative total energy/query` 열에만 표시한다.
- Write energy는 모든 행에 반복해서 더하지 않고 전체 시스템에서 공유되는 한 번의 비용이므로 첫 행에 수치를 쓰고 이후 행에는 `shared`라고 표시한다.
- DAC는 신호 흐름에 맞추어 ADC 다음, driver 이전에 배치한다.

## 4. GPU baseline 비교

| 항목 | RTX 3070 Ti GPU | This work |
|---|---:|---:|
| Energy/query | 3,949,707.8 nJ | 309,126.58 nJ |
| Energy efficiency | 0.0270 TOPS/W | 0.3455 TOPS/W |
| Relative efficiency | 1× | 12.8× |

검산 결과는 다음과 같다.

- Energy ratio = 3,949,707.8 ÷ 309,126.58 = 12.77699 ≈ 12.8×
- Energy reduction = 1 − 309,126.58 ÷ 3,949,707.8 = 92.173% ≈ 92.2%
- Analog efficiency = 106,790,912 FLOPs ÷ 309,126.58 nJ = 0.34546 TOPS/W ≈ 0.3455 TOPS/W
- GPU efficiency = 106,790,912 FLOPs ÷ 3,949,707.8 nJ = 0.02704 TOPS/W ≈ 0.0270 TOPS/W

TOPS/W는 `Tera Operations Per Second per Watt`의 약자로, 1 W의 전력으로 초당 수행할 수 있는 연산량을 뜻한다.

## 5. 단위와 계산 시 주의사항

- 10^-6은 micro, 10^-9은 nano, 10^-12은 pico이다.
- Power × time의 결과는 power가 아니라 energy이다.
- `0.45 mW × 1 us = 450 pJ`이며 `450 pW`가 아니다.
- TIA 값도 `45 pJ/event`로 써야 하며 `45 pW`와 혼동하지 않는다.
- Driver가 TIA보다 event당 10배 큰 이유는 driver의 사용 전력이 0.45 mW, 즉 450 uW이고 TIA의 환산 전력이 45 uW이기 때문이다.
- 최종 analog energy는 `309,126.58 nJ`이다. `309,128.58 nJ`로 잘못 표기하지 않도록 주의한다.

## 6. 표 파일

기존 원본은 유지하고 새 파일을 생성했다.

### 원본

- `table_periphery_full_with_write.py`
- `table_periphery_full_with_write.png`

### Figure 5 범위 및 간격 수정 버전

- `table_periphery_full_with_write_fig5_scope_compact10.py`
- `table_periphery_full_with_write_fig5_scope_compact10.png`

### GPU baseline 행 추가 버전

- `table_periphery_full_with_write_fig5_scope_compact10_gpu_row.py`
- `table_periphery_full_with_write_fig5_scope_compact10_gpu_row.png`

표의 현재 제목은 다음과 같다.

> System-level scale used in Table 1 and Fig. 5a (2,048 centroids + 255 queries + 10,988 documents)

추가된 GPU 행은 다음 정보를 포함한다.

> GPU baseline (RTX 3070 Ti) | — | — | 3,949,707.8 nJ | 0.0270 TOPS/W | 1×

## 7. 그래프 파일과 시각화 결정

그래프 생성 스크립트는 다음과 같다.

- `fig_full_periphery_with_write_gpu_vs_analog_bar.py`

생성된 주요 그래프는 다음과 같다.

- `fig_full_periphery_with_write_topsw_bar.png`
- `fig_full_periphery_with_write_topsw_bar_axes15smaller.png`
- `fig_full_periphery_with_write_topsw_bar_linear.png`
- `fig_full_periphery_with_write_topsw_bar_linear_ticks0p1.png`
- `fig_full_periphery_with_write_energy_bar.png`
- `fig_full_periphery_with_write_energy_bar_linear.png`

현재 권장 구성은 다음과 같다.

- Main panel은 `fig_full_periphery_with_write_topsw_bar_linear_ticks0p1.png`를 사용한다.
- Inset은 `fig_full_periphery_with_write_energy_bar_linear.png`를 사용한다.
- 12.8× 차이는 두 막대의 상대적 크기를 직접 보여주는 linear scale이 log scale보다 직관적이다.
- Energy-efficiency main panel의 y축 범위는 약 0–0.4이며 눈금은 0.0, 0.1, 0.2, 0.3, 0.4로 설정했다.
- Energy inset의 y축 제목은 `Energy / query (×10⁶ nJ)`로 정리했다.
- Inset의 막대 위에는 축 배율과 무관하게 실제 값을 `3,949,707.8 nJ`와 `309,126.58 nJ`로 표시한다.
- Inset으로 축소했을 때 읽을 수 있도록 숫자 label을 확대하고, main figure의 x축과 y축 글자는 별도 버전에서 축소했다.

## 8. Supplementary Table 문구

### 제목

> Table S_X1. System-level energy accounting including the core cell, full peripheral circuitry, and GPU baseline

### 본문

> The system-level analysis used 2,048 centroids, 10,988 candidate documents, 255 queries, and 32 query tokens, resulting in 417,152 row–token evaluations and 4,096 column–token events per query. The Xyce-simulated cell-search energy was 274.03 nJ/query, while the one-time index-write energy was amortized over 255 queries to 12,289.33 nJ/query.
>
> The TIA, ADC, DAC, driver, and ML-buffer energies were 18,771.84, 625.73, 2.13, 189,561.60, and 87,601.92 nJ/query, respectively. The cumulative energy of this work was 309,126.58 nJ/query, yielding 0.3455 TOPS/W. The measured RTX 3070 Ti GPU baseline consumed 3,949,707.8 nJ/query and achieved 0.0270 TOPS/W, corresponding to a 12.8× energy-efficiency improvement for this work. Process technologies and numerical precisions were not normalized between the analog system and GPU.

## 9. Figure 5c caption 권장안

> **Figure 5c | System-level energy efficiency and energy-per-query comparison with the GPU baseline.** The main panel compares the energy efficiency of the Vth-compensated analog system and the measured RTX 3070 Ti GPU baseline under the workload used in Table 1 and Figure 5a. This work achieves 0.3455 TOPS/W compared with 0.0270 TOPS/W for the GPU, corresponding to a 12.8× improvement. The inset shows the corresponding energy consumption of 309,126.58 nJ/query for this work and 3,949,707.8 nJ/query for the GPU. The analog estimate includes core-cell search, the one-time index-write energy distributed across 255 queries, and the modeled TIA, ADC, DAC, RWL/SBL search drivers, and ML buffer. Detailed component-level energy accounting is provided in Supplementary Table S_X1.

공정과 precision 차이에 관한 제한 사항은 본문에서 생략하더라도 Supplement 또는 caption에는 남기는 것이 안전하다. Caption에 포함할 경우 다음 문장을 마지막에 추가한다.

> Process technologies and numerical precisions were not normalized between the analog system and GPU.

## 10. Figure 5c 논문 본문 최종안

> Figure 5c evaluates the energy cost using the same system-level workload as Table 1 and Figure 5a, consisting of 2,048 centroids, 10,988 candidate documents, 255 queries, and 32 query tokens. The estimate includes the Xyce-simulated core-cell search energy, the one-time index-write energy distributed across the 255-query workload, and the modeled peripheral circuitry comprising the TIA, ADC, DAC, RWL/SBL search drivers, and ML buffer. Detailed component-level energy accounting is provided in Supplementary Table S_X1. Under this full accounting, the Vth-compensated analog system consumes 309,126.58 nJ/query, compared with 3,949,707.8 nJ/query for the RTX 3070 Ti GPU baseline measured at a batch size of 256 using nvidia-smi power polling. This corresponds to a 92.2% reduction in energy consumption. The analog system achieves 0.3455 TOPS/W compared with 0.0270 TOPS/W for the GPU, yielding a 12.8× improvement in energy efficiency. These results demonstrate that the retrieval robustness provided by Vth compensation retains a substantial system-level energy advantage even after accounting for the complete peripheral circuitry and one-time write cost.

## 11. 문장과 표기 통일

- 본문에서 `amortized`가 어렵게 느껴지면 `distributed across the 255-query workload`를 사용한다.
- `Detailed component-level energy accounting is provided in Supplementary Table S_X1.`는 본문과 caption에 모두 짧게 포함할 수 있다.
- 공정과 numerical precision의 차이는 본문에서는 생략하고 caption 또는 Supplement에서 명시하는 방향으로 정리했다.
- GPU baseline은 RTX 3070 Ti, batch size 256, nvidia-smi power polling 측정값이다.
- GPU는 FP32이고 analog는 8-bit ADC/DAC quantization을 사용한다.
- Process node는 GPU Samsung 8 nm, ADC 45 nm, DAC는 인용 문헌의 공정, cell은 IGZO TFT이다.
- 기존 원고의 `digitabaseline`은 `digital baseline`으로 수정한다.
- `no compensation's dead zone`보다는 `the uncompensated cell's dead zone`이 자연스럽다.
- `Vth compensation`, `Vth-compensated`, `No compensation`의 대소문자와 하이픈 사용을 원고 전체에서 통일한다.

## 12. 최종 핵심 결과

전체 core cell, amortized index write, TIA, ADC, DAC, RWL/SBL driver 및 ML buffer를 포함한 Vth-compensated analog system의 에너지는 `309,126.58 nJ/query`이다. 동일한 연산량에서 RTX 3070 Ti GPU는 `3,949,707.8 nJ/query`를 소비한다. 이에 따라 analog system은 `0.3455 TOPS/W`를 달성하며 GPU의 `0.0270 TOPS/W`보다 `12.8×` 높은 에너지 효율과 약 `92.2%` 낮은 query당 에너지를 보인다.
