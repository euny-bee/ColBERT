# TOPS/W 에너지 분석 — 결정 사항(축별) 로그

각 축마다 검토한 옵션과 현재 채택안을 기록. 최종 결과 확인 후 다른 조합도 시도 가능하도록 유지.

## 축1: 에너지 계산 방법
- **채택**: Tellegen 기반 — 서브회로에 연결된 모든 독립 전압원에 대해 `E = ∫ -V(t)×I(t) dt` 적분 후 합산
- 대안: 소자(트랜지스터)별 Vds×Ids 직접 적분 (아직 안 해봄, 필요시 cross-check용)

## 축2: Phase2(store/write) 에너지 적분 구간
- 방법1: Vth별 실제 settling 시간까지만 적분 → **기각** (회로가 Vth를 사전에 안다고 가정하는 비현실적 조건)
- **방법2 (채택)**: 전체 Vth 공통 고정 write-time = worst-case Vth 기준 settling×1.2 = 3200.1us

## 축3: Write 에너지 — PBS Vth 변이 분포 가중평균
5개 sweep point(V0=0.03/0.53/1.03/2.03/3.03V)를 TruncNorm 분포로 가중평균:

| 조건 | Vth shift 범위 | std | E_phase2 기댓값(fJ) | E_total 기댓값(fJ) |
|---|---|---|---|---|
| v2 | [0,1]V | 0.30 | 1379.6 | 1379.6 |
| v3 | [0,2]V | 0.60 | 1128.0 | 1128.0 |
| **v4 (채택, headline)** | **[0,3]V** | **0.90** | **923.5** | **923.5** |

- Option C write 에너지 = Phase1만 = 0.0072 fJ (Vth/PBS 조건 무관, 항상 동일)

## 축4: Phase3(search) V1/V2 전압 스케일
`maxsim_simulation_summary.md`(centroidset 구 폴더, vector 크기 조절 이전 예비실험)의 scale 비교:

| Scale | 개별값 범위 | VGS max | Top-1 | Spearman ρ | 비고 |
|---|---|---|---|---|---|
| x1 | ±0.34V | 0.83V | 94.8% | 0.9592 | **구(舊) 데이터 — 100-centroid 예비실험, vector 크기 조절 전** |
| x2 | ±0.68V | 1.51V | 92.7% | 0.9063 | |
| x3 | ±1.02V | 2.19V | 95.8% | 0.9491 | |

**정정(중요)**: 위 표는 `centroidset`(구) 폴더의 예비실험 데이터 기준. 실제 이 세션·Table 2·seed19에 쓰이는 데이터는
`centroidset_vector 크기 조절/06_newq_margin/[clip99.9]query_embs_96x128.xlsx`, `[clip99.9]centroids_100x128.xlsx` (vector 크기 조절 적용 후).
이 데이터의 실제 범위:
- Q: -0.98 ~ +1.00,  C: -0.86 ~ +0.81
- diff(Q-C): mean≈0, std≈0.375, 95%ile=±0.62, 99%ile=±0.89, max=±1.69

→ **기존 V1=1.0V, V2=[-1,+1]V sweep이 실제 seed19 데이터 스케일과 잘 맞음 (재조정 불필요)**. ±0.34V 재시뮬레이션 계획은 폐기.

## 축5: Phase3 V2(query 차이값) 가중 방법
- 방법A (구계산, 참고용): 5개 스윕값 단순 산술평균 → Option A ≈ 3788.7 fJ, Option C ≈ 1759.7 fJ *(폐기: V1/V2 스케일 오해 하에 계산됨)*
- **방법B (채택, 최종)**: q2 실제 (Q-C) diff 값 분포(1,228,800 샘플, `q2_diff_flat.npy`)로 가중평균
  - **Option A E_phase3 기댓값 = 716.5 fJ/search**
  - **Option C E_phase3 기댓값 = 505.9 fJ/search**
  - 앵커포인트: Option A는 phase123_dual_die4.cir (V1=1.0 고정, V2 스윕 → diff=|1-V2|), Option C는 phase13_optionC_die4.cir (bitline=VDD 고정, V2=diff 직접 주입, V2 스윕 -2~2 대칭 확장)
  - Option C 물리적 해석(사용자 확인): "No comp 조건, phase3에서는 Q-C 값 자체를 넣어준다" — bitline은 고정 VDD 레퍼런스, V2 자체가 diff(Q-C)를 직접 표현
  - 주의: Option A는 diff=1.0에서 에너지가 diff=1.5보다 큰 비단조 현상 있음 (nominal Vth=0.151V 고정 시뮬 영향으로 추정, 추가 조사 필요할 수 있음)

## 축8: 최종 에너지 요약 (fJ)
| 항목 | Option A | Option C |
|---|---|---|
| Write (Phase1+2, v4 [0,3]V PBS 가중) | 923.5 | 0.0072 |
| Search (Phase3, q2 실제 diff 가중, 1us sensing) | 716.5 | 505.9 |

**미해결**: Write 에너지가 "1회 인덱스 빌드당" 비용인지 "쿼리당" 비용인지 스코프 정리 필요 (기존 공식 초안은 write를 쿼리당 1회로 더했으나, 실제로는 인덱스 빌드시 1회만 발생하고 여러 쿼리에 걸쳐 재사용되는 게 맞을 가능성이 큼) → 사용자 확인 필요

## 축6: Phase3 sensing time
- **고정 1μs** (기존 합의, ~0.01μs–1000μs 안정 구간 내 안전마진)

## 축7 (미정): Phase3 Vth 처리
- 지금까지 Phase3 자체는 고정 nominal Vth(0.1510V)로만 시뮬 — Option C의 실제 PBS Vth 분포를 Phase3(search)에도 반영할지는 아직 미결정 (Phase2/write에만 반영함)

---
### 다음 작업
1. 축4를 x1(±0.34V)로 변경 → V1, V2 sweep 범위 재설정, Xyce 재시뮬레이션
2. 축5(q2 실제 diff 분포) 추출 → 가중평균 재계산
