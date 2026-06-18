# ColBERT MaxSim Analog Circuit Simulation 정리

> 작성일: 2026-06-02
> 참조 폴더: `C:\Users\nmdl-khb\ColBERT\centroidset`
> 회로 파라미터 출처: `xyce_ColBERT용소자\260505공정 결과\260505공정 결과`

---

## 1. 회로 개요 — 3-Phase 동작

### Phase 1 — Pre-charge
- VCOMP=3V, VREAD=3V, VML=0V
- 모든 노드를 VPRE로 초기화

### Phase 2 — Data Store + Vth Compensation
- VCOMP=3V, VREAD=-3V, VML=0V
- 비트라인에 ±V1 인가 → M0 꺼질 때까지 VSN 방전
- **저장 결과**: `VSN = -V1 + Vth`,  `VSN_bar = V1 + Vth`
- Vth 편차가 저장값에 자동 포함됨 (보상 원리)

### Phase 3 — L2 Search
- VCOMP=-3V, VREAD=3V, VML=0V, VDD=1.7V
- VSL0 = V2 (쿼리 전압) → 캐패시티브 커플링
- **VGS_M0 = V2 - V1 + Vth** (V1: 저장된 데이터, V2: 쿼리)
- 출력 전류: `IDS = f(|V2-V1| + Vth)` — **차이만 의존, 절댓값 무관**

### 핵심 성질 (V2-V1 불변성)
V2=0.5V/V1=-0.5V 와 V2=0.7V/V1=-0.3V 는 V2-V1=1V 로 동일 → **IDS 완전히 동일**.
회로는 두 전압의 절댓값이 아닌 차이만을 연산한다.

---

## 2. die4 소자 파라미터 (회로 모델 기반)

| 파라미터 | 값 |
|---|---|
| L | 6.112020 |
| K | 2.731949 |
| V0 (Vth) | 0.151045 V |
| B | -10.713911 |
| VDS (fitting) | 1.0 V |
| VDS (동작) | 1.7 V |

**VDS 보정**: MOSFET linear/saturation 모델 적용 (VoD < VDS: saturation, VoD > VDS: linear)

---

## 3. Option A vs C 비교

### Option A — Vth 보상 있음
- `VGS = |V2-V1| + Vth_new`, model V0 = Vth_new
- `VGS - V0 = |V2-V1|` → **Vth에 완전히 무관**
- Vth가 최대 +1.0V shift해도 출력 전류 변화 없음 (5가지 Vth 커브 완전 겹침)

### Option C — 보상 없음
- `VGS = |V2|` (V1 없음, V2 직접 인가)
- Dead zone = ±Vth_new → Vth 증가할수록 dead zone 확대
- Vth=1.151V: dead zone ±1.151V → V2∈[-2,2]의 대부분 OFF

---

## 4. V1=0 검증 (plot_V1zero_vs_optionA.py)

| 케이스 | V2 | V1 | VGS |
|--------|-----|-----|-----|
| Option A | 0.5V | -0.5V | 1.0 + Vth |
| V1=0 고정 | 1.0V | 0V | 1.0 + Vth |

→ x = V2-V1 축 기준으로 **완전히 겹침** (Difference = 0.000000 uA)

---

## 5. ColBERT MaxSim — 회로 연결

**MaxSim**: `S(q,d) = Σᵢ maxⱼ(Qᵢ·Dⱼᵀ)`

L2-normalized 벡터에서: MaxSim ≡ MinL2

**회로 어레이 동작**:
```
      dim0    dim1   ...  dim127       ← 쿼리 전압 (열)
       |       |            |
Row 0: [D00]  [D01]  ...  [D0,127] → ML0: I₀ = Σₖ f(|Qₖ-D0ₖ|)
Row 1: [D10]  [D11]  ...  [D1,127] → ML1: I₁ = Σₖ f(|Qₖ-D1ₖ|)
  ...
Row M: [...]  [...]  ...  [...]    → ML_M

argmin(ML 전류) = MaxSim winner (최소 거리 = 최대 유사도)
```

---

## 6. Python 시뮬레이션 — 데이터

| 파일 | 내용 |
|---|---|
| `centroids_100x128.xlsx` | 100개 centroid 벡터, 128차원, L2-normalized |
| `query_embs_96x128.xlsx` | 96개 쿼리 토큰 (q0~q2, 각 32토큰), L2-normalized |

**임베딩 값 범위**:
- centroids: [-0.3194, +0.3402]
- query_embs: [-0.3396, +0.3355]
- **L2 norm = 1.0000** (모두 단위벡터)

**VGS 동작 범위 (scale x1)**:
- `|V2-V1|` max = 0.34+0.34 = **0.68V** (개별값 ±0.34V이지만 차이는 최대 0.68V)
- VGS 범위: Vth(0.151V) ~ 0.83V

---

## 7. Python 시뮬레이션 결과

### 핵심 수식
```python
diff[i,j,k]  = |Q[i,k] - C[j,k]|        # (96, 100, 128)
VGS[i,j,k]   = diff + Vth                # Option A
I_total[i,j] = sum_k(IDS(VGS)) [uA]     # Match Line KCL 합산
best_analog   = argmin_j(I_total[i])     # 최소 전류 = MaxSim winner
```

### 결과 (scale x1 — 임베딩 그대로)

| 지표 | 값 |
|------|-----|
| **Top-1 accuracy** | **94.8%** (91/96 tokens) |
| **Top-3 accuracy** | **100%** |
| **Top-5 accuracy** | **100%** |
| **Avg Spearman rho** | **0.9592** |

| Query | Top-1 |
|-------|-------|
| q0 (t0-31) | 90.6% |
| q1 (t0-31) | 96.9% |
| q2 (t0-31) | 96.9% |

### Scale 비교

| Scale | VGS max | Top-1 | Spearman ρ |
|-------|---------|-------|------------|
| x1 (±0.34V) | 0.83V | **94.8%** | **0.9592** |
| x2 (±0.68V) | 1.51V | 92.7% | 0.9063 |
| x3 (±1.02V) | 2.19V | 95.8% | 0.9491 |

> **x1이 rho 최고인 이유**: VGS 0.15~0.83V 구간이 logistic curve의 가장 선형적인 구간 → ranking 왜곡 최소

### Dot Product vs Analog Current 관계
- 반비례: dot product 높음 → 유사 → |V2-V1| 작음 → IDS 작음
- Spearman rho > 0.95 → argmin(전류) = argmax(dot product) 신뢰 가능

### 실패 원인 (5.2%)
- 실패한 토큰은 모두 ideal에서도 top-1/top-2 dot product score 차이(margin)가 매우 작은 near-tie 케이스
- 회로 정밀도 문제가 아닌, 애초에 두 centroid가 거의 같은 거리인 경우

---

## 8. Option C 시뮬레이션 (Vth 랜덤 편차 모델)

### 8-1. 배경

Option A는 Vth compensation으로 소자 편차를 소거하지만, 실제 하드웨어(PBS 효과 등)에서는
Vth가 시간에 따라 양(+)의 방향으로 drift함. Option C는 이를 모델링한 no-compensation 시나리오.

### 8-2. Option C 회로 모델

```
Option A: vgs = |Q_k - D_k| + Vth   → Vth 소거, 소자 편차 무관
Option C: vgs = |Q_k - D_k|         → Vth 소거 없음, dead zone 발생
          dead zone = |Q-D| < Vth_ij  (소자마다 다름)
```

### 8-3. Vth 분포 설계

| 항목 | 값 |
|------|---|
| 분포 형태 | Truncated Gaussian |
| mean | 0.0 V |
| std | 0.15 V |
| range | [0, 0.5] V (PBS: 양의 shift만) |
| Vth_actual | 0.151045 + shift |
| 할당 단위 | (Qi, Dj) pair마다 독립 샘플링 |
| seed | 42 (재현 가능) |

**분포 형태**: 0V에서 피크(shift 없는 소자가 가장 많음), 0.5V쪽으로 decay

### 8-4. 시뮬레이션 구조

```python
# 문서 pid당 1회 샘플링
Vth_mat = sample_vth((32, M))   # shape: (32, M) — Qi × Dj pair마다 독립

# Option C IDS 계산
vgs = |Q_k - D_k|               # Vth 보상 없음
IDS = f(vgs, Vth_mat[i,j])      # 소자별 Vth 적용

# MinCurrent → rank (낮을수록 유사)
score = Σᵢ minⱼ Σₖ IDS(|Qᵢₖ - Dⱼₖ|, Vth[i,j])
```

### 8-5. 6가지 방법 비교 결과 (3 queries, 200 docs)

| Query | Digital f32 | Digital 2bit | OptionA f32 | OptionA 2bit | OptionC f32 | OptionC 2bit |
|-------|------------|------------|------------|------------|------------|------------|
| q0 | rank **1**/127 | rank **1**/127 | rank **1**/124 | rank **1**/124 | rank 163/198 | rank 167/198 |
| q1 | rank **1**/70 | rank **1**/70 | rank **1**/82 | rank **1**/82 | rank 22/193 | rank 31/193 |
| q2 | rank **1**/108 | rank **1**/108 | rank **1**/85 | rank **1**/85 | rank **4**/194 | rank **6**/194 |

| | dig_f32 | dig_2bt | optA_f32 | optA_2bt | optC_f32 | optC_2bt |
|--|---------|---------|---------|---------|---------|---------|
| Success@10 | 100% | 100% | 100% | 100% | 0% | 0% |
| Success@20 | 100% | 100% | 100% | 100% | 0% | 0% |
| Success@50 | 100% | 100% | 100% | 100% | 67% | 67% |

### 8-6. 후보 집합 비교 (Step 3, nprobe=2)

| Query | Digital | OptionA 교집합 | Jaccard(D↔A) | OptionC 교집합 | Jaccard(D↔C) |
|-------|---------|-------------|------------|-------------|------------|
| q0 | 127 | 124/127 | **0.976** | 127/127 | 0.641 |
| q1 | 70 | 64/70 | 0.727 | 70/70 | 0.363 |
| q2 | 108 | 85/85 | 0.787 | 108/108 | 0.557 |

- OptionA: Digital과 Jaccard 높음 → Vth compensation 덕분에 centroid 선택 유사
- OptionC: Digital ⊂ OptionC (Digital 후보 전체 포함), Vth 노이즈로 더 넓게 커버

### 8-7. 시각화 파일 목록

| 파일 | 내용 |
|------|------|
| `vth_shift_distribution.png` | Truncated Gaussian Vth 분포 |
| `compare_candidates.png` | Digital vs OptionA 후보 집합 비교 |
| `compare_candidates_optionC.png` | Digital vs OptionC 후보 집합 비교 |
| `compare_candidates_all.png` | 3-way 통합 비교 (7-way Venn 분할 bar) |
| `compare_ranking_L2_optA.png` | Digital L2 vs OptionA rank scatter + bump chart |
| `compare_ranking_L2_optC.png` | Digital L2 vs OptionC rank scatter + bump chart |
| `compare_ranking_all3.png` | Digital L2 / OptionA / OptionC 3-way 비교 |

---

## 9. 생성 파일 목록

| 파일 | 위치 | 내용 |
|------|------|------|
| `plot_phase3_L2_die4.py/png` | 260505공정 결과/ | Phase 3 VDS 1V vs 1.7V 비교 |
| `plot_AC_vth_comparison.py/png` | 260505공정 결과/ | Option A vs C, 5종 Vth 비교 |
| `plot_L2_analysis.py` | 260505공정 결과/ | 범용 분석 스크립트 (임의 die 적용) |
| `plot_V1zero_vs_optionA.py/png` | 260505공정 결과/ | V1=0 vs Option A 동일성 검증 |
| `plot_L2_die4_AC_reordered.png` | 260505공정 결과/ | 서브플롯 재배치 (A_log, A_lin, C_log, C_lin) |
| `sim_maxsim_analog.py/png` | centroidset/ | MaxSim 시뮬 (3가지 scale 비교) |
| `sim_maxsim_v2.py/png` | centroidset/ | MaxSim 시뮬 개선 시각화 (6-panel) |

---

## 10. 다음 단계

- [ ] Xyce 재설치 후 단일 셀 V-I 검증 (단계 1)
- [ ] 1-row Xyce 시뮬레이션 (128 차원, 1 쿼리 vs 1 centroid)
- [ ] Xyce vs Python 모델 정량 비교
- [ ] 전체 array (100×128) Xyce 시뮬레이션 (규모 확인 후 결정)
