# q2 (seed19) Figure 작업 요약 — Vth Compensation 논문 supplementary/본문용

이 문서는 `06_newq_margin` 폴더의 q2("where is hartwell ga", margin=20.06, seed=19) 관련 figure들을
논문/supple용으로 다듬은 작업 내역과, 그 과정에서 작성한 본문·캡션 초안을 정리한 것.

---

## 1. 다룬 그림 3종 + 대응 스크립트

| 내용 | 스크립트 | 최종 출력 이미지 |
|---|---|---|
| Digital vs Analog rank scatter (+ 범례, R² 텍스트박스) | `plot_seed19_q2_r2.py` | `[newq_margin_seed19]q2_scatter_r2.png`, `..._legend.png`, `..._legend_vertical_boxed.png`, `..._textbox.png` |
| ΔRank 분포 violin+box+jitter (inset) + std dev 통계박스 | `plot_q2_residual_violin_inset.py` | `[newq_margin_seed19]q2_residual_violin_w50/w70.png`, `..._stats_box_no_levene_blackedge.png` 등 |
| Table 2 (PBS-invariant baseline + Vth shift range별 성능) | `논문 figure/plot_table_optC_owncandidate_v2.py`(원본), `_v3.py`(sentence case) | `table_optC_owncandidate_v2.png`, `_v3.png` |
| q2 candidate 집합 겹침 stacked bar (Digital/Vth comp/No comp) | `plot_q2_candidate_bar_seed19.py` → `plot_q2_candidate_bar_seed19_v2.py`(최종 스타일) | `[seed19]q2_candidate_bar_v2.png`, `..._legend.png` |

---

## 2. 중요 발견 — 데이터셋 불일치

사용자가 처음 가져온 candidate-overlap bar chart(q0/q1/q2 3-panel)는
`06_newq_margin/plot_combined_split_newline_fixedcell.py`가 그린 것인데, 이 스크립트는
**`[vth_v3_fixedcell]step3_*_candidate_pids.xlsx` / `[vth_v3_fixedcell]step6_all_results.xlsx`**를 사용함.

반면 지금 다루는 q2 scatter/violin figure는 **`[seed19]step3_*.xlsx` / `[seed19]step6_all_results.xlsx`**를 사용 —
같은 폴더 안에 **서로 다른 시뮬레이션 run**이 공존하고 있었음 (파일 날짜도 vth_v3_fixedcell=7/1, seed19=7/2로 다름).

그 결과 q2 Venn 수치가 서로 달랐음:

| 항목 | vth_v3_fixedcell (원본 bar chart) | seed19 (scatter/violin과 동일 run) |
|---|---|---|
| dig∩A∩C | 132 | **130** |
| dig∩A\C | (표시 안 됨, 1) | **3** |
| dig∩C\A | (표시 안 됨) | **1** |
| A only | 8 | **0** |
| C only | 10 | **11** |
| A∩C\dig | 43 | **51** |
| \|OptionC\| | 185 | **193** |
| Jaccard(Dig↔A) | 0.719 | 0.719 (우연히 동일) |
| Jaccard(Dig↔C) | 0.706 | 0.668 |

→ `plot_q2_candidate_bar_seed19.py`를 새로 작성해서 **seed19 데이터로 q2만** 다시 그림 (수치는 step6 기반으로 직접 계산한 값과 정확히 일치 확인 완료).

---

## 3. Digital rank 축에 대한 이해 (supple 서술용)

- 이 실험은 3개 query, 200개 document pool(정답문서 3개 포함)로 구성된 통제 실험.
- q2 기준 digital의 coarse search가 pool 중 **134개**를 candidate로 선정 → 그 134개에 MaxSim 적용해 digital rank 1~134 부여 (완전 연속, 결측 없음).
- scatter figure의 목적: 이 134개 문서 각각이 analog(Vth comp / No comp)에서는 어떻게 되는지 확인.
- **analog의 coarse search가 그 문서를 candidate로 아예 뽑지 못하면** df_A/df_C에 존재 자체가 없어서 (x=digital rank, y=analog rank) 쌍이 성립하지 않고 그림에 안 나타남.
- 실제 예시 (가상 아님):
  - digital 38등 문서(pid 713562) → Vth comp(Option A) candidate에 없음 → 파란 원 계열에 x=38 없음
  - digital 93/101/122등 문서(pid 445913/127114/517219) → No comp(Option C) candidate에 없음 → 주황 다이아몬드 계열에 x=93,101,122 없음
- candidate bar chart의 "dig∩A\C"(3개), "dig∩C\A"(1개) 조각이 바로 이 결측 문서들의 개수와 일치.

## 4. Jaccard 유사도 설명

$$J(X,Y) = \dfrac{|X \cap Y|}{|X \cup Y|}$$

candidate **집합** 자체의 일치도(순위 무관). 예: Jaccard(Dig↔A) = |dig∩A| / |dig∪A| = 133/185 ≈ 0.719
(|dig∩A| = dig∩A∩C + dig∩A\C = 130+3 = 133, |dig∪A| = 134+184−133 = 185).

---

## 5. Figure 스타일링 반복 내역 (최종 상태 기준)

### violin inset (`plot_q2_residual_violin_inset.py`)
- Levene p 줄 제외한 std-dev 통계박스 버전 추가, 테두리 검은색 버전 추가
- y축 제목 "Analog rank - Digital rank" → "ΔRank (analog-digital)"로 축약 + 소문자
- x tick label "Vth Comp"/"No Comp" → "Vth comp"/"No comp", 두 줄바꿈(`Vth\ncomp`)
- 폰트: Malgun Gothic → **Arial** 우선순위로 변경
- x/y축 폰트 크기를 여러 차례 20%씩 키우면서, 라벨 겹침 발생 시 **최소한으로만** 그림 폭을 넓히는 방식으로 해결 (물리적 폭 대신 폭 확대를 택함)
- "Std Dev" → "Std dev"로, 그 글자만 20% 확대(나머지 본문과 붙는 간격은 유지) — `matplotlib.offsetbox`(TextArea+VPacker+AnnotationBbox)로 구현해 자연스러운 줄간격 확보

### scatter (`plot_seed19_q2_r2.py`)
- 세로 배치 + 흰 배경 + 검은 테두리 범례(`q2_scatter_r2_legend_vertical_boxed.png`) 추가

### scatter 색상 변형본 (`plot_seed19_q2_r2_1565C0.py`, 2026-07-14 ~ 07-21)
- `plot_seed19_q2_r2.py`를 복제한 별도 스크립트. 원본 파일/출력은 그대로 두고, Vth comp(Option A) 계열 색상만
  `#2196F3` → `#1565C0`(TOPSW 분석의 "this work" 파랑과 통일)로 교체, 출력 파일명에 `_1565C0` suffix를 붙여 저장
  (`[newq_margin_seed19]q2_scatter_r2_1565C0.png`, `..._textbox_1565C0.png`, `..._legend_1565C0.png`,
  `..._legend_vertical_boxed_1565C0.png`)
- 세로 배치 박스형 범례(`..._legend_vertical_boxed_1565C0.png`)의 "Vth comp" 라벨을 **"Vth comp\n(this work)"**
  로 수정 (마커는 첫 줄에 정렬). `■Fig.5b` 폴더에도 최신본을 복사해둠 — 스크립트 자체의 저장 경로(`OUT_BASE`)는
  `centroidset_vector 크기 조절\06_newq_margin`이라, 재실행할 때마다 그 폴더에 저장된 파일을 `■Fig.5b`로
  다시 복사해야 함
- 가로 배치 범례(`..._legend_1565C0.png`, ncol=3)도 동일하게 "Vth Comp (this work)"로 통일 (라벨만 세로형과
  대소문자 표기 차이 있음: 세로형은 "Vth comp", 가로형은 "Vth Comp" — 각 스크립트 원래 표기를 그대로 유지).
  라벨 길이 때문에 그림 폭을 4.6in → 6.2in로 확대

### Table (`plot_table_optC_owncandidate_v2.py` / `_v3.py`)
- v2는 원본 그대로 유지 (Vth Shift Range / Coarse Search / Cand. Recall)
- v3를 새로 만들어 "각 항목 첫 글자만 대문자, 나머지 소문자"로 변경 (Vth shift range / Coarse search / Cand. recall / Vth compensation / No compensation / PBS-invariant baseline / Under PBS-induced Vth variation)

### candidate bar (`plot_q2_candidate_bar_seed19_v2.py`, 최종)
- x축 라벨: Digital / OptionA / OptionC → **Digital / Vth comp / No comp**
- y축 "Candidate passages"도 x tick과 같은 굵기·크기로 통일, 이후 20%씩 2회 확대
- 값이 너무 작아(threshold=8 미만) 막대 안에 안 들어가는 라벨(예: 3, 1)은 **리더선으로 바깥에 빼서** 표시, 프레임 밖으로 튀어나가지 않도록 y좌표 하한 clamp
- 그림 폭 10% 확대
- 범례를 그래프에서 분리해 **별도 파일**로 (`_legend.png`)
- 값이 0인 범례 항목(dig only, A only) 자동 제외
- Jaccard 텍스트 주석 제거

---

## 6. 논문 본문/캡션 초안

### 6.1 위치 관계 (논의 시점 기준 넘버링)
- **Table 1** = candidate bar 아님, PBS-invariant baseline + Vth shift range table (Table 2 as coded, 논문에서는 Table 1)
- **Figure 5a** = MRR@10 line chart (Vth Compensation vs No Compensation, robust/degradation)
- **Figure 5b** = q2 scatter(Digital vs Analog rank) + violin inset

### 6.2 챕터 도입 (앞 챕터 "analog가 2bit quantization 손실을 없애고 Vth-compensation이 read-out non-ideality를 억제한다"는 주장과 연결)

> The preceding discussion argued, at a conceptual level, that an analog in-memory representation removes the 2-bit quantization loss inherent to digital residual storage, and that Vth-compensation keeps the resulting analog read-out close to this near-ideal operating point despite process-induced Vth variation. We now test this claim directly by simulating the full ColBERTv2 retrieval pipeline under PBS-induced Vth variation, comparing Vth-compensated and uncompensated analog scoring against the digital baseline in terms of end-to-end retrieval quality.

### 6.3 Table 1 문단 (coarse search 관련 강조 반영, dead-zone 메커니즘 문장 추가)

> Table 1 summarizes the resulting retrieval quality. Under the PBS-invariant baseline, that is, in the absence of any Vth shift, Vth Compensation attains MRR@10, nDCG@10, R@50, and R@1k values that are indistinguishable from those of standard digital scoring, both with and without 2-bit quantization. This confirms that the analog engine, once compensated, matches the digital baseline when no device variation is present. Under PBS-induced Vth variation, however, uncompensated scoring degrades markedly as the shift range widens from [0, 0.5] V to [0, 3.0] V, with MRR@10 falling from 78.1 to 2.1. As detailed in Supplementary Figure S_X7, this degradation arises because No Compensation's dead zone widens with increasing PBS, driving the cell current away from the ideal computation, whereas Vth Compensation cancels the threshold voltage and therefore never develops a dead zone. This degradation is not confined to the scoring stage. Cand. Recall, the coarse-search-stage metric, falls from 100.0 to 51.0 over the same shift range, tracking the decline seen in MRR@10, nDCG@10, R@50, and R@1k almost step for step. Coarse search and scoring, the two phases mapped onto a single AIMDC array in Fig. 4b, are therefore corrupted alike once the Vth shift exceeds the range the uncompensated cell can tolerate. This establishes the central problem addressed in this section. At the shift ranges expected under realistic PBS conditions, an uncompensated analog engine is not a viable substitute for the digital scorer.

### 6.4 Figure 5a 문단 (em dash 없이)

> Figure 5a isolates the MRR@10 trend from Table 1 to make this contrast explicit. Vth Compensation remains flat at approximately 79% across the entire shift range, effectively robust to PBS-induced Vth variation, while uncompensated scoring declines in an almost linear fashion. The shaded region between the two curves quantifies the benefit of Vth compensation, defined as the fraction of MRR@10 that would otherwise be lost to PBS-induced Vth variation. This region widens as the shift range increases, showing that the value of compensation grows precisely where it is needed most.

### 6.5 Figure 5b 문단 (em dash 없이, scoring-stage 명시)

> Table 1 and Figure 5a report aggregate retrieval metrics, which leaves open how the underlying rankings are disturbed at the level of individual documents. Figure 5b examines this for a representative query, plotting each candidate document's analog rank against its digital rank under both conditions. Under Vth Compensation, the points remain tightly clustered along the y = x diagonal, indicating that the relative ordering of documents is largely preserved. Under No Compensation, the points scatter broadly on both sides of the diagonal, so that documents are frequently promoted or demoted by tens of ranks relative to their digital position. The inset quantifies this spread as the distribution of ΔRank, the difference between analog and digital rank, for each condition. The standard deviation is 18.2 for Vth Compensation versus 47.9 for No Compensation, a more than two-fold difference. This rank-level view supplies the mechanism behind the aggregate degradation observed in Table 1 and Figure 5a. Uncompensated Vth variation does more than add uniform noise to the scores. It disorders the scoring-stage ranking enough to move relevant documents outside the top-k cutoff used by MRR@10 and R@50. Loss of compensation also enlarges the candidate set returned by coarse search (Figure S_X8), so No Compensation requires more MaxSim computation downstream than either Vth Compensation or Digital.

> **참고**: Figure 5b의 "ranking"은 coarse search(candidate 선택)가 아니라 **scoring 단계(MaxSim 연산)** 결과인 최종 문서 순위를 가리킴. 이미 coarse search를 통과해 양쪽에 공통으로 존재하는 문서(common_A/common_C)들끼리의 순위 흔들림만 보는 것이고, coarse search 자체의 붕괴(candidate 자체가 빠지는 것)는 Table 1의 Cand. Recall이 별도로 다룸.

### 6.6 캡션 초안

**Figure 5b 캡션**
> Per-document analog vs. digital rank for a single query. Blue circles are Vth-compensated retrieval (Vth comp), and orange diamonds are uncompensated retrieval (No comp). The dashed diagonal marks y = x, the case of perfect rank preservation. Vth compensation keeps analog ranks tightly clustered along the diagonal, while uncompensated ranks scatter substantially above and below it. The inset shows the distribution of the rank difference (ΔRank = analog rank − digital rank) for each condition (violin + box + jittered points), with standard deviations of 18.2 (Vth comp) vs. 47.9 (No comp), confirming a markedly narrower error spread under Vth compensation.

**Table 1 캡션**
> Retrieval quality under increasing PBS-induced Vth variation. The top block reports the PBS-invariant baseline (standard digital scoring with and without 2-bit quantization, and Vth-compensated analog scoring, all unaffected by Vth shift). The bottom block reports uncompensated analog scoring as the PBS-induced Vth shift range widens from [0, 0.5] V to [0, 3.0] V. MRR@10, nDCG@10, R@50, and R@1k all degrade sharply with wider shift ranges in the absence of compensation, and Cand. Recall, the coarse-search-stage metric, falls in step from 100.0 to 51.0 over the same range, while the PBS-invariant Vth-compensated result remains close to the digital baseline across all metrics.

**Figure 5a 캡션**
> MRR@10 as a function of PBS-induced Vth shift range. Vth compensation (blue) stays robust to PBS-induced Vth variation across the full shift range, while uncompensated retrieval (red) declines sharply from 78.4% to 2.1%. The shaded region marks the benefit of Vth compensation, the MRR@10 recovered relative to the uncompensated case.

### 6.7 Supplementary Figure S_X7 / S_X8 캡션 초안 (이번 세션 신규)

번호 순서: **S_X7 = dead-zone 회로 원인 그림** (Table 1 문단이 참조), **S_X8 = candidate turnover 그림** (Fig.5b 문단이 참조). S_X7이 "왜 dead zone이 생기는가"를 소자 레벨에서 먼저 보여주고, S_X8이 "그 결과 실제 candidate set이 어떻게 바뀌는가"를 보여주는 순서.

**Figure S_X7 (dead-zone 회로 원인, 2-panel)**

> Figure S_X7. Origin of the No Compensation dead zone. This figure grounds the dead-zone mechanism invoked in Table 1 at the device level, using the dual 3T1C cell (M0 handles d = V1 − V2 > 0, M3 handles d < 0) under an illustrative PBS mismatch. From a baseline Vth of 0.03 V, M0 shifts by 0.5 V and M3 by 1.5 V, both within the [0, 3.0] V variation range used for the system-level evaluation in Table 1. The left panel shows Vth Compensation alone: each device cancels its own shift individually, so the current stays symmetric in d regardless of the mismatch, with no dead zone. The right panel overlays No Compensation on the same axes; the uncancelled shifts set the dead-zone width directly, spanning d ∈ [−1.53, +0.53] (shaded). Any dimension whose difference falls in this range contributes almost no signal to MaxSim regardless of its true magnitude, while Vth Compensation continues to respond normally over the same range. This device-level asymmetry is the physical origin of the ranking degradation reported in Table 1.

- 스크립트: `03_vth_v3/plot_vth_split_linear.py` (신규 작성)
- 출력: `plot_vth_split_linear_vthcomp.png`(+ `_legend.png`), `plot_vth_split_linear_nocomp.png`(+ `_legend.png`) — 왼쪽/오른쪽 패널과 범례를 각각 분리, 범례 테두리 검은색
- No Comp 패널에 dead zone 구간 `[-1.53, +0.53]`을 보라색(violet) `axvspan`으로 음영 처리 (텍스트 라벨은 제거, 음영만 유지)
- 이전 버전(M0=0.5V, M3=3.0V, Vth_orig 미반영)에서 이번 세션에 **Vth_orig=0.03V를 실제로 더하는 방식**(M0=0.53V, M3=1.53V)으로 수정 — 라벨은 "shift"라고 쓰면서 실제로는 절대값을 넣던 이전 스크립트의 불일치를 바로잡음

**Figure S_X8 (candidate turnover, 기존 S_X7에서 번호만 변경)**

> Figure S_X8. Candidate turnover behind Figure 5b. Figure 5b plots analog vs. digital rank only for documents that remain candidates under both conditions, omitting any that enter or exit the set. For one representative query from a diagnostic pool of 3 queries and 200 documents, Digital returns 134 candidates: 130 shared by all three conditions, 3 lost under No Compensation and 1 under Vth Compensation — each a gap in Figure 5b. No Compensation also adds 11 candidates absent from Digital, and both analog conditions jointly add 51 more; lacking a digital rank, none of these can appear in Figure 5b. Consequently, Figure 5b's ΔRank statistic (σ = 18.2 vs. 47.9), computed only over documents with a defined analog rank, understates No Compensation's true disruption. Candidate-set size also grows without compensation (134 → 184 → 193, consistent across all 3 queries), adding to MaxSim's downstream cost.
> \* D = Digital, VC = Vth compensation, NC = No compensation.

- 그림 자체는 `plot_q2_candidate_bar_seed19_v2.py`의 `[seed19]q2_candidate_bar_v2_legend.png` (라벨은 최종적으로 원래 `dig ∩ A ∩ C` 표기로 원복 — 아래 6.8 참고)

### 6.8 범례 라벨 실험 (`plot_q2_candidate_bar_seed19_v2.py`) — 최종적으로 원복

- `dig`→D, `A`→Vth comp, `C`→No comp로 풀어쓰면 너무 길어서, 약어(D/VC/NC) + 캡션, 박스 안에 캡션 포함, 행 간격 확대 + 구분선 등 여러 버전을 시도
- 사용자가 최종적으로 **원래 표기(`dig ∩ A ∩ C` 등, 약어 없음, 캡션 없음, plain `ax.legend()`)로 원복** — 현재 `[seed19]q2_candidate_bar_v2_legend.png`는 이 세션 이전과 동일한 상태
- 약어(D/VC/NC) 자체는 Supplementary 텍스트(S_X7/S_X8, 6.7 참고)에는 계속 사용 중 — 그림 라벨과 논문 텍스트의 표기가 다르다는 점 유의

### 6.9 챕터 제목 후보 (미확정 — 뒷부분 완성 후 결정 예정)

이 챕터는 상위 섹션의 일부이며, 상위 섹션에는 이후 (1) retrieval quality (본 챕터), (2) GPU 대비 TOPS/W, (3) digital quantization 대비 에너지효율, (4) scalability(digital vs analog) 4개 서브섹션이 들어갈 예정.

상위 제목 후보:
- System-Level Performance and Efficiency Projections
- System-Level Evaluation: Retrieval Quality, Efficiency, and Scalability
- Projected System-Level Impact of the AIMDC

("System-level performance projection against GPU baselines"는 GPU 비교만 함의해서 4개 서브섹션을 다 포괄하기엔 좁다고 판단, 보류)
