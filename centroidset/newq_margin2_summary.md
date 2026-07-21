# 논문 Figure 작업 요약 (2026-07-02)

---

## 1. table_optC_owncandidate_v2.png 수정

**스크립트**: `논문 figure/plot_table_optC_owncandidate_v2.py`

- `TABLE_RIGHT = 0.86` 도입 → 수평선을 표 우측 경계까지만 그려 오른쪽 빈 공간 제거
- `hline()` 기본 x1을 TABLE_RIGHT로 변경 (`transform=ax.transAxes` 때문에 bbox crop이 안 되던 문제 해결)
- `draw_section()` 중심을 `TABLE_RIGHT/2`로 변경
- `COL_X[1]` 조정 시도 후 원복 (최종 `[0.02, 0.22]` 유지)
- 폰트: Times New Roman serif 유지
- `subplots_adjust` 적용 시도 → 원복 (tight_layout 유지)

---

## 2. [graph_optC]option1_single.png 수정

**스크립트**: `논문 figure/plot_optC_degradation_all.py`

- 4면 박스 추가 (`for spine in ax.spines.values(): spine.set_visible(True)`)
- x축 레이블에서 v1/v2/v3/v4 제거: `"[0, 0.5] V"` 형식으로 변경
- tick fontsize 14→20, axis label fontsize 16→23 (20% × 2회 증가)
- `xlim(-0.4, 3.4)` → 첫 tick이 y축에 붙는 문제 해결

---

## 3. R@50 비교 figure 신규 생성

**스크립트**: `논문 figure/plot_r50_final.py`  
**출력**: `[graph_optC]r50_final.png`, `[graph_optC]r50_legend.png`

### 설계
- Vth Compensation (This Work, 파란색 굵은선) vs No Compensation (빨간색 얇은선)
- `fill_between`으로 손실 영역 빨간 음영 표시
- Option 2 스타일: lw_comp=4.5 / ms_comp=12 vs lw_nocomp=1.5 / ms_nocomp=7
- 레전드 별도 파일 분리 (검정 테두리 1.5pt, bbox.padded(0.05))
- "Robust to PBS-induced Vth variation" 텍스트 파란선 위 이탤릭 표시
- "Vth Compensation\n(This Work)" 줄바꿈으로 세로 레전드

### 주요 설정값
```python
figsize     = (5.9, 4.6)
xlim        = (-0.15, 3.15)
ylim        = (0, 112)
xlabel      = "PBS-Induced Vth Shift Range [V]"   # x tick에서 V 제거, 제목에 [V] 추가
ylabel      = "R@50 (%)"
labelpad_x  = 8
labelpad_y  = -6
x tick labels: ["[0, 0.5]", "[0, 1.0]", "[0, 2.0]", "[0, 3.0]"]
```

### 데이터
| 조건 | Vth Compensation | No Compensation |
|---|---|---|
| [0, 0.5] V | 98.8 | 98.8 |
| [0, 1.0] V | 98.8 | 94.1 |
| [0, 2.0] V | 98.8 | 58.8 |
| [0, 3.0] V | 98.8 | 23.1 |

---

## 4. MRR@10 비교 figure 신규 생성

**스크립트**: `논문 figure/plot_mrr_final.py`  
**출력**: `[graph_optC]mrr_final.png`, `[graph_optC]mrr_legend.png`

- R@50 figure와 동일한 스타일
- `ylim=(0, 92)` — 데이터 최대값(78.8%) 기준으로 상단 여백 제거
- `labelpad=4` — tick 숫자가 2자리(최대 80)라 r50보다 간격 넓게 조정
- "Robust to PBS-induced Vth variation" 텍스트 없음

### 데이터
| 조건 | Vth Compensation | No Compensation |
|---|---|---|
| [0, 0.5] V | 78.8 | 78.1 |
| [0, 1.0] V | 78.8 | 71.3 |
| [0, 2.0] V | 78.8 | 27.8 |
| [0, 3.0] V | 78.8 |  2.1 |

---

## 5. R@50 강조 방식 variant 4종 생성

**스크립트**: `논문 figure/plot_r50_comparison_variants.py`

| 파일 | 방식 |
|---|---|
| `[graph_optC]r50_opt1_label.png` | "This Work" 직접 라벨 + 화살표 |
| `[graph_optC]r50_opt2_thickness.png` | 선 굵기·마커 차별화 |
| `[graph_optC]r50_opt3_band.png` | 파란 수평 밴드 (94~101.5%) |
| `[graph_optC]r50_opt4_star.png` | This Work 마커 ★ (zorder=6) |

→ 최종 채택: **Option 2 스타일** (`r50_final.py`에 적용)

---

## 6. 기타 정리

- `plot_combined_final.py` (R@50+MRR@10 혼합 figure): 미채택, 삭제 예정
- `vth_comp_vs_nocomp_summary.md` Section 6~7 업데이트 (figure 목록, 스타일 기준)
