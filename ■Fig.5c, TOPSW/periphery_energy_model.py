"""
주변회로(driver/TIA/ML buffer/ADC) 에너지를 우리 Option A/C 코어 셀 에너지에 추가.
참고자료(energy_calcul_revised_v2.xlsx, Revised_v2 시트)의 per-component 전력을 그대로 인용하되,
on-time은 우리 자체 회로에서 실측/검증한 값(write=3.2ms(A)/0.2us(C), search=1us)을 사용.
"""
import numpy as np

# ---- 참고자료에서 인용한 주변회로 전력/에너지 (기술 자체는 범용 citation) ----
P_DRIVER = 0.45e-3   # W, WWL/WBL/SBL/RWL 공통
P_TIA    = 9e-3       # W
P_MLBUF  = 42e-3      # W
E_ADC    = 1.5e-12    # J per conversion (8-bit, 45nm)

N_COLS = 128
N_TOKENS = 32

# ---- 우리 자체 검증된 타이밍 ----
T_WRITE_A = 3200.1e-6   # s, Option A (Phase1+Phase2 worst-case)
T_WRITE_C = 0.2e-6      # s, Option C (Phase1만)
T_SEARCH  = 1e-6        # s

# ---- 코어 셀 에너지 (이미 계산된 값) ----
E_WRITE_CELL_A = 923.46e-15   # J per cell (coarse table)
E_WRITE_CELL_C = 0.0072e-15
E_SEARCH_CELL_A = 656.90e-15  # J per (row,token) 연산
E_SEARCH_CELL_C = 607.78e-15

def compute(scale_name, N_coarse, N_cand_A, N_cand_C):
    print(f"\n{'='*70}\n{scale_name}  (coarse={N_coarse}, cand_A={N_cand_A}, cand_C={N_cand_C})\n{'='*70}")

    # ===== WRITE (one-time, coarse table 기준) =====
    for opt, N_cand, T_write, E_cell in [('A', N_cand_A, T_WRITE_A, E_WRITE_CELL_A),
                                          ('C', N_cand_C, T_WRITE_C, E_WRITE_CELL_C)]:
        N_cells = N_coarse * N_COLS
        E_cell_total = N_cells * E_cell
        E_wwl = P_DRIVER * T_write * N_coarse      # row driver, 1/row
        E_wbl = P_DRIVER * T_write * N_COLS         # col driver, 1/col
        E_write_periphery = E_wwl + E_wbl
        E_write_full = E_cell_total + E_write_periphery
        print(f"\n[Write, Option {opt}] cell={E_cell_total*1e9:.4f}nJ  "
              f"+WWL={E_wwl*1e9:.4f}nJ +WBL={E_wbl*1e9:.4f}nJ  "
              f"periphery={E_write_periphery*1e9:.4f}nJ  TOTAL={E_write_full*1e9:.4f}nJ  "
              f"(periphery/cell = {E_write_periphery/E_cell_total:,.1f}x)")

    # ===== SEARCH (per query) =====
    for opt, N_cand, E_cell in [('A', N_cand_A, E_SEARCH_CELL_A), ('C', N_cand_C, E_SEARCH_CELL_C)]:
        N_rows_total = N_coarse + N_cand   # coarse search rows + scoring rows, 32 token 씩 반복
        # cell energy (coarse + scoring), 이미 계산된 형태 재현
        E_cell_total = (N_coarse + N_cand) * N_TOKENS * E_cell

        E_rwl  = P_DRIVER * T_SEARCH * N_rows_total * N_TOKENS   # row enable, 매 토큰마다
        E_sbl  = P_DRIVER * T_SEARCH * N_COLS * N_TOKENS         # V2 column driver, 매 토큰마다
        E_tia  = P_TIA    * T_SEARCH * N_rows_total * N_TOKENS   # TIA per row per token
        E_mlb  = P_MLBUF  * T_SEARCH * N_rows_total * N_TOKENS   # ML buffer per row per token
        E_adc  = E_ADC    * N_rows_total * N_TOKENS              # ADC conversion per row per token

        E_driver = E_rwl + E_sbl
        E_periphery = E_driver + E_tia + E_mlb + E_adc
        E_full = E_cell_total + E_periphery

        print(f"\n[Search, Option {opt}] cell={E_cell_total*1e9:.4f}nJ")
        print(f"   +driver(RWL+SBL)={E_driver*1e9:.4f}nJ  +TIA={E_tia*1e9:.4f}nJ  "
              f"+MLbuf={E_mlb*1e9:.4f}nJ  +ADC={E_adc*1e9:.4f}nJ")
        print(f"   periphery_total={E_periphery*1e9:.4f}nJ  TOTAL={E_full*1e9:.4f}nJ  "
              f"(periphery/cell = {E_periphery/E_cell_total:,.1f}x)")

        # 시나리오별 누적 (참고자료의 9번 섹션 스타일)
        print(f"   -- scenario breakdown --")
        cum = E_cell_total
        print(f"      cell only        : {cum*1e9:12.4f} nJ")
        cum += E_tia
        print(f"      + TIA            : {cum*1e9:12.4f} nJ")
        cum += E_adc
        print(f"      + ADC            : {cum*1e9:12.4f} nJ")
        cum += E_driver
        print(f"      + driver         : {cum*1e9:12.4f} nJ")
        cum += E_mlb
        print(f"      + ML buffer(full): {cum*1e9:12.4f} nJ   <- final")

compute("(i) Prototype scale", 100, 184, 193)
compute("(ii) System-level scale", 2048, 10988, 10292)
