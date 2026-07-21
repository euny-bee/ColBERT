"""
주변회로 에너지 -- TIA/ML buffer 전력을 우리 속도(1us)에 맞게 스케일링.
속도비 = 1us / 5ns(참고자료) = 200x 더 느려도 됨 -> 전력도 대략 1/200으로 낮출 수 있다고 가정
(analog 회로는 대역폭-전력이 대략 비례한다는 일반적인 설계 경향 기반 근사).
Option A만 고려.
"""
import numpy as np

SPEED_RATIO = 1e-6 / 5e-9   # = 200

P_DRIVER = 0.45e-3    # W, 그대로 유지
P_TIA    = 9e-3 / SPEED_RATIO      # 45 uW
P_MLBUF  = 42e-3 / SPEED_RATIO     # 210 uW
E_ADC    = 1.5e-12    # J per conversion, 그대로 유지

print(f"Scaled P_TIA = {P_TIA*1e6:.2f} uW,  P_MLBUF = {P_MLBUF*1e6:.2f} uW  (speed ratio={SPEED_RATIO:.0f}x)")

N_COLS = 128
N_TOKENS = 32

T_WRITE_A = 3200.1e-6
T_SEARCH  = 1e-6

E_WRITE_CELL_A = 923.46e-15
E_SEARCH_CELL_A = 656.90e-15

def compute(scale_name, N_coarse, N_cand_A):
    print(f"\n{'='*70}\n{scale_name}  (coarse={N_coarse}, cand_A={N_cand_A})\n{'='*70}")

    # WRITE
    N_cells = N_coarse * N_COLS
    E_cell_total = N_cells * E_WRITE_CELL_A
    E_wwl = P_DRIVER * T_WRITE_A * N_coarse
    E_wbl = P_DRIVER * T_WRITE_A * N_COLS
    E_write_periphery = E_wwl + E_wbl
    E_write_full = E_cell_total + E_write_periphery
    print(f"\n[Write] cell={E_cell_total*1e9:.4f}nJ  periphery(driver only)={E_write_periphery*1e9:.4f}nJ  "
          f"TOTAL={E_write_full*1e9:.4f}nJ")

    # SEARCH
    N_rows_total = N_coarse + N_cand_A
    E_cell_total_s = N_rows_total * N_TOKENS * E_SEARCH_CELL_A

    E_rwl  = P_DRIVER * T_SEARCH * N_rows_total * N_TOKENS
    E_sbl  = P_DRIVER * T_SEARCH * N_COLS * N_TOKENS
    E_tia  = P_TIA    * T_SEARCH * N_rows_total * N_TOKENS
    E_mlb  = P_MLBUF  * T_SEARCH * N_rows_total * N_TOKENS
    E_adc  = E_ADC    * N_rows_total * N_TOKENS

    E_driver = E_rwl + E_sbl
    E_periphery = E_driver + E_tia + E_mlb + E_adc
    E_full = E_cell_total_s + E_periphery

    print(f"\n[Search] cell={E_cell_total_s*1e9:.4f}nJ")
    print(f"   +driver(RWL+SBL)={E_driver*1e9:.4f}nJ  +TIA(scaled)={E_tia*1e9:.4f}nJ  "
          f"+MLbuf(scaled)={E_mlb*1e9:.4f}nJ  +ADC={E_adc*1e9:.4f}nJ")
    print(f"   periphery_total={E_periphery*1e9:.4f}nJ  TOTAL={E_full*1e9:.4f}nJ  ({E_full*1e6:.4f}uJ)")
    print(f"   (periphery/cell = {E_periphery/E_cell_total_s:,.1f}x)")

    cum = E_cell_total_s
    print(f"   -- scenario breakdown --")
    print(f"      cell only          : {cum*1e9:12.4f} nJ")
    cum += E_tia
    print(f"      + TIA(scaled)      : {cum*1e9:12.4f} nJ")
    cum += E_adc
    print(f"      + ADC              : {cum*1e9:12.4f} nJ")
    cum += E_driver
    print(f"      + driver           : {cum*1e9:12.4f} nJ")
    cum += E_mlb
    print(f"      + ML buffer(scaled): {cum*1e9:12.4f} nJ   <- final ({cum*1e6:.4f} uJ)")
    return E_full

e_proto = compute("(i) Prototype scale", 100, 184)
e_sys   = compute("(ii) System-level scale", 2048, 10988)

print(f"\n\n=== GPU(batch=256, 실측) 대비 비교 ===")
gpu_proto = 200.6e-6   # J, RTX 3070Ti, Vth comp, prototype
gpu_sys   = 3949.7e-6  # J, RTX 3070Ti, Vth comp, system-level
print(f"Prototype: analog(full periphery)={e_proto*1e6:.2f}uJ vs GPU={gpu_proto*1e6:.2f}uJ -> ratio={gpu_proto/e_proto:.4f}x (GPU/analog, <1 means analog wins)")
print(f"System   : analog(full periphery)={e_sys*1e6:.2f}uJ vs GPU={gpu_sys*1e6:.2f}uJ -> ratio={gpu_sys/e_sys:.4f}x")
print(f"(analog가 유리하면 analog 에너지가 더 작아야 함 -> gpu/analog 비율이 1보다 크면 analog 우세)")
