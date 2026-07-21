import numpy as np

SPEED_RATIO = 1e-6 / 5e-9
P_DRIVER = 0.45e-3
P_TIA    = 9e-3 / SPEED_RATIO
P_MLBUF  = 42e-3 / SPEED_RATIO
E_ADC    = 1.5e-12

N_COLS = 128
N_TOKENS = 32
T_SEARCH  = 1e-6
E_SEARCH_CELL_A = 656.90e-15

def scenario(N_coarse, N_cand_A, gpu_uJ):
    N_rows_total = N_coarse + N_cand_A
    E_cell = N_rows_total * N_TOKENS * E_SEARCH_CELL_A
    E_rwl  = P_DRIVER * T_SEARCH * N_rows_total * N_TOKENS
    E_sbl  = P_DRIVER * T_SEARCH * N_COLS * N_TOKENS
    E_driver = E_rwl + E_sbl
    E_tia  = P_TIA    * T_SEARCH * N_rows_total * N_TOKENS
    E_mlb  = P_MLBUF  * T_SEARCH * N_rows_total * N_TOKENS
    E_adc  = E_ADC    * N_rows_total * N_TOKENS

    gpu = gpu_uJ * 1e-6
    rows = []
    cum = E_cell
    rows.append(("Cell only", cum, gpu/cum))
    cum += E_tia
    rows.append(("+ TIA", cum, gpu/cum))
    cum += E_adc
    rows.append(("+ ADC", cum, gpu/cum))
    cum += E_driver
    rows.append(("+ Driver", cum, gpu/cum))
    cum += E_mlb
    rows.append(("+ ML buffer (final)", cum, gpu/cum))
    return rows

print(f"{'Scenario':22s} {'Energy(nJ)':>14s} {'Energy(uJ)':>12s} {'GPU/analog':>12s}")
print("--- (i) Prototype scale (GPU batch=256 = 200.6 uJ) ---")
for name, e, ratio in scenario(100, 184, 200.6):
    print(f"{name:22s} {e*1e9:14.4f} {e*1e6:12.4f} {ratio:11.1f}x")

print("\n--- (ii) System-level scale (GPU batch=256 = 3949.7 uJ) ---")
for name, e, ratio in scenario(2048, 10988, 3949.7):
    print(f"{name:22s} {e*1e9:14.4f} {e*1e6:12.4f} {ratio:11.1f}x")
