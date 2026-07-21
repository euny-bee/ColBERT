import pandas as pd

rows = [
    ["E_phase1 (precharge, Vth 무관)", "0.0072 fJ", "0.0072 fJ", "write에 포함, 둘 다 동일"],
    ["E_phase2 (store, v4 PBS[0,3]V 가중)", "923.45 fJ", "0 fJ (Phase2 없음)", "Option C는 Vth 보상 skip"],
    ["E_write = E_phase1+E_phase2 (셀당 1회, 인덱스 빌드)", "923.46 fJ", "0.0072 fJ", ""],
    ["E_search = E_phase3 (q2 실제 diff 분포 가중, 쿼리당)", "656.90 fJ", "607.78 fJ", "1us sensing"],
]
df = pd.DataFrame(rows, columns=["항목", "Option A (Vth comp)", "Option C (No comp)", "비고"])
print(df.to_string(index=False))
