import pandas as pd

path = "/mnt/c/Users/nmdl-khb/ColBERT/centroidset 만들기/colbert_2bit_correct - 복사본.xlsx"
xl = pd.ExcelFile(path)
print("Sheets:", xl.sheet_names)
for sheet in xl.sheet_names:
    df = pd.read_excel(path, sheet_name=sheet)
    print(f"\n--- Sheet: {sheet} ---")
    print(f"Shape: {df.shape}")
    print(df.to_string())
