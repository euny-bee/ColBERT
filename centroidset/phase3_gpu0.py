"""GPU0: digital(0~499) 완료 후 → analog(250~499) 자동 실행"""
import subprocess, sys, os

python = sys.executable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
base = r'C:\Users\nmdl-khb\ColBERT\centroidset'

print("=== GPU0: Step 1 - Digital (0~499) ===")
subprocess.run([python, f'{base}/phase3_digital.py'], check=True)

print("\n=== GPU0: Step 2 - Analog (128~254) ===")
subprocess.run([python, f'{base}/phase3_analog.py', '0', '128', '255'], check=True)

print("\nGPU0 모든 작업 완료!")
