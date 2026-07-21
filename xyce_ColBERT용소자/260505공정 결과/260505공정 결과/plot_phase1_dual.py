"""Phase1(precharge) dual Xyce 결과 플롯: V(SN) vs time, 정착시간 확인용."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('phase1_dual_die4.cir.csv')
df = df.loc[:, ~df.columns.duplicated()]  # TIME 중복 컬럼 제거
df = df[df['TIME'] > 0]  # t=0 DC operating-point 아티팩트 제외
t_us = df['TIME'] * 1e6
sn = df['V(SN)']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(t_us, sn, color='steelblue', lw=2)
axes[0].axhline(0, color='gray', ls='--', lw=1, alpha=0.7, label='VML = 0V')
axes[0].set_xlabel('Time [us]')
axes[0].set_ylabel('V(SN) [V]')
axes[0].set_title('Phase1 precharge: full window (0-2us)')
axes[0].legend()
axes[0].grid(True, ls='--', alpha=0.4)

# 초반 확대 (settling 디테일)
mask = t_us <= 0.05
axes[1].plot(t_us[mask], sn[mask], color='steelblue', lw=2, marker='o', ms=3)
axes[1].axhline(0, color='gray', ls='--', lw=1, alpha=0.7, label='VML = 0V')
axes[1].set_xlabel('Time [us]')
axes[1].set_ylabel('V(SN) [V]')
axes[1].set_title('Phase1 precharge: zoom (0-0.05us)')
axes[1].legend()
axes[1].grid(True, ls='--', alpha=0.4)

fig.suptitle('Phase1 (dual, die4): SN settling toward VML=0V  (SN_init=-1.5V)')
fig.tight_layout()
fig.savefig('phase1_dual_settling.png', dpi=150)
print("Saved: phase1_dual_settling.png")

# 정착시간 추정: |SN| < 1mV 이후 유지되는 첫 시점
tol = 1e-3
below = sn.abs() < tol
if below.any():
    idx = below.idxmax()
    print(f"t_settle (|SN|<{tol*1e3:.0f}mV) = {t_us.iloc[idx]:.4f} us")
print(f"SN at t=2us (final): {sn.iloc[-1]*1e6:.2f} uV")
