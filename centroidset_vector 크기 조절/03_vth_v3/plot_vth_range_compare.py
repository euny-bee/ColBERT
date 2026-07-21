"""
Vth shift 분포 비교: [0, 0.5]V (현재) vs [0, 3]V (새)
std 비율 유지 vs 자유 선택
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import truncnorm

_avail = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _avail:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

N = 100_000
SEED = 42

configs = [
    # (label, mean, std, lo, hi, color)
    ("현재\nmean=0, std=0.15\nrange=[0, 0.5]V",  0.0, 0.15, 0.0, 0.5,  "#E57373"),
    ("비율 유지\nmean=0, std=0.90\nrange=[0, 3]V", 0.0, 0.90, 0.0, 3.0,  "#64B5F6"),
    ("std=0.5\nrange=[0, 3]V\n(더 가파름)",        0.0, 0.50, 0.0, 3.0,  "#81C784"),
    ("std=1.5\nrange=[0, 3]V\n(더 완만)",          0.0, 1.50, 0.0, 3.0,  "#FFB74D"),
]

fig, axes = plt.subplots(1, len(configs), figsize=(18, 5))
fig.suptitle("Vth shift 분포 비교 — Truncated Gaussian (mean=0, lower=0)", fontsize=12, fontweight='bold')

for ax, (label, mu, std, lo, hi, color) in zip(axes, configs):
    a = (lo - mu) / std
    b = (hi - mu) / std
    samples = truncnorm.rvs(a, b, loc=mu, scale=std, size=N, random_state=SEED)

    # histogram
    bins = np.linspace(lo, hi, 40)
    ax.hist(samples, bins=bins, density=True, color=color, alpha=0.7, label=f'Sampled (n={N//1000}k)')

    # PDF
    x = np.linspace(lo, hi, 500)
    pdf = truncnorm.pdf(x, a, b, loc=mu, scale=std)
    ax.plot(x, pdf, color='navy', lw=2, label='Truncated Gaussian PDF')

    # 통계
    smean = float(samples.mean())
    sp50  = float(np.percentile(samples, 50))
    ax.axvline(lo,   color='black', lw=1.2, ls='--', alpha=0.6, label=f'Min = {lo}V')
    ax.axvline(hi,   color='gray',  lw=1.0, ls=':',  alpha=0.6, label=f'Max = {hi}V')
    ax.axvline(smean, color='orange', lw=1.8, ls='-', label=f'mean = {smean:.3f}V')

    stats_txt = (f"mean  = {smean:.3f}V\n"
                 f"median= {sp50:.3f}V\n"
                 f"P(<{hi*0.1:.1f}V)= {100*(samples<hi*0.1).mean():.1f}%\n"
                 f"P(<{hi*0.33:.1f}V)= {100*(samples<hi/3).mean():.1f}%\n"
                 f"P(<{hi*0.67:.1f}V)= {100*(samples<hi*2/3).mean():.1f}%")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title(label, fontsize=9, fontweight='bold')
    ax.set_xlabel("Vth shift [V]", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_xlim(lo - hi*0.02, hi * 1.02)

plt.tight_layout()
out = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\03_vth_v3\plot_vth_range_compare.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

# ── 같은 y축 범위 비교: x를 [0,1]로 정규화 ──────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Vth shift 분포 — x축 [0,1] 정규화 후 shape 비교\n(같은 y축 범위: 모양이 완전히 동일함을 확인)",
              fontsize=12, fontweight='bold')

colors2 = ["#E57373", "#64B5F6", "#81C784", "#FFB74D"]
labels2 = ["현재  std=0.15, [0,0.5]V",
           "비율유지 std=0.90, [0,3]V",
           "std=0.50, [0,3]V",
           "std=1.50, [0,3]V"]

x_norm = np.linspace(0, 1, 500)

# 왼쪽: 정규화된 PDF 오버레이
ax = axes2[0]
for (label, mu, std, lo, hi, _), color, lbl in zip(configs, colors2, labels2):
    a = (lo - mu) / std
    b = (hi - mu) / std
    # x_norm ∈ [0,1] → 실제 값: lo + x_norm*(hi-lo)
    x_real = lo + x_norm * (hi - lo)
    pdf_real = truncnorm.pdf(x_real, a, b, loc=mu, scale=std)
    # 정규화된 x에서의 pdf: pdf_norm = pdf_real * (hi - lo)
    pdf_norm = pdf_real * (hi - lo)
    ax.plot(x_norm, pdf_norm, lw=2.5, color=color, label=lbl)

ax.set_xlabel("Vth shift (정규화, 0=min, 1=max)", fontsize=10)
ax.set_ylabel("Density (정규화)", fontsize=10)
ax.set_title("PDF shape 비교 (x 정규화)", fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, ls='--', alpha=0.3)
ax.set_xlim(0, 1)

# 오른쪽: 실제 x축, 같은 y범위 (ylim 고정)
ax2 = axes2[1]
x_full = np.linspace(0, 3.05, 1000)
max_pdf = 0
for label, mu, std, lo, hi, color in configs:
    a = (lo - mu) / std
    b = (hi - mu) / std
    pdf = truncnorm.pdf(x_full, a, b, loc=mu, scale=std)
    max_pdf = max(max_pdf, pdf.max())

for (label, mu, std, lo, hi, _), color, lbl in zip(configs, colors2, labels2):
    a = (lo - mu) / std
    b = (hi - mu) / std
    pdf = truncnorm.pdf(x_full, a, b, loc=mu, scale=std)
    ax2.plot(x_full, pdf, lw=2.5, color=color, label=lbl)

ax2.set_xlabel("Vth shift [V]", fontsize=10)
ax2.set_ylabel("Density", fontsize=10)
ax2.set_title("실제 x축, 같은 y범위\n→ 넓은 분포는 낮게 보임", fontsize=10, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, ls='--', alpha=0.3)
ax2.set_xlim(0, 3.05)
ax2.set_ylim(0, max_pdf * 1.05)

plt.tight_layout()
out2 = r'C:\Users\nmdl-khb\ColBERT\centroidset_vector 크기 조절\03_vth_v3\plot_vth_shape_compare.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")

# 수치 요약
print("\n" + "="*55)
for label, mu, std, lo, hi, _ in configs:
    a = (lo - mu) / std
    b = (hi - mu) / std
    s = truncnorm.rvs(a, b, loc=mu, scale=std, size=N, random_state=SEED)
    print(f"\n{label.replace(chr(10), ' | ')}")
    print(f"  mean={s.mean():.3f}V  std={s.std():.3f}V  max={s.max():.3f}V")
    for thr in [hi*0.1, hi/3, hi*2/3]:
        print(f"  P(shift < {thr:.2f}V) = {100*(s<thr).mean():.1f}%")
