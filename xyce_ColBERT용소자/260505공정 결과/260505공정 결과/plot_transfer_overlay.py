import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm

# 한글 폰트 설정 (맑은 고딕 우선, 없으면 기본)
_korean_fonts = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]
_available = [f.name for f in fm.fontManager.ttflist]
for _fn in _korean_fonts:
    if _fn in _available:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_b1500_csv(filepath):
    """B1500A CSV에서 DataValue 행만 파싱해 gate, IG, ID 반환."""
    gate, ig, id_ = [], [], []
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("DataValue"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                gate.append(float(parts[1]))
                ig.append(float(parts[2]))
                id_.append(float(parts[3]))
            except ValueError:
                continue
    return np.array(gate), np.array(ig), np.array(id_)

def die_label(filename):
    """파일명에서 die 레이블 추출 (예: die1_ovl20_R1)."""
    m = re.search(r"(die\d+_ovl\d+_R\d+)", filename)
    return m.group(1) if m else filename

# ── 파일 수집 ────────────────────────────────────────────────────────────────
csv_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f.startswith("IGZO_TR")],
    key=lambda f: (int(re.search(r"die(\d+)", f).group(1)),
                   int(re.search(r"_R(\d+)", f).group(1)))
)
print(f"Found {len(csv_files)} files:")
for f in csv_files:
    print(f"  {f}")

# ── 색상 설정 ────────────────────────────────────────────────────────────────
colors = cm.tab10(np.linspace(0, 1, len(csv_files)))

# ── Figure: ID overlay ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("IGZO TFT Transfer Curve — 260505 공정 (VDS = 1 V)", fontsize=13)

ax_log = axes[0]
ax_lin = axes[1]

for i, fname in enumerate(csv_files):
    fpath = os.path.join(DATA_DIR, fname)
    gate, ig, id_ = parse_b1500_csv(fpath)
    label = die_label(fname)
    color = colors[i]

    # |ID| (양수만 표시, 노이즈 플로어 클립)
    id_abs = np.abs(id_).clip(1e-14)

    ax_log.semilogy(gate, id_abs, color=color, linewidth=1.5, label=label)
    ax_lin.plot(gate, id_abs * 1e6, color=color, linewidth=1.5, label=label)

# ── Log scale 축 설정 ────────────────────────────────────────────────────────
ax_log.set_xlabel("V$_{GS}$ (V)", fontsize=12)
ax_log.set_ylabel("|I$_{D}$| (A)", fontsize=12)
ax_log.set_title("Log scale", fontsize=11)
ax_log.set_xlim(-3, 3)
ax_log.set_ylim(1e-13, 1e-3)
ax_log.grid(True, which="both", linestyle="--", alpha=0.4)
ax_log.legend(fontsize=8, loc="upper left")

# ── Linear scale 축 설정 ─────────────────────────────────────────────────────
ax_lin.set_xlabel("V$_{GS}$ (V)", fontsize=12)
ax_lin.set_ylabel("|I$_{D}$| (μA)", fontsize=12)
ax_lin.set_title("Linear scale", fontsize=11)
ax_lin.set_xlim(-3, 3)
ax_lin.grid(True, linestyle="--", alpha=0.4)
ax_lin.legend(fontsize=8, loc="upper left")

plt.tight_layout()
out = os.path.join(DATA_DIR, "transfer_overlay_260505.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"\nSaved: {out}")

# ── 소자별 Ion, Ioff, on/off ratio 출력 ─────────────────────────────────────
print("\n{:<22} {:>12} {:>12} {:>12}".format("Die", "Ion (A)", "Ioff (A)", "Ion/Ioff"))
print("-" * 62)
for fname in csv_files:
    fpath = os.path.join(DATA_DIR, fname)
    gate, ig, id_ = parse_b1500_csv(fpath)
    label = die_label(fname)
    id_abs = np.abs(id_)
    ion  = id_abs[gate >= 2.5].mean() if np.any(gate >= 2.5) else np.max(id_abs)
    ioff = id_abs[gate <= -2.0].mean() if np.any(gate <= -2.0) else np.min(id_abs)
    ratio = ion / ioff if ioff > 0 else float("inf")
    print(f"{label:<22} {ion:>12.3e} {ioff:>12.3e} {ratio:>12.2e}")
