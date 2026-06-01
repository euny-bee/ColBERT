import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_available = [f.name for f in fm.fontManager.ttflist]
for _fn in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
    if _fn in _available:
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_b1500_csv(filepath, sweep_index=0):
    """sweep_index=0 이면 첫 번째 sweep만 반환."""
    sweeps_gate, sweeps_id = [[]], [[]]
    current = 0
    in_data = False
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("DataName"):
                if in_data:          # 새 sweep 시작
                    current += 1
                    sweeps_gate.append([])
                    sweeps_id.append([])
                in_data = True
                continue
            if not in_data or not line.startswith("DataValue"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                sweeps_gate[current].append(float(parts[1]))
                sweeps_id[current].append(float(parts[3]))
            except ValueError:
                continue
    idx = min(sweep_index, len(sweeps_gate) - 1)
    return np.array(sweeps_gate[idx]), np.array(sweeps_id[idx])

def die_label(filename):
    m = re.search(r"(die\d+_ovl\d+_R\d+)", filename)
    return m.group(1) if m else filename

# die8_R0 제외
all_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f.startswith("IGZO_TR")],
    key=lambda f: (int(re.search(r"die(\d+)", f).group(1)),
                   int(re.search(r"_R(\d+)", f).group(1)))
)
files = [f for f in all_files if "die8_ovl20_R0" not in f]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
fig.suptitle("IGZO TFT Transfer Curve — VDS = 1 V  (die8_R0 제외)", fontsize=13)

for idx, fname in enumerate(files):
    ax = axes.flatten()[idx]
    gate, id_ = parse_b1500_csv(os.path.join(DATA_DIR, fname))
    label = die_label(fname)

    id_abs = np.abs(id_).clip(1e-14)
    ion  = id_abs[gate >= 2.5].mean()
    ioff = id_abs[gate <= -2.0].mean()

    ax.semilogy(gate, id_abs, color="steelblue", linewidth=1.8)
    ax.set_title(label, fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(1e-13, 1e-3)
    ax.set_xlabel("VGS (V)", fontsize=9)
    ax.set_ylabel("|ID| (A)", fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)
    ax.text(0.97, 0.05,
            f"Ion = {ion:.2e}\nIoff = {ioff:.2e}\nRatio = {ion/ioff:.1e}",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
out = os.path.join(DATA_DIR, "transfer_individual_260505.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")
