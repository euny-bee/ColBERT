"""
plot_fig5.py — fig5_norm_distribution: ||c+r||² broken-axis histogram

Data  : colbert_float16_correct.xlsx + colbert_2bit_correct.csv (same dir)
Output: fig5_norm_distribution.png (300 dpi) in the same directory
Style : matches [vth_v3]row0_scatter.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
F16_XLSX   = SCRIPT_DIR / "colbert_float16_correct.xlsx"
BIT2_CSV   = SCRIPT_DIR / "colbert_2bit_correct.csv"
OUTPUT     = SCRIPT_DIR / "fig5_norm_distribution.png"

DIM           = 128
CENTROID_COLS = [f"centroid_dim_{i}" for i in range(DIM)]
RESIDUAL_COLS = [f"residual_dim_{i}" for i in range(DIM)]

# ── Style (matches [vth_v3]row0_scatter.png) ─────────────────────────────────
FONT_BASE  = 16
FONT_PANEL = 18
FONT_LABEL = 22
FONT_TICK  = 18
FONT_LEG   = 17
GRID_ALPHA = 0.3

COLOR_F16  = "#4878CF"
COLOR_2BIT = "#E8604C"
ALPHA_HIST = 0.75


def _setup_matplotlib() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    sans = [f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    sans,
        "font.size":          FONT_BASE,
        "font.weight":        "bold",
        "axes.labelsize":     FONT_LABEL,
        "axes.labelweight":   "bold",
        "axes.titlesize":     FONT_PANEL,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    FONT_TICK,
        "ytick.labelsize":    FONT_TICK,
        "axes.unicode_minus": False,
    })


def _style_ax(ax) -> None:
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8, ls="--")
    ax.minorticks_off()


def load_data():
    print("Loading colbert_float16_correct.xlsx ...")
    df_f16  = pd.read_excel(F16_XLSX)
    print("Loading colbert_2bit_correct.csv ...")
    df_2bit = pd.read_csv(BIT2_CSV, encoding="cp949")

    centroid      = df_f16[CENTROID_COLS].to_numpy(dtype=np.float32)
    residual_f16  = df_f16[RESIDUAL_COLS].to_numpy(dtype=np.float32)
    residual_2bit = df_2bit[RESIDUAL_COLS].to_numpy(dtype=np.float32)

    v_f16  = centroid + residual_f16
    v_2bit = centroid + residual_2bit

    n = len(v_f16)
    print(f"  {n:,} tokens loaded")
    return v_f16, v_2bit, n


def plot_fig5(v_f16: np.ndarray, v_2bit: np.ndarray, n: int,
              font_label: int = FONT_LABEL, font_tick: int = FONT_TICK,
              output: Path = OUTPUT) -> None:
    norm_f16  = (v_f16  ** 2).sum(axis=1)
    norm_2bit = (v_2bit ** 2).sum(axis=1)
    mean_2bit = float(norm_2bit.mean())

    BINS  = 100
    X_MIN = 0.5
    X_MAX = 1.1

    # density 미리 계산 (제목용)
    counts_f16, _ = np.histogram(norm_f16, bins=BINS, range=(X_MIN, X_MAX), density=True)
    max_f16 = float(counts_f16.max())

    # ── y축 고정 ──────────────────────────────────────────────────────────────
    BOT_YLIM   = (0, 6.5)
    BOT_YTICKS = [0, 2, 4, 6]
    TOP_YLIM   = (163, 169)
    TOP_YTICKS = [166, 168]

    # ── 두 패널: y 범위 비율에 맞게 높이 설정 ───────────────────────────────
    top_range = TOP_YLIM[1] - TOP_YLIM[0]   # 169-163 = 6
    bot_range = BOT_YLIM[1] - BOT_YLIM[0]   # 6.5-0   = 6.5
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 7),
        gridspec_kw={"height_ratios": [top_range, bot_range], "hspace": 0},
    )

    for ax in (ax_top, ax_bot):
        ax.hist(norm_2bit, bins=BINS, range=(X_MIN, X_MAX), density=True,
                alpha=ALPHA_HIST, color=COLOR_2BIT, edgecolor="none",
                label=f"2-bit  (mean = {mean_2bit:.4f})")
        ax.hist(norm_f16,  bins=BINS, range=(X_MIN, X_MAX), density=True,
                alpha=0.75, color=COLOR_F16, edgecolor="none",
                label=f"float16 = 1.0  (δ-function, density ≈ {int(max_f16)})")
        ax.axvline(mean_2bit, color=COLOR_2BIT, linestyle="--", linewidth=2.0,
                   label=f"2-bit mean = {mean_2bit:.4f}")
        ax.axvline(1.0, color=COLOR_F16, linestyle="--", linewidth=2.0,
                   label="no quant mean = 1.0000")

    ax_top.set_ylim(*TOP_YLIM)
    ax_bot.set_ylim(*BOT_YLIM)
    ax_top.set_yticks(TOP_YTICKS)
    ax_bot.set_yticks(BOT_YTICKS)

    # _style_ax 먼저 → 그 다음 spine 숨기기 (순서 중요)
    _style_ax(ax_top)
    _style_ax(ax_bot)
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    # x축
    ax_bot.set_xticks(np.arange(X_MIN, X_MAX + 0.05, 0.1))
    ax_bot.set_xlim(X_MIN, X_MAX)
    ax_bot.set_xlabel("||Centroid + Residual||²", fontsize=font_label)
    ax_bot.tick_params(axis="x", labelsize=font_tick)
    ax_top.tick_params(axis="y", labelsize=font_tick)
    ax_bot.tick_params(axis="y", labelsize=font_tick)

    # y=0 라벨이 x축 "0.5"와 겹치지 않도록: 기본 라벨을 숨기고 위로 살짝 띄워서 다시 그림
    for tick, val in zip(ax_bot.yaxis.get_major_ticks(), BOT_YTICKS):
        if val == 0:
            tick.label1.set_visible(False)
    ax_bot.annotate("0", xy=(0, 0), xycoords=("axes fraction", "data"),
                     xytext=(-8, 6), textcoords="offset points",
                     ha="right", va="center", fontsize=font_tick, fontweight="bold")

    # 제목
    fig.suptitle(
        f"Figure 5: Reconstruction Norm Distribution  [N={n:,} tokens]\n"
        f"(float16: δ at 1.0, density≈{int(max_f16)}  |  2-bit: spread around {mean_2bit:.3f})",
        fontsize=FONT_PANEL, fontweight="bold",
    )

    # tight_layout 먼저 → 이후 실제 위치로 y축 레이블 배치
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.canvas.draw()
    top_pos = ax_top.get_position()
    bot_pos = ax_bot.get_position()
    y_center = (top_pos.y1 + bot_pos.y0) / 2
    x_label  = bot_pos.x0 - 0.10
    fig.text(x_label, y_center, "Density", va="center", ha="center",
             rotation="vertical", fontsize=font_label, fontweight="bold")

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output}")

    # ── 범례 단독 figure ─────────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    handles = [
        mpatches.Patch(color=COLOR_2BIT, alpha=ALPHA_HIST,
                       label=f"2-bit  (mean = {mean_2bit:.4f})"),
        mpatches.Patch(color=COLOR_F16,  alpha=0.75,
                       label=f"float16 = 1.0  (δ-function, density ≈ {int(max_f16)})"),
    ]

    fig_leg, ax_leg = plt.subplots(figsize=(5, 1.6))
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, fontsize=FONT_LEG,
                  title="Quantization Comparison", title_fontsize=FONT_LEG,
                  loc="center", frameon=True)
    out_leg = SCRIPT_DIR / "fig5_legend.png"
    fig_leg.savefig(out_leg, dpi=300, bbox_inches="tight")
    plt.close(fig_leg)
    print(f"  → {out_leg}")

    # ── 범례 v2: 괄호 없이, 이름 변경 ───────────────────────────────────────
    import matplotlib.lines as mlines

    handles_v2 = [
        mpatches.Patch(color=COLOR_2BIT, alpha=ALPHA_HIST,
                       label="2-bit quantized"),
        mpatches.Patch(color=COLOR_F16,  alpha=0.75,
                       label="no quantized"),
        mlines.Line2D([], [], color=COLOR_2BIT, linestyle="--", linewidth=2.0,
                      label=f"2-bit mean = {mean_2bit:.4f}"),
        mlines.Line2D([], [], color=COLOR_F16, linestyle="--", linewidth=2.0,
                      label="no quant mean = 1.0000"),
    ]

    fig_leg2, ax_leg2 = plt.subplots(figsize=(4, 2.2))
    ax_leg2.axis("off")
    ax_leg2.legend(handles=handles_v2, fontsize=FONT_LEG,
                   title="Quantization\nComparison", title_fontsize=FONT_LEG,
                   loc="center", frameon=True)
    out_leg2 = SCRIPT_DIR / "fig5_legend_v2.png"
    fig_leg2.savefig(out_leg2, dpi=300, bbox_inches="tight")
    plt.close(fig_leg2)
    print(f"  → {out_leg2}")

    # ── 범례 v2 no title ─────────────────────────────────────────────────────
    fig_leg3, ax_leg3 = plt.subplots(figsize=(4, 2.2))
    ax_leg3.axis("off")
    ax_leg3.legend(handles=handles_v2, fontsize=FONT_LEG,
                   loc="center", frameon=True)
    out_leg3 = SCRIPT_DIR / "fig5_legend_v2_notitle.png"
    fig_leg3.savefig(out_leg3, dpi=300, bbox_inches="tight")
    plt.close(fig_leg3)
    print(f"  → {out_leg3}")


def main() -> None:
    _setup_matplotlib()
    v_f16, v_2bit, n = load_data()
    print("Plotting Figure 5 ...")
    plot_fig5(v_f16, v_2bit, n)
    print("Plotting Figure 5 (+10% fonts) ...")
    plot_fig5(v_f16, v_2bit, n,
              font_label=int(FONT_LABEL * 1.1),
              font_tick=int(FONT_TICK * 1.1),
              output=SCRIPT_DIR / "fig5_norm_distribution_large10.png")
    print("Plotting Figure 5 (+20% fonts) ...")
    plot_fig5(v_f16, v_2bit, n,
              font_label=int(FONT_LABEL * 1.2),
              font_tick=int(FONT_TICK * 1.2),
              output=SCRIPT_DIR / "fig5_norm_distribution_large20.png")
    print("Done.")


if __name__ == "__main__":
    main()
