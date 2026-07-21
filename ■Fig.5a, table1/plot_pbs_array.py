"""
PBS 5x5 IGZO array figures: per-cell I-V grid, aggregate spread, drift quantification.

Reads the sole .xlsx workbook in this directory; saves fig_a/b/c as PNG (300 dpi).
Run from array/: python plot_pbs_array.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR

ARRAY_SIZE = 5
N_CELLS = ARRAY_SIZE * ARRAY_SIZE
ROW_LEN = ARRAY_SIZE

PBS_TIMES_MIN = [0, 1, 10, 60]

Condition = Literal["NoComp", "VthComp"]

# Style
FONT_BASE = 10
FONT_AXIS = 11
FONT_PANEL = 12
FONT_SUPTITLE = 14
COLOR_NOCOMP = "#D95F02"  # warm amber
COLOR_VTHCOMP = "#1B9E77"  # teal
LW_CURVE = 1.2
LW_MEAN = 2.0
ALPHA_ENVELOPE = 0.25
GRID_ALPHA = 0.3

_SHEETS: Dict[str, pd.DataFrame] = {}


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT_BASE,
            "axes.labelsize": FONT_AXIS,
            "axes.titlesize": FONT_PANEL,
            "xtick.labelsize": FONT_BASE,
            "ytick.labelsize": FONT_BASE,
        }
    )


def _style_ax(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.8)
        ax.minorticks_off()


def cell_to_grid(bit_idx: int) -> Tuple[int, int]:
    """Column-major 5x5 layout: bit0 = (0,0), bit5 = (1,0), bit24 = (4,4)."""
    return bit_idx // ROW_LEN, bit_idx % ROW_LEN


def active_cells() -> List[int]:
    return list(range(N_CELLS))


def load_cell(bit_idx: int, condition: Condition) -> pd.DataFrame:
    sheet = f"bit{bit_idx}_{condition}"
    if sheet not in _SHEETS:
        raise KeyError(f"Missing sheet {sheet!r}")
    raw = _SHEETS[sheet]
    if condition == "NoComp":
        # NoComp: single-channel sweep on CH2 (-2 V to +2 V).
        v = raw["CH2 [V]"].to_numpy()
    else:
        # VthComp: differential sweep V1 - V2 (-2 V to +2 V).
        v = raw["V1-V2"].to_numpy()
    return pd.DataFrame(
        {
            "v": v,
            "adc_0": raw["0min_ADC"].to_numpy(),
            "adc_1": raw["1min_ADC"].to_numpy(),
            "adc_10": raw["10min_ADC"].to_numpy(),
            "adc_60": raw["60min_ADC"].to_numpy(),
        }
    )


def _adc_columns(condition: Condition) -> List[str]:
    return ["0min_ADC", "1min_ADC", "10min_ADC", "60min_ADC"]


def _pbs_time_colors() -> Dict[int, tuple]:
    cmap = plt.get_cmap("viridis")
    n = len(PBS_TIMES_MIN)
    return {t: cmap(i / max(n - 1, 1)) for i, t in enumerate(PBS_TIMES_MIN)}


def compute_drift(cell_df: pd.DataFrame) -> float:
    a0 = cell_df["adc_0"].to_numpy(dtype=float)
    a60 = cell_df["adc_60"].to_numpy(dtype=float)
    denom = float(np.mean(a0))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.mean(np.abs(a60 - a0)) / denom)


def _stack_active(condition: Condition, time_min: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (v_ref, adc_stack shape (n_active, n_steps)) for one condition and time."""
    col = f"adc_{time_min}"
    frames = [load_cell(b, condition) for b in active_cells()]
    v_ref = frames[0]["v"].to_numpy()
    stack = np.vstack([f[col].to_numpy(dtype=float) for f in frames])
    for f in frames[1:]:
        if not np.allclose(f["v"].to_numpy(), v_ref):
            raise ValueError(
                f"Voltage grid mismatch for {condition} across active cells"
            )
    return v_ref, stack


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_input_workbook(data_dir: Path) -> Path:
    """Return the sole .xlsx workbook in data_dir (ignoring Excel lock files)."""
    candidates = sorted(
        p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx workbook found in {data_dir.resolve()}"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"Multiple .xlsx workbooks found in {data_dir.resolve()}: {names}. "
            "Leave only one input file or disambiguate manually."
        )
    return candidates[0]


def _condition_row_counts() -> Dict[Condition, int]:
    counts: Dict[Condition, int] = {}
    for condition in ("NoComp", "VthComp"):
        sheets = [
            (name, df)
            for name, df in _SHEETS.items()
            if name.endswith(f"_{condition}")
        ]
        if not sheets:
            raise AssertionError(f"No sheets found for condition {condition!r}")
        ref = len(sheets[0][1])
        for name, df in sheets:
            n = len(df)
            if n != ref:
                raise AssertionError(
                    f"{name}: expected {ref} rows for {condition}, got {n}"
                )
        counts[condition] = ref
    return counts


def run_sanity_checks() -> None:
    row_counts = _condition_row_counts()

    for name, df in _SHEETS.items():
        if name.endswith("_NoComp"):
            v = df["CH2 [V]"].to_numpy()
        elif name.endswith("_VthComp"):
            v = df["V1-V2"].to_numpy()
        else:
            raise AssertionError(f"Unexpected sheet name: {name}")
        dv = np.diff(v)
        assert np.all(dv <= 0) or np.all(dv >= 0), (
            f"{name}: voltage sweep not monotonic"
        )

    nc0 = load_cell(0, "NoComp")
    vc0 = load_cell(0, "VthComp")
    print(
        f"Summary: {N_CELLS} cells, "
        f"NoComp rows={row_counts['NoComp']}, VthComp rows={row_counts['VthComp']}, "
        f"NoComp V=[{nc0['v'].min():.1f}, {nc0['v'].max():.1f}] V, "
        f"VthComp V=[{vc0['v'].min():.1f}, {vc0['v'].max():.1f}] V, "
        f"PBS times (min)={PBS_TIMES_MIN}"
    )


def plot_figure_a() -> None:
    time_colors = _pbs_time_colors()
    fig, axes = plt.subplots(
        ARRAY_SIZE,
        ARRAY_SIZE,
        figsize=(14, 14),
        sharex=True,
        sharey=True,
    )

    legend_handles = []
    legend_labels = []
    legend_done: Set[Tuple[int, str]] = set()

    for bit_idx in range(N_CELLS):
        row, col = cell_to_grid(bit_idx)
        ax = axes[row, col]
        ax.set_title(f"({row}, {col})", fontsize=FONT_PANEL)

        for t in PBS_TIMES_MIN:
            color = time_colors[t]
            col_adc = f"adc_{t}"

            nc = load_cell(bit_idx, "NoComp")
            (ln_nc,) = ax.plot(
                nc["v"],
                nc[col_adc],
                color=color,
                linestyle="-",
                linewidth=LW_CURVE,
                label=f"{t} min NoComp",
            )
            key_nc = (t, "NoComp")
            if key_nc not in legend_done:
                legend_handles.append(ln_nc)
                legend_labels.append(f"{t} min NoComp")
                legend_done.add(key_nc)

            vc = load_cell(bit_idx, "VthComp")
            (ln_vc,) = ax.plot(
                vc["v"],
                vc[col_adc],
                color=color,
                linestyle="--",
                linewidth=LW_CURVE,
                label=f"{t} min VthComp",
            )
            key_vc = (t, "VthComp")
            if key_vc not in legend_done:
                legend_handles.append(ln_vc)
                legend_labels.append(f"{t} min VthComp")
                legend_done.add(key_vc)

        _style_ax(ax)

    fig.supxlabel("Input voltage (V)", fontsize=FONT_AXIS)
    fig.supylabel("ADC (code)", fontsize=FONT_AXIS)
    fig.suptitle(
        "5x5 array I-V under PBS — V_th compensation suppresses drift\n"
        "(NoComp: CH2; VthComp: V1-V2)",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=FONT_BASE,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    save_figure(fig, "fig_a_iv_grid")


def _mean_envelope_width(condition: Condition) -> float:
    """Mean of per-voltage std, averaged over PBS time points (one scalar per panel)."""
    widths = []
    for t in PBS_TIMES_MIN:
        _, stack = _stack_active(condition, t)
        widths.append(float(np.mean(stack.std(axis=0, ddof=1))))
    return float(np.mean(widths))


def plot_figure_b() -> None:
    time_colors = _pbs_time_colors()
    fig, (ax_nc, ax_vc) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    panel_xlabels = {
        "NoComp": "CH2 voltage (V)",
        "VthComp": "V1 - V2 (V)",
    }
    for ax, condition, title in (
        (ax_nc, "NoComp", "NoComp"),
        (ax_vc, "VthComp", "VthComp"),
    ):
        for t in PBS_TIMES_MIN:
            v, stack = _stack_active(condition, t)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0, ddof=1)
            color = time_colors[t]
            ax.plot(v, mean, color=color, linewidth=LW_MEAN, label=f"{t} min")
            ax.fill_between(
                v,
                mean - std,
                mean + std,
                color=color,
                alpha=ALPHA_ENVELOPE,
                linewidth=0,
            )

        avg_width = _mean_envelope_width(condition)
        ax.text(
            0.03,
            0.97,
            f"avg envelope width = {avg_width:.1f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=FONT_BASE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
        ax.set_title(title, fontsize=FONT_PANEL)
        ax.set_xlabel(panel_xlabels[condition], fontsize=FONT_AXIS)
        _style_ax(ax)

    ax_nc.set_ylabel("ADC (code)", fontsize=FONT_AXIS)
    ax_nc.legend(loc="lower left", fontsize=FONT_BASE - 1)
    fig.suptitle(
        f"Cell-to-cell variation across 5x5 array (n={len(active_cells())} cells)",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig_b_aggregate_spread")


def plot_figure_c() -> None:
    bits = active_cells()
    drift_nc = [compute_drift(load_cell(b, "NoComp")) for b in bits]
    drift_vc = [compute_drift(load_cell(b, "VthComp")) for b in bits]
    labels = [f"bit{b}" for b in bits]

    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(bits))
    width = 0.38
    ax_bar.bar(
        x - width / 2,
        drift_nc,
        width,
        label="NoComp",
        color=COLOR_NOCOMP,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_bar.bar(
        x + width / 2,
        drift_vc,
        width,
        label="VthComp",
        color=COLOR_VTHCOMP,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_BASE - 1)
    ax_bar.set_ylabel("Drift metric", fontsize=FONT_AXIS)
    ax_bar.set_xlabel("Active cell", fontsize=FONT_AXIS)
    ax_bar.set_title("Per-cell drift (0 to 60 min)", fontsize=FONT_PANEL)
    ax_bar.legend(fontsize=FONT_BASE)
    _style_ax(ax_bar)

    bp = ax_box.boxplot(
        [drift_nc, drift_vc],
        patch_artist=True,
        widths=0.5,
    )
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(["NoComp", "VthComp"])
    bp["boxes"][0].set_facecolor(COLOR_NOCOMP)
    bp["boxes"][1].set_facecolor(COLOR_VTHCOMP)
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    med_nc = float(np.nanmedian(drift_nc))
    med_vc = float(np.nanmedian(drift_vc))
    ax_box.text(
        1,
        med_nc,
        f"median={med_nc:.3f}",
        ha="center",
        va="bottom",
        fontsize=FONT_BASE - 1,
    )
    ax_box.text(
        2,
        med_vc,
        f"median={med_vc:.3f}",
        ha="center",
        va="bottom",
        fontsize=FONT_BASE - 1,
    )
    ax_box.set_ylabel("Drift metric", fontsize=FONT_AXIS)
    ax_box.set_title("Distribution across active cells", fontsize=FONT_PANEL)
    _style_ax(ax_box)

    fig.suptitle(
        "PBS-induced drift (0 to 60 min) per cell",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig_c_drift")


def main() -> None:
    global _SHEETS
    _setup_matplotlib()
    input_path = find_input_workbook(DATA_DIR)

    print(f"Loading {input_path.name} ...")
    _SHEETS = pd.read_excel(input_path, sheet_name=None)
    run_sanity_checks()

    print("Plotting Figure A ...")
    plot_figure_a()
    print("Plotting Figure B ...")
    plot_figure_b()
    print("Plotting Figure C ...")
    plot_figure_c()
    print(f"Done. Outputs in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
