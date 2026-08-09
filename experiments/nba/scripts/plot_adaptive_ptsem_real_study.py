"""Create publication-ready figures for the adaptive PT-SEM NBA study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("three_team", "five_team", "five_game")
ESTIMATORS = ("moment", "optimization")
LABELS = {
    "three_team": "3 variables\nteam-quarter",
    "five_team": "5 variables\nteam-quarter",
    "five_game": "5 variables\ngame-quarter",
    "moment": "Moment plug-in",
    "optimization": "Joint optimization",
}
VARIABLE_ORDER = ("FOUL", "FTA", "FTM", "MISS_FG", "REB")
FAMILY_ORDER = ("Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli")
COLORS = {"moment": "#E69F00", "optimization": "#0072B2"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.75,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        outdir / f"{stem}.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=600,
    )
    plt.close(fig)


def add_panel_labels(axes: np.ndarray) -> None:
    for index, axis in enumerate(axes.ravel()):
        axis.text(
            -0.16,
            1.07,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.2,
            va="top",
        )


def plot_edge_stability(edge_frame: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), squeeze=False)
    mesh = None
    for row, estimator in enumerate(ESTIMATORS):
        for column, dataset in enumerate(DATASETS):
            axis = axes[row, column]
            panel = edge_frame.loc[
                edge_frame["dataset"].eq(dataset)
                & edge_frame["estimator"].eq(estimator)
            ]
            variables = (
                VARIABLE_ORDER[:3] if dataset == "three_team" else VARIABLE_ORDER
            )
            matrix = np.full((len(variables), len(variables)), np.nan)
            reference = np.zeros_like(matrix, dtype=bool)
            for record in panel.itertuples(index=False):
                if record.source not in variables or record.target not in variables:
                    continue
                source = variables.index(record.source)
                target = variables.index(record.target)
                matrix[target, source] = record.selection_rate
                reference[target, source] = bool(record.in_reference_graph)
            mesh = axis.imshow(
                np.ma.masked_invalid(matrix),
                vmin=0,
                vmax=1,
                cmap="Blues",
                interpolation="nearest",
            )
            axis.set_xticks(range(len(variables)), variables, rotation=40, ha="right")
            axis.set_yticks(range(len(variables)), variables)
            axis.set_xlabel("Source")
            axis.set_ylabel("Target")
            if row == 0:
                axis.set_title(LABELS[dataset])
            if column == 2:
                axis.text(
                    1.12,
                    0.5,
                    LABELS[estimator],
                    transform=axis.transAxes,
                    rotation=-90,
                    va="center",
                    ha="left",
                )
            for target in range(len(variables)):
                for source in range(len(variables)):
                    value = matrix[target, source]
                    if np.isnan(value):
                        continue
                    text = f"{value:.2f}"
                    if reference[target, source]:
                        text += "*"
                    axis.text(
                        source,
                        target,
                        text,
                        ha="center",
                        va="center",
                        fontsize=6.6,
                        color="white" if value >= 0.58 else "#222222",
                    )
            axis.set_xticks(np.arange(-0.5, len(variables), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(variables), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.6)
            axis.tick_params(which="minor", bottom=False, left=False)
    add_panel_labels(axes)
    if mesh is not None:
        colorbar_axis = fig.add_axes([0.935, 0.18, 0.013, 0.67])
        colorbar = fig.colorbar(mesh, cax=colorbar_axis)
        colorbar.set_label("Selection rate across seasons")
    fig.text(
        0.5,
        0.015,
        "* denotes an edge in the rule-based reference graph",
        ha="center",
        fontsize=7.8,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.90,
        top=0.93,
        bottom=0.16,
        wspace=0.30,
        hspace=0.44,
    )
    save_figure(fig, outdir, "fig_real_edge_stability")


def plot_method_summary(summary: pd.DataFrame, outdir: Path) -> None:
    metrics = (
        ("core_chain_recovery_rate", "Core-chain recovery", (0, 1.05)),
        ("mean_directed_f1", "Directed F1 vs. reference", (0, 1.05)),
        ("median_bic_delta_second", r"Median $\Delta$BIC to second DAG", None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.05))
    x = np.arange(len(DATASETS))
    width = 0.36
    for axis, (metric, ylabel, ylim) in zip(axes, metrics):
        for offset, estimator in enumerate(ESTIMATORS):
            panel = (
                summary.loc[summary["estimator"].eq(estimator)]
                .set_index("dataset")
                .reindex(DATASETS)
            )
            axis.bar(
                x + (offset - 0.5) * width,
                panel[metric].to_numpy(float),
                width=width,
                label=LABELS[estimator],
                color=COLORS[estimator],
                edgecolor="white",
                linewidth=0.5,
            )
        axis.set_xticks(x, [LABELS[item] for item in DATASETS])
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    add_panel_labels(np.asarray(axes))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.27,
        wspace=0.34,
    )
    save_figure(fig, outdir, "fig_real_method_comparison")


def plot_family_selection(families: pd.DataFrame, outdir: Path) -> None:
    panel = families.loc[families["estimator"].eq("optimization")].copy()
    row_labels = []
    matrices = []
    for dataset in DATASETS:
        nodes = VARIABLE_ORDER[:3] if dataset == "three_team" else VARIABLE_ORDER
        for node in nodes:
            subset = panel.loc[
                panel["dataset"].eq(dataset) & panel["node"].eq(node)
            ]
            counts = subset["family"].value_counts()
            denominator = max(1, subset["season"].nunique())
            matrices.append(
                [counts.get(family, 0) / denominator for family in FAMILY_ORDER]
            )
            row_labels.append(f"{dataset}: {node}")
    matrix = np.asarray(matrices, dtype=float)
    fig, axis = plt.subplots(figsize=(7.1, 5.0))
    mesh = axis.imshow(matrix, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(FAMILY_ORDER)), FAMILY_ORDER, rotation=35, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel("Selected exogenous working family")
    axis.set_title("Joint-optimization family selection across six seasons")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if value >= 0.55 else "#222222",
            )
    colorbar = fig.colorbar(mesh, ax=axis, fraction=0.025, pad=0.025)
    colorbar.set_label("Selection rate")
    fig.subplots_adjust(left=0.25, right=0.93, top=0.91, bottom=0.18)
    save_figure(fig, outdir, "fig_real_family_selection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/adaptive_ptsem_real_study"),
    )
    args = parser.parse_args()
    root = args.results_dir.resolve()
    outdir = root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    configure_style()
    edge_frame = pd.read_csv(root / "edge_stability.csv")
    summary = pd.read_csv(root / "method_summary.csv")
    families = pd.read_csv(root / "selected_node_families.csv")
    plot_edge_stability(edge_frame, outdir)
    plot_method_summary(summary, outdir)
    plot_family_selection(families, outdir)
    print(f"Figures written to {outdir}", flush=True)


if __name__ == "__main__":
    main()
