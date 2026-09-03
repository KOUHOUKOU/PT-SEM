#!/usr/bin/env python3
"""Draw Figures 2-4 from complete, validated mixed-family summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "mixed_family.json").read_text(encoding="utf-8"))
SUMMARY_ROOT: Path | None = None
FINAL_ROOT: Path | None = None
PLOTDATA_ROOT: Path | None = None


def configure_run_root(run_root: Path) -> None:
    global SUMMARY_ROOT, FINAL_ROOT, PLOTDATA_ROOT
    resolved = run_root.resolve()
    SUMMARY_ROOT = resolved / "mixed_family" / "summaries"
    FINAL_ROOT = resolved / "paper_objects"
    PLOTDATA_ROOT = resolved / "mixed_family" / "plotdata"

METHOD_ORDER = ("LibraryDP", "LibraryGreedy", "OracleDP", "PoissonDAG-ODS", "PC-RCIT", "PBSCM", "PBSCM_PGF")
STYLE = {
    "LibraryDP": dict(color="#CC3366", marker="o", linestyle="-"),
    "LibraryGreedy": dict(color="#E69F00", marker="s", linestyle="-"),
    "OracleDP": dict(color="#4D4D4D", marker="D", linestyle="--"),
    "PoissonDAG-ODS": dict(color="#CC79A7", marker="d", linestyle="-."),
    "PC-RCIT": dict(color="#009E73", marker="^", linestyle="--"),
    "PBSCM": dict(color="#D55E00", marker="v", linestyle=":"),
    "PBSCM_PGF": dict(color="#56B4E9", marker="P", linestyle="-."),
}
SWEEPS = ("dimension", "sample_size", "average_in_degree")
XLABEL = {"dimension": "Number of nodes $d$", "sample_size": "Sample size $N$", "average_in_degree": "Average in-degree"}
GRID = dict(color="#D9D9D9", linestyle="--", linewidth=.45, alpha=.30)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12.5,
        "axes.labelsize": 12, "legend.fontsize": 10.5, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "axes.linewidth": .8, "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def setup_figure4_style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11.5,
        "axes.labelsize": 10.5, "legend.fontsize": 9, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })


def require_validated() -> pd.DataFrame:
    if SUMMARY_ROOT is None:
        raise RuntimeError("configure_run_root() must be called first")
    report_path = SUMMARY_ROOT / "validation_report.json"
    summary_path = SUMMARY_ROOT / "mixed_family_all_metrics_summary.csv"
    if not report_path.exists() or not summary_path.exists():
        raise SystemExit("Validated mixed-family summaries do not exist; run summarize_mixed_family.py")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("R") != 100:
        raise RuntimeError("Mixed-family validation gate is not passed at R=100")
    return pd.read_csv(summary_path)


def configure_axis(axis: plt.Axes, sweep: str) -> None:
    axis.grid(True, **GRID)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if sweep == "sample_size":
        axis.set_xscale("log")
        axis.set_xticks((100, 1000, 10000), ("100", "1,000", "10,000"))


def save(fig: plt.Figure, name: str, plotdata: pd.DataFrame, metadata: dict[str, object], *, tight: bool = True) -> None:
    if FINAL_ROOT is None or PLOTDATA_ROOT is None:
        raise RuntimeError("configure_run_root() must be called first")
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTDATA_ROOT.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(FINAL_ROOT / f"{name}.pdf", **save_options)
    fig.savefig(FINAL_ROOT / f"{name}.png", dpi=400, **save_options)
    plt.close(fig)
    plotdata.to_csv(PLOTDATA_ROOT / f"{name}_plotdata.csv", index=False)
    (PLOTDATA_ROOT / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def six_panel(summary: pd.DataFrame, metric: str, ylabel: str, methods: tuple[str, ...], fixed_ylim: tuple[float, float] | None) -> plt.Figure:
    data = summary.loc[summary["metric"] == metric].copy()
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 5.85), sharey=False)
    handles: dict[str, object] = {}
    for row, regime in enumerate(("restricted", "extended")):
        allowed = methods if regime == "restricted" else tuple(method for method in methods if method not in {"PBSCM", "PBSCM_PGF"})
        for column, sweep in enumerate(SWEEPS):
            axis = axes[row, column]
            panel = data.loc[(data["regime"] == regime) & (data["sweep"] == sweep)]
            for method in allowed:
                curve = panel.loc[panel["method"] == method].sort_values("sweep_value")
                if curve.empty:
                    raise RuntimeError(f"Missing plot curve: {regime}/{sweep}/{method}/{metric}")
                x = curve["sweep_value"].to_numpy(float)
                style = STYLE[method]
                line = axis.plot(x, curve["mean"], linewidth=2, markersize=4.2, label=CONFIG["display_names"][method], **style)[0]
                axis.fill_between(x, curve["ci95_low"], curve["ci95_high"], color=style["color"], alpha=.10, linewidth=0)
                handles.setdefault(method, line)
            if row == 0:
                title = {
                    "dimension": "Dimension sweep",
                    "sample_size": "Sample-size sweep",
                    "average_in_degree": "Average-in-degree sweep",
                }[sweep]
                axis.set_title(f"({chr(ord('a') + column)}) {title}")
            else:
                axis.set_xlabel(XLABEL[sweep])
            if column == 0:
                axis.set_ylabel(("Restricted\n" if row == 0 else "Extended\n") + ylabel)
            if fixed_ylim is not None:
                axis.set_ylim(*fixed_ylim)
            else:
                upper = float(np.nanmax(panel["ci95_high"]))
                axis.set_ylim(0, max(1e-8, 1.12 * upper))
            if sweep == "dimension":
                axis.set_xticks((4, 5, 6, 7, 8, 9, 10))
            elif sweep == "average_in_degree":
                axis.set_xticks((1, 1.5, 2, 2.5, 3))
            configure_axis(axis, sweep)
    ordered = [method for method in METHOD_ORDER if method in handles]
    fig.legend([handles[method] for method in ordered], [CONFIG["display_names"][method] for method in ordered], loc="lower center", ncol=len(ordered), frameon=False, bbox_to_anchor=(.5, .018))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.175,
        wspace=0.23,
        hspace=0.20,
    )
    return fig


def figure1(summary: pd.DataFrame) -> None:
    data = summary.loc[summary["metric"] == "directed_f1_recomputed"].copy()
    fig = six_panel(summary, "directed_f1_recomputed", "Directed F1", METHOD_ORDER, (0, 1.02))
    save(fig, "figure2_dag_recovery_f1", data, {"source": "validated raw directed edge sets", "R": 100, "layout": "restricted/extended by dimension/sample-size/average-indegree"}, tight=False)


def figure2(summary: pd.DataFrame) -> None:
    methods = ("LibraryDP", "LibraryGreedy", "OracleDP")
    data = summary.loc[summary["metric"] == "alpha_mape_recomputed"].copy()
    fig = six_panel(summary, "alpha_mape_recomputed", r"Conditional $\alpha$-MAPE (%)", methods, None)
    save(fig, "figure3_coefficient_mape", data, {"source": "recomputed from true and estimated alpha matrices over correctly recovered directed edges", "no_correct_edge": "NA", "R": 100}, tight=False)


def figure3(summary: pd.DataFrame) -> None:
    setup_figure4_style()
    accuracy = summary.loc[(summary["metric"] == "family_accuracy_recomputed") & (summary["sweep"] == "sample_size")].copy()
    confusion = pd.read_csv(SUMMARY_ROOT / "mixed_family_anchor_confusion.csv")
    PLOTDATA_ROOT.mkdir(parents=True, exist_ok=True)
    confusion.to_csv(
        PLOTDATA_ROOT / "figure4_family_selection_confusion_plotdata.csv",
        index=False,
    )
    families = tuple(CONFIG["family_library"])
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    axis = axes[0]
    extended_colors = {"LibraryDP": "#AD2B57", "LibraryGreedy": "#C38700"}
    for method in ("LibraryDP", "LibraryGreedy"):
        for regime, line_style in (("restricted", "-"), ("extended", (0, (5.0, 3.5)))):
            curve = accuracy.loc[(accuracy["method"] == method) & (accuracy["regime"] == regime)].sort_values("sweep_value")
            x = curve["sweep_value"].to_numpy(float)
            style = STYLE[method]
            line_color = style["color"] if regime == "restricted" else extended_colors[method]
            label = f"{CONFIG['display_names'][method]} - {regime}"
            axis.plot(x, curve["mean"], color=line_color, marker=style["marker"], linestyle=line_style, linewidth=2, markersize=4.2, label=label)
            axis.fill_between(x, curve["ci95_low"], curve["ci95_high"], color=style["color"], alpha=.10, linewidth=0)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Sample size $N$")
    axis.set_ylabel("Family-selection accuracy")
    configure_axis(axis, "sample_size")
    axis.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.30), borderaxespad=0, handlelength=3.2)
    image = None
    for axis, regime in zip(axes[1:], ("restricted", "extended")):
        matrix = confusion.loc[confusion["regime"] == regime].pivot(index="true_family", columns="selected_family", values="row_proportion").reindex(index=families, columns=families).fillna(0).to_numpy(float)
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(6):
            for j in range(6):
                axis.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8.2, color="white" if matrix[i,j] >= .5 else "#222222")
        axis.set_xticks(range(6), families, rotation=45, ha="right")
        axis.set_yticks(range(6), families)
        axis.set_xlabel("Selected exogenous family")
        axis.set_ylabel("Generating exogenous family")
    fig.subplots_adjust(left=.065, right=.90, top=.90, bottom=.37, wspace=.42)
    fig.canvas.draw()
    matrix_position = axes[1].get_position()
    color_axis = fig.add_axes([.925, matrix_position.y0, .014, matrix_position.height])
    fig.colorbar(image, cax=color_axis, label="Row proportion")
    for panel_axis, panel_label in zip(axes, ("(a)", "(b)", "(c)")):
        position = panel_axis.get_position()
        fig.text(position.x0 + position.width / 2, .025, panel_label, ha="center", va="bottom", fontsize=10.5)
    save(fig, "figure4_family_selection", accuracy, {"accuracy_methods": ["Proposed DP-BIC", "Greedy-BIC"], "confusion_method": "Proposed DP-BIC", "confusion_setting": {"d": 8, "N": 3200, "average_in_degree": 1.5}, "confusion_plotdata": "figure4_family_selection_confusion_plotdata.csv", "R": 100})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    configure_run_root(args.run_root)
    setup_style()
    summary = require_validated()
    figure1(summary)
    figure2(summary)
    figure3(summary)
    print(FINAL_ROOT)


if __name__ == "__main__":
    main()
