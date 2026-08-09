"""Produce the final PT-SEM paper figures and tables from completed results.

This script does not import simulation, estimator, or baseline code. It reads
completed per-replication CSV files only. Out-of-library diagnostic results
are deliberately not read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
MAIN_RESULTS = HERE / "ptsem_six_family_final_results"
EXTENDED_RESULTS = HERE / "ptsem_extended_results"
BASKETBALL = Path(r"C:\Users\ROG\Desktop\Basketball")
OUT = Path(
    os.environ.get("PTSEM_FINAL_OUT", str(HERE / "final_paper_figures"))
).expanduser().resolve()
MAIN_FIG = OUT / "main_figures"
APP_FIG = OUT / "appendix_figures"
APP_TABLE = OUT / "appendix_tables"
MAIN_TABLE = OUT / "main_tables"
NBA_FIG = OUT / "figures"
NBA_DIAGNOSTIC_FIG = OUT / "nba_figures"
NBA_TABLE = OUT / "nba_tables"
META = OUT / "metadata"

FAMILIES = ("Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli")
METHOD_MAP = {
    "LibraryDP": "LibraryDP",
    "LibraryGreedy": "LibraryGreedy",
    "OracleDP": "OracleDP",
    "FixedPoissonDP": "Fixed-Poisson DP",
    "PoissonDAG-ODS": "ODS",
    "ODS": "ODS",
    "PC-RCIT": "PC-RCIT",
    "PBSCM": "PB-SCM",
    "PB-SCM": "PB-SCM",
    "PBSCM_PGF": "PB-SCM-PGF",
    "PB-SCM-PGF": "PB-SCM-PGF",
}
METHOD_ORDER = (
    "LibraryDP",
    "LibraryGreedy",
    "OracleDP",
    "Fixed-Poisson DP",
    "ODS",
    "PC-RCIT",
    "PB-SCM",
    "PB-SCM-PGF",
)
STYLE = {
    "LibraryDP": dict(color="#CC3366", marker="o", linestyle="-"),
    "LibraryGreedy": dict(color="#E69F00", marker="s", linestyle="-"),
    "OracleDP": dict(color="#4D4D4D", marker="D", linestyle="--"),
    "Fixed-Poisson DP": dict(color="#0072B2", marker="o", linestyle="-"),
    "ODS": dict(color="#CC79A7", marker="d", linestyle="-."),
    "PC-RCIT": dict(color="#009E73", marker="^", linestyle="--"),
    "PB-SCM": dict(color="#D55E00", marker="v", linestyle=":"),
    "PB-SCM-PGF": dict(color="#56B4E9", marker="P", linestyle="-."),
}
SWEEP_ORDER = ("d-sweep", "N-sweep", "density-sweep")
SWEEP_XLABEL = {
    "d-sweep": "Number of nodes $d$",
    "N-sweep": "Sample size $N$",
    "density-sweep": "Average indegree",
}
N_TICKS = (100, 1000, 10000)
GRID = dict(color="#D9D9D9", linestyle="--", linewidth=0.45, alpha=0.30)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def dataframe_to_latex(
    frame: pd.DataFrame,
    column_formats: dict[str, str] | None = None,
    align: str | None = None,
) -> str:
    column_formats = column_formats or {}

    def escape(value: object, column: str) -> str:
        if pd.isna(value):
            return "--"
        if column in column_formats and isinstance(value, (int, float, np.number)):
            return column_formats[column].format(float(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        text = str(value)
        replacements = (
            ("\\", r"\textbackslash{}"),
            ("_", r"\_"),
            ("%", r"\%"),
            ("&", r"\&"),
            ("#", r"\#"),
            ("->", r"$\to$"),
        )
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    alignment = align or ("l" + "c" * (len(frame.columns) - 1))
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(escape(column, column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            " & ".join(
                escape(value, str(column))
                for value, column in zip(row, frame.columns)
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def save_table(
    frame: pd.DataFrame,
    directory: Path,
    stem: str,
    formats: dict[str, str] | None = None,
) -> None:
    atomic_csv(frame, directory / f"{stem}.csv")
    (directory / f"{stem}.tex").write_text(
        dataframe_to_latex(frame, formats), encoding="utf-8"
    )


def save_figure(
    fig: plt.Figure,
    directory: Path,
    stem: str,
    plotdata: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    atomic_csv(plotdata, directory / f"{stem}_plotdata.csv")
    metadata = {
        "ci_convention": "pointwise mean +/- 1.96*SE; SD uses ddof=1",
        "sd_bands_used": False,
        "simulation_rerun": False,
        **metadata,
    }
    (META / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def configure_axis(axis: plt.Axes, sweep: str | None = None) -> None:
    axis.set_axisbelow(True)
    axis.grid(True, **GRID)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if sweep == "N-sweep":
        axis.set_xscale("log")
        axis.set_xticks(N_TICKS)
        axis.set_xticklabels((r"$10^2$", r"$10^3$", r"$10^4$"))
        axis.grid(
            True,
            which="minor",
            color="#E6E6E6",
            linestyle="--",
            linewidth=0.35,
            alpha=0.15,
        )


def load_main_raw() -> pd.DataFrame:
    sources = (
        (
            "d-sweep",
            MAIN_RESULTS / "experiment1_d_sweep/experiment1_d_sweep_raw.csv",
        ),
        (
            "N-sweep",
            MAIN_RESULTS / "experiment2_N_sweep/experiment2_N_sweep_raw.csv",
        ),
        (
            "density-sweep",
            MAIN_RESULTS / "experiment3_kin_sweep/experiment3_kin_sweep_raw.csv",
        ),
    )
    frames = []
    for sweep, path in sources:
        frame = pd.read_csv(path)
        frame["sweep_type"] = sweep
        frame["x_value"] = (
            frame["d"]
            if sweep == "d-sweep"
            else (frame["N"] if sweep == "N-sweep" else frame["kbar_in"])
        )
        frame["method"] = frame["method"].map(METHOD_MAP)
        frame["replication"] = frame["rep"]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    return raw


def aggregate_metric(
    raw: pd.DataFrame,
    value_column: str,
    metric: str,
    methods: Sequence[str] | None = None,
    valid_domain: tuple[float, float | None] = (0.0, None),
) -> pd.DataFrame:
    data = raw.copy()
    if methods is not None:
        data = data[data["method"].isin(methods)]
    rows = []
    keys = ["alpha_regime", "sweep_type", "x_value", "method"]
    for key, group in data.groupby(keys, sort=True, dropna=False):
        all_values = pd.to_numeric(group[value_column], errors="coerce")
        values = all_values.dropna().to_numpy(float)
        n_total = int(group["replication"].nunique())
        n_nonNA = int(len(values))
        mean = float(values.mean()) if n_nonNA else float("nan")
        sd = float(values.std(ddof=1)) if n_nonNA > 1 else float("nan")
        se = sd / math.sqrt(n_nonNA) if n_nonNA > 1 else float("nan")
        low = mean - 1.96 * se if n_nonNA > 1 else mean
        high = mean + 1.96 * se if n_nonNA > 1 else mean
        plot_low = low
        plot_high = high
        lower, upper = valid_domain
        if np.isfinite(plot_low):
            plot_low = max(lower, plot_low)
        if upper is not None and np.isfinite(plot_high):
            plot_high = min(upper, plot_high)
        rows.append(
            {
                "regime": key[0],
                "sweep": key[1],
                "x_value": key[2],
                "method": key[3],
                "metric": metric,
                "N_total": n_total,
                "N_nonNA": n_nonNA,
                "NA_rate": 1.0 - n_nonNA / n_total,
                "mean": mean,
                "sd": sd,
                "se": se,
                "ci95_low": low,
                "ci95_high": high,
                "plot_ci95_low": plot_low,
                "plot_ci95_high": plot_high,
            }
        )
    return pd.DataFrame(rows)


def plot_summary_line(
    axis: plt.Axes,
    panel: pd.DataFrame,
    method: str,
    band: bool,
    label: str | None = None,
) -> object | None:
    values = panel[panel["method"] == method].sort_values("x_value")
    if values.empty:
        return None
    style = STYLE[method]
    x = values["x_value"].to_numpy(float)
    line = axis.plot(
        x,
        values["mean"].to_numpy(float),
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2.0,
        markersize=4.2,
        label=label or method,
        zorder=4 if method in ("LibraryDP", "OracleDP") else 3,
    )[0]
    if band:
        axis.fill_between(
            x,
            values["plot_ci95_low"].to_numpy(float),
            values["plot_ci95_high"].to_numpy(float),
            color=style["color"],
            alpha=0.11,
            linewidth=0,
            zorder=1,
        )
    return line


def six_panel_figure(
    summary: pd.DataFrame,
    metric_label: str,
    methods_common: Sequence[str],
    methods_expanded: Sequence[str],
    band_methods: set[str],
    y_limits_by_row: tuple[tuple[float, float] | None, tuple[float, float] | None],
    y_limits_by_panel: dict[
        tuple[str, str], tuple[float, float]
    ] | None = None,
) -> tuple[plt.Figure, dict[str, object]]:
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.25), sharey=False)
    handles: dict[str, object] = {}
    letters = iter("abcdef")
    for row, regime in enumerate(("common", "expanded")):
        methods = methods_common if regime == "common" else methods_expanded
        for col, sweep in enumerate(SWEEP_ORDER):
            axis = axes[row, col]
            panel = summary[
                (summary["regime"] == regime) & (summary["sweep"] == sweep)
            ]
            for method in methods:
                handle = plot_summary_line(
                    axis, panel, method, method in band_methods
                )
                if handle is not None:
                    handles[method] = handle
            letter = next(letters)
            axis.set_title(f"({letter}) {sweep}")
            axis.set_xlabel(SWEEP_XLABEL[sweep])
            if col == 0:
                row_name = "Common" if regime == "common" else "Expanded"
                axis.set_ylabel(f"{row_name}\n{metric_label}")
            if (
                y_limits_by_panel is not None
                and (regime, sweep) in y_limits_by_panel
            ):
                axis.set_ylim(*y_limits_by_panel[(regime, sweep)])
            elif y_limits_by_row[row] is not None:
                axis.set_ylim(*y_limits_by_row[row])
            if sweep == "d-sweep":
                axis.set_xticks(sorted(panel["x_value"].unique().astype(int)))
            elif sweep == "density-sweep":
                axis.set_xticks(sorted(panel["x_value"].unique()))
            configure_axis(axis, sweep)
    ordered = [method for method in METHOD_ORDER if method in handles]
    fig.legend(
        [handles[method] for method in ordered],
        ordered,
        loc="lower center",
        ncol=min(7, len(ordered)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    return fig, handles


def make_figure1(raw: pd.DataFrame) -> pd.DataFrame:
    summary = aggregate_metric(
        raw,
        "directed_f1",
        "directed_f1",
        valid_domain=(0, 1),
    )
    common = (
        "LibraryDP",
        "LibraryGreedy",
        "OracleDP",
        "ODS",
        "PC-RCIT",
        "PB-SCM",
        "PB-SCM-PGF",
    )
    expanded = common[:5]
    fig, _ = six_panel_figure(
        summary,
        "Directed F1",
        common,
        expanded,
        set(common),
        ((0, 1.02), (0, 1.02)),
    )
    save_figure(
        fig,
        MAIN_FIG,
        "fig1_directed_f1_final",
        summary,
        {
            "metric_definition": "Directed edge-set F1; unresolved undirected PC edges are not duplicated into two directions.",
            "methods_common": list(common),
            "methods_expanded": list(expanded),
            "bands_drawn_for": list(common),
            "caption": "Directed-F1 recovery in heterogeneous six-family PT-SEM simulations. Rows correspond to the common and expanded coefficient regimes; columns correspond to the d-sweep, N-sweep, and density-sweep. Curves report means over R=100 replications, with shaded bands indicating pointwise 95% Monte Carlo confidence intervals for the mean.",
        },
    )
    return summary


def make_figure2(raw: pd.DataFrame) -> pd.DataFrame:
    methods = ("LibraryDP", "LibraryGreedy", "OracleDP")
    summary = aggregate_metric(
        raw,
        "alpha_mape_on_correct_edges_pct",
        "conditional_alpha_mape",
        methods=methods,
        valid_domain=(0, None),
    )
    panel_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for regime in ("common", "expanded"):
        for sweep in SWEEP_ORDER:
            panel = summary[
                (summary["regime"] == regime)
                & (summary["sweep"] == sweep)
            ]
            maximum = float(np.nanmax(panel["plot_ci95_high"]))
            panel_limits[(regime, sweep)] = (
                0.0,
                max(1e-6, 1.12 * maximum),
            )
    fig, _ = six_panel_figure(
        summary,
        r"Conditional $\alpha$-MAPE (%)",
        methods,
        methods,
        set(methods),
        (None, None),
        y_limits_by_panel=panel_limits,
    )
    save_figure(
        fig,
        MAIN_FIG,
        "fig2_conditional_alpha_mape_final",
        summary,
        {
            "metric_definition": "Within each replication, 100 times the mean of abs(alpha_hat-alpha_true)/alpha_true over true directed edges recovered with the correct orientation. It is not conditioned on exact graph recovery and excludes missed or wrongly oriented true edges. If no true directed edge is recovered with the correct orientation, the value is NA rather than zero.",
            "source_field": "alpha_mape_on_correct_edges_pct from the completed per-replication raw CSV files.",
            "aggregation": "Mean/SD/SE are computed over non-NA replications only. Plot data records N_total, N_nonNA, and NA_rate.",
            "units": "Percent",
            "denominator_note": "Simulated coefficients are bounded away from zero in both regimes, so the relative-error denominator is well-defined.",
            "correct_edge_coverage_output": "appendix_tables/correct_edge_coverage_summary_table.csv",
            "methods": list(methods),
            "panel_scales": {
                f"{regime}/{sweep}": list(limits)
                for (regime, sweep), limits in panel_limits.items()
            },
            "caption": "Conditional relative alpha-error on correctly recovered directed edges. A true directed edge contributes to the metric only when it is selected with the correct orientation. The plotted value is the mean absolute percentage error of the thinning coefficient, expressed in percent. Since simulated coefficients are bounded away from zero, the relative error is well-defined. Directed-edge recovery itself is evaluated in Figure 1.",
        },
    )
    return summary


def make_figureA2(raw: pd.DataFrame) -> pd.DataFrame:
    methods = ("LibraryDP", "LibraryGreedy", "OracleDP")
    summary = aggregate_metric(
        raw,
        "alpha_rmse_on_true_edges",
        "structure_penalized_alpha_rmse",
        methods=methods,
        valid_domain=(0, None),
    )
    common_high = float(
        np.nanmax(summary.loc[summary["regime"] == "common", "plot_ci95_high"])
    )
    expanded_high = float(
        np.nanmax(summary.loc[summary["regime"] == "expanded", "plot_ci95_high"])
    )
    fig, _ = six_panel_figure(
        summary,
        r"Structure-penalized $\alpha$-RMSE",
        methods,
        methods,
        set(methods),
        ((0, common_high * 1.08), (0, expanded_high * 1.08)),
    )
    save_figure(
        fig,
        APP_FIG,
        "figA2_structure_penalized_alpha_rmse_final",
        summary,
        {
            "metric_definition": "RMSE over all true directed edges; missed or wrongly oriented true edges receive alpha_hat=0.",
            "interpretation": "End-to-end directed-effect recovery, not pure conditional coefficient accuracy.",
            "methods": list(methods),
            "caption": "Structure-penalized alpha-RMSE. The metric is computed over all true directed edges; missed or wrongly oriented true edges are assigned estimated coefficient zero. It therefore measures end-to-end recovery of directed effects rather than coefficient accuracy conditional on a correctly recovered edge.",
        },
    )
    return summary


def make_figureA3(raw: pd.DataFrame) -> pd.DataFrame:
    methods = ("LibraryDP", "LibraryGreedy", "OracleDP")
    data = raw.copy()
    data["correct_edge_coverage"] = (
        data["n_correct_directed_edges"] / data["n_true_edges"]
    )
    summary = aggregate_metric(
        data,
        "correct_edge_coverage",
        "correct_edge_coverage",
        methods=methods,
        valid_domain=(0, 1),
    )
    fig, _ = six_panel_figure(
        summary,
        "Correct-edge coverage",
        methods,
        methods,
        set(methods),
        ((0, 1.02), (0, 1.02)),
    )
    save_figure(
        fig,
        APP_FIG,
        "figA3_correct_edge_coverage_final",
        summary,
        {
            "metric_definition": "Number of true directed edges recovered with correct orientation divided by number of true directed edges.",
            "methods": list(methods),
            "caption": "Correct-edge coverage for the conditional coefficient RMSE. Coverage is the proportion of true directed edges recovered with the correct orientation. It is reported alongside conditional coefficient RMSE to show how many true effects enter the conditional RMSE calculation.",
        },
    )
    return summary


def family_confusion(raw: pd.DataFrame, regime: str) -> pd.DataFrame:
    selected = raw[
        (raw["sweep_type"] == "N-sweep")
        & (raw["alpha_regime"] == regime)
        & (raw["x_value"] == 3200)
        & (raw["method"] == "LibraryDP")
    ]
    counts: dict[tuple[str, str], int] = {
        (true, estimate): 0 for true in FAMILIES for estimate in FAMILIES
    }
    for row in selected.itertuples(index=False):
        truth = str(row.true_families).split(",")
        estimate = str(row.selected_families).split(",")
        for true, chosen in zip(truth, estimate):
            counts[(true, chosen)] += 1
    rows = []
    for true in FAMILIES:
        total = sum(counts[(true, selected_family)] for selected_family in FAMILIES)
        for selected_family in FAMILIES:
            value = counts[(true, selected_family)]
            rows.append(
                {
                    "regime": regime,
                    "method": "LibraryDP",
                    "d": 8,
                    "N": 3200,
                    "average_indegree": 1.5,
                    "true_family": true,
                    "selected_working_family": selected_family,
                    "count": value,
                    "row_total": total,
                    "row_proportion": value / total,
                }
            )
    return pd.DataFrame(rows)


def make_figure3(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_raw = raw[
        (raw["sweep_type"] == "N-sweep")
        & (raw["method"].isin(["LibraryDP", "LibraryGreedy"]))
    ]
    accuracy = aggregate_metric(
        n_raw,
        "family_accuracy",
        "working_family_accuracy",
        methods=("LibraryDP", "LibraryGreedy"),
        valid_domain=(0, 1),
    )
    common_conf = family_confusion(raw, "common")
    expanded_conf = family_confusion(raw, "expanded")

    fig, axes = plt.subplots(
        1, 3, figsize=(11.8, 3.6), gridspec_kw={"width_ratios": [1.25, 1, 1]}
    )
    axis = axes[0]
    handles = []
    for method in ("LibraryDP", "LibraryGreedy"):
        for regime, linestyle in (("common", "-"), ("expanded", "--")):
            panel = accuracy[
                (accuracy["method"] == method) & (accuracy["regime"] == regime)
            ].sort_values("x_value")
            style = STYLE[method]
            x = panel["x_value"].to_numpy(float)
            line = axis.plot(
                x,
                panel["mean"],
                color=style["color"],
                marker=style["marker"],
                linestyle=linestyle,
                linewidth=2,
                markersize=4.2,
                label=(
                    f"{'Proposed DP-BIC' if method == 'LibraryDP' else 'Greedy-BIC'} - "
                    f"{'restricted' if regime == 'common' else 'extended'}"
                ),
            )[0]
            axis.fill_between(
                x,
                panel["plot_ci95_low"],
                panel["plot_ci95_high"],
                color=style["color"],
                alpha=0.10,
                linewidth=0,
            )
            handles.append(line)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Sample size $N$")
    axis.set_ylabel("Family-selection accuracy")
    configure_axis(axis, "N-sweep")
    axis.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
    )

    image = None
    for axis, frame, title in (
        (axes[1], common_conf, "(b) Common"),
        (axes[2], expanded_conf, "(c) Expanded"),
    ):
        matrix = (
            frame.pivot(
                index="true_family",
                columns="selected_working_family",
                values="row_proportion",
            )
            .reindex(index=FAMILIES, columns=FAMILIES)
            .to_numpy(float)
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(6):
            for j in range(6):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] >= 0.5 else "#222222",
                )
        axis.set_xticks(range(6), FAMILIES, rotation=45, ha="right")
        axis.set_yticks(range(6), FAMILIES)
        axis.set_xlabel("Selected exogenous family")
        axis.set_ylabel("Generating exogenous family")
    fig.subplots_adjust(left=0.065, right=0.90, top=0.78, bottom=0.22, wspace=0.42)
    color_axis = fig.add_axes([0.925, 0.26, 0.014, 0.62])
    fig.colorbar(image, cax=color_axis, label="Row proportion")

    save_figure(
        fig,
        MAIN_FIG,
        "fig3_working_family_diagnostics_final",
        accuracy,
        {
            "metric_definition": "Working-family accuracy is the proportion of nodes whose selected working family equals the generated family, pooled over all nodes and all replications in a setting. It is not conditioned on exact graph recovery; OracleDP is excluded because its family is fixed to truth by construction.",
            "representative_confusion_setting": {
                "d": 8,
                "N": 3200,
                "average_indegree": 1.5,
                "method": "LibraryDP",
            },
            "caption": "Working-family diagnostics. Panel (a) reports node-wise working-family accuracy along the N-sweep. Panels (b) and (c) show row-normalized confusion matrices for LibraryDP in representative common and expanded settings. Family labels are interpreted as working choices for finite-family scoring rather than as uniquely identifiable labels in all overlapping or boundary cases.",
        },
    )
    atomic_csv(
        common_conf,
        MAIN_FIG / "fig3_confusion_common_plotdata.csv",
    )
    atomic_csv(
        expanded_conf,
        MAIN_FIG / "fig3_confusion_expanded_plotdata.csv",
    )
    return accuracy, common_conf, expanded_conf


def load_all_poisson_raw() -> pd.DataFrame:
    frame = pd.read_csv(EXTENDED_RESULTS / "synthetic_per_replication.csv")
    frame = frame[frame["scenario"] == "all_poisson"].copy()
    frame["alpha_regime"] = "common"
    frame["sweep_type"] = "N-sweep"
    frame["method"] = frame["method"].map(METHOD_MAP)
    return frame


def make_figure4() -> pd.DataFrame:
    raw = load_all_poisson_raw()
    methods = ("Fixed-Poisson DP", "LibraryDP", "LibraryGreedy", "ODS", "PC-RCIT")
    summary = aggregate_metric(
        raw,
        "directed_f1",
        "directed_f1",
        methods=methods,
        valid_domain=(0, 1),
    )
    fixed = raw[raw["method"] == "Fixed-Poisson DP"][
        ["x_value", "replication", "seed", "directed_f1"]
    ].rename(columns={"directed_f1": "fixed_f1"})
    library = raw[raw["method"] == "LibraryDP"][
        ["x_value", "replication", "seed", "directed_f1"]
    ].rename(columns={"directed_f1": "library_f1"})
    paired = fixed.merge(
        library, on=["x_value", "replication", "seed"], validate="one_to_one"
    )
    paired["paired_directed_f1_gap_percentage_points"] = 100.0 * (
        paired["fixed_f1"] - paired["library_f1"]
    )
    paired["alpha_regime"] = "common"
    paired["sweep_type"] = "N-sweep"
    paired["method"] = "Fixed-Poisson DP - LibraryDP"
    gap = aggregate_metric(
        paired,
        "paired_directed_f1_gap_percentage_points",
        "paired_directed_f1_gap_percentage_points",
        methods=("Fixed-Poisson DP - LibraryDP",),
        valid_domain=(-np.inf, None),
    )
    summary["panel"] = "main_directed_f1"
    gap["panel"] = "supplementary_paired_gap_not_plotted"
    plotdata = pd.concat([summary, gap], ignore_index=True, sort=False)

    fig, axis = plt.subplots(figsize=(6.7, 4.6))
    handles = []
    for method in methods:
        handle = plot_summary_line(
            axis,
            summary,
            method,
            band=True,
            label="Poisson DAG (ODS)" if method == "ODS" else method,
        )
        if handle is not None:
            handles.append(handle)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Sample size $N$")
    axis.set_ylabel("Directed F1")
    axis.set_title("All-Poisson overlap")
    configure_axis(axis, "N-sweep")
    axis.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower right",
    )
    fig.tight_layout()
    save_figure(
        fig,
        MAIN_FIG,
        "fig4_all_poisson_library_cost_final",
        plotdata,
        {
            "scenario": "All node-wise exogenous noises are Poisson; common coefficient regime; d=8; average indegree=1.5.",
            "displayed_methods": list(methods),
            "displayed_uncertainty": "Pointwise 95% Monte Carlo CI bands are drawn for every displayed method.",
            "layout": "Single-panel directed-F1 figure. The former paired-gap panel is not drawn.",
            "paired_gap_definition": "Replication-level paired gap 100*{F1(Fixed-Poisson DP)-F1(LibraryDP)}, using the same generated dataset and expressed in percentage points.",
            "paired_gap_retention": "The paired-gap calculation and summary remain in the plotdata CSV under panel=supplementary_paired_gap_not_plotted; they are retained for text or an appendix table.",
            "ods_note": "The ODS curve is a Poisson-DAG baseline using its implemented scoring convention; it is not an oracle for the additive-link PT-SEM likelihood.",
            "interpretation": "All-Poisson additive-overlap and library-cost sensitivity analysis; not a fully oracle comparison with all Poisson DAG models.",
            "caption": "All-Poisson additive-overlap experiment. Directed F1 is reported along the N-sweep for Fixed-Poisson DP, LibraryDP, LibraryGreedy, Poisson DAG (ODS), and PC-RCIT. Curves show replication means, with shaded bands denoting pointwise 95% Monte Carlo confidence intervals for the mean. The experiment assesses performance in the additive-Poisson overlap regime and the finite-sample cost of profiling over the six-family working library.",
        },
    )
    return plotdata


def make_figureA1() -> pd.DataFrame:
    frame = pd.read_csv(
        EXTENDED_RESULTS
        / "existing_results_augmented/existing_per_replication_metrics.csv"
    )
    frame["method"] = frame["method"].map(METHOD_MAP)
    frame["alpha_regime"] = frame["coefficient_regime"]
    frame["sweep_type"] = frame["sweep_type"].map(
        {"d_sweep": "d-sweep", "N_sweep": "N-sweep", "density_sweep": "density-sweep"}
    )
    mixed_graph_methods = {"PC-RCIT", "PB-SCM", "PB-SCM-PGF"}
    frame["shd_for_plot"] = np.where(
        frame["method"].isin(mixed_graph_methods),
        frame["skeleton_shd"],
        frame["directed_shd_rev1"],
    )
    frame["shd_definition"] = np.where(
        frame["method"].isin(mixed_graph_methods),
        "skeleton_shd",
        "directed_shd_reversal_1",
    )
    summary = aggregate_metric(
        frame,
        "shd_for_plot",
        "shd_diagnostic",
        valid_domain=(0, None),
    )
    definition = (
        frame[
            [
                "alpha_regime",
                "sweep_type",
                "x_value",
                "method",
                "shd_definition",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "alpha_regime": "regime",
                "sweep_type": "sweep",
            }
        )
    )
    summary = summary.merge(
        definition,
        on=["regime", "sweep", "x_value", "method"],
        how="left",
        validate="one_to_one",
    )
    common = (
        "LibraryDP",
        "LibraryGreedy",
        "OracleDP",
        "PC-RCIT",
        "ODS",
        "PB-SCM",
        "PB-SCM-PGF",
    )
    expanded = common[:5]
    common_high = float(
        np.nanmax(summary.loc[summary["regime"] == "common", "plot_ci95_high"])
    )
    expanded_high = float(
        np.nanmax(summary.loc[summary["regime"] == "expanded", "plot_ci95_high"])
    )
    fig, _ = six_panel_figure(
        summary,
        "SHD (lower is better)",
        common,
        expanded,
        {"LibraryDP", "LibraryGreedy", "OracleDP"},
        ((0, common_high * 1.05), (0, expanded_high * 1.05)),
    )
    save_figure(
        fig,
        APP_FIG,
        "figA1_shd_main_six_family_final",
        summary,
        {
            "metric_definition": "Directed SHD for fully directed methods: missing/extra/reversed edge each costs one and each unordered pair is counted once. PC-RCIT, PB-SCM, and PB-SCM-PGF use skeleton SHD consistently because their saved outputs can be mixed/partially directed.",
            "methods_common": list(common),
            "methods_expanded": list(expanded),
            "caption": "Structural Hamming distance in the main heterogeneous six-family simulations. Lower values indicate better structural recovery. For directed SHD, a reversed edge is counted as one operation. PC-RCIT and the PB-SCM variants are reported using skeleton SHD because their saved outputs can be partially directed.",
        },
    )
    return summary


def _summarize_saved_runtime_components(
    frame: pd.DataFrame,
    x_column: str,
    sweep: str,
    source: str,
    protocol: str,
) -> pd.DataFrame:
    component_columns = {
        "local_score": "local_score_runtime_sec",
        "dp_search": "search_runtime_sec",
    }
    rows = []
    for x_value, group in frame.groupby(x_column, sort=True):
        for component, column in component_columns.items():
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            se = sd / math.sqrt(len(values))
            rows.append(
                {
                    "runtime_sweep": sweep,
                    "x_value": float(x_value),
                    "component": component,
                    "R": int(len(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "mean": mean,
                    "sd": sd,
                    "se": se,
                    "ci95_low": max(0.0, mean - 1.96 * se),
                    "ci95_high": mean + 1.96 * se,
                    "source_file": source,
                    "protocol": protocol,
                }
            )
    return pd.DataFrame(rows)


def load_saved_runtime_component_summaries() -> pd.DataFrame:
    d_path = MAIN_RESULTS / "experiment1_d_sweep/experiment1_d_sweep_raw.csv"
    n_path = MAIN_RESULTS / "experiment2_N_sweep/experiment2_N_sweep_raw.csv"
    d_raw = pd.read_csv(d_path)
    n_raw = pd.read_csv(n_path)
    d_raw = d_raw[
        (d_raw["alpha_regime"] == "common")
        & (d_raw["method"] == "LibraryDP")
    ].copy()
    n_raw = n_raw[
        (n_raw["alpha_regime"] == "common")
        & (n_raw["method"] == "LibraryDP")
    ].copy()
    protocol = (
        "Main heterogeneous six-family sensitivity experiment; common "
        "coefficient regime; LibraryDP; saved per-replication timings."
    )
    return pd.concat(
        [
            _summarize_saved_runtime_components(
                n_raw,
                "N",
                "N",
                str(n_path.resolve()),
                protocol + " d=8 and average indegree=1.5.",
            ),
            _summarize_saved_runtime_components(
                d_raw,
                "d",
                "d",
                str(d_path.resolve()),
                protocol + " N=3200 and average indegree=1.5.",
            ),
        ],
        ignore_index=True,
    )


def make_runtime_component_figure() -> pd.DataFrame:
    summary = load_saved_runtime_component_summaries()
    styles = {
        "local_score": {
            "color": "#0072B2",
            "marker": "s",
            "linestyle": "--",
            "ylabel": "Precompute/local-score runtime (seconds)",
            "title": "Precompute/local-score",
        },
        "dp_search": {
            "color": "#009E73",
            "marker": "^",
            "linestyle": ":",
            "ylabel": "Order-DP runtime (seconds)",
            "title": "Order-DP",
        },
    }
    panels = (
        ("N", "local_score", "(a) Precompute/local-score vs $N$"),
        ("N", "dp_search", "(b) Order-DP vs $N$"),
        ("d", "local_score", "(c) Precompute/local-score vs $d$"),
        ("d", "dp_search", "(d) Order-DP vs $d$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 6.3))
    for axis, (sweep, component, title) in zip(axes.flat, panels):
        values = summary[
            (summary["runtime_sweep"] == sweep)
            & (summary["component"] == component)
        ].sort_values("x_value")
        style = styles[component]
        x = values["x_value"].to_numpy(float)
        axis.plot(
            x,
            values["median"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=4.5,
        )
        axis.fill_between(
            x,
            values["q25"],
            values["q75"],
            color=style["color"],
            alpha=0.13,
            linewidth=0,
        )
        axis.set_ylim(0, max(1e-12, 1.12 * float(values["q75"].max())))
        axis.set_ylabel(style["ylabel"])
        axis.set_title(title)
        if sweep == "N":
            axis.set_xlabel("Sample size $N$")
            axis.set_xlim(0, 10500)
            axis.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
            configure_axis(axis)
        else:
            axis.set_xlabel("Number of nodes $d$")
            axis.set_xticks(values["x_value"].astype(int))
            configure_axis(axis)
    fig.tight_layout()
    save_figure(
        fig,
        APP_FIG,
        "figA1_runtime_components_final",
        summary,
        {
            "components_plotted": [
                "Precompute/local-score evaluation",
                "Order-DP recursion",
            ],
            "total_runtime_plotted": False,
            "summary_statistic_plotted": "Median with IQR band in every panel.",
            "ci_convention": "Not used in this runtime figure; medians and IQR bands are plotted.",
            "runtime_sources": sorted(summary["source_file"].unique().tolist()),
            "protocol": "Common-regime LibraryDP timings from the completed main heterogeneous six-family d- and N-sweeps.",
            "axes": "Linear, panel-specific y-axis ranges starting at zero. N and d both use linear x-axes; N ticks are uniformly spaced at 0, 2000, ..., 10000 so the shape can be assessed on the original sample-size scale.",
            "caption": "Runtime component diagnostics. The figure separates precompute/local-score evaluation from the final order-DP recursion. Total runtime is not plotted because it nearly coincides with the precompute/local-score component in the saved logs. Runtime values are computed only from existing saved timing logs; no additional timing run was performed.",
        },
    )
    make_runtime_component_tables(summary)
    write_runtime_log_audit(summary)
    return summary


def _component_runtime_table(
    summary: pd.DataFrame,
    sweep: str,
    x_label: str,
) -> pd.DataFrame:
    panel = summary[summary["runtime_sweep"] == sweep]
    local = panel[panel["component"] == "local_score"].set_index("x_value")
    search = panel[panel["component"] == "dp_search"].set_index("x_value")
    rows = []
    for x_value in sorted(local.index):
        rows.append(
            {
                x_label: int(x_value),
                "Median precompute/local-score runtime": float(
                    local.loc[x_value, "median"]
                ),
                "IQR precompute/local-score runtime": (
                    f"[{float(local.loc[x_value, 'q25']):.6g}, "
                    f"{float(local.loc[x_value, 'q75']):.6g}]"
                ),
                "Median order-DP runtime": float(
                    search.loc[x_value, "median"]
                ),
                "IQR order-DP runtime": (
                    f"[{float(search.loc[x_value, 'q25']):.6g}, "
                    f"{float(search.loc[x_value, 'q75']):.6g}]"
                ),
                "Repetitions": int(local.loc[x_value, "R"]),
            }
        )
    return pd.DataFrame(rows)


def make_runtime_component_tables(main_summary: pd.DataFrame) -> None:
    number_formats = {
        "Median precompute/local-score runtime": "{:.4f}",
        "Median order-DP runtime": "{:.6f}",
    }
    save_table(
        _component_runtime_table(main_summary, "N", "N"),
        APP_TABLE,
        "runtime_vs_N_components_table",
        number_formats,
    )
    save_table(
        _component_runtime_table(main_summary, "d", "d"),
        APP_TABLE,
        "runtime_vs_d_components_table",
        number_formats,
    )

    raw = pd.read_csv(EXTENDED_RESULTS / "runtime/runtime_per_replication.csv")
    saved = pd.read_csv(EXTENDED_RESULTS / "runtime/runtime_summary.csv")
    library = saved[saved["runtime_sweep"] == "library_size"].copy()
    local = library[library["component"] == "local_score"].set_index("x_value")
    search = library[library["component"] == "dp_search"].set_index("x_value")
    rows = []
    for size in sorted(local.index):
        family_rows = raw[
            (raw["runtime_sweep"] == "library_size")
            & (raw["x_value"] == size)
        ]
        rows.append(
            {
                "|I|": int(size),
                "Family set used": str(family_rows["families"].iloc[0]),
                "Median precompute/local-score runtime": float(
                    local.loc[size, "median"]
                ),
                "IQR precompute/local-score runtime": (
                    f"[{float(local.loc[size, 'q25']):.6g}, "
                    f"{float(local.loc[size, 'q75']):.6g}]"
                ),
                "Median order-DP runtime": float(
                    search.loc[size, "median"]
                ),
                "IQR order-DP runtime": (
                    f"[{float(search.loc[size, 'q25']):.6g}, "
                    f"{float(search.loc[size, 'q75']):.6g}]"
                ),
                "Repetitions": int(local.loc[size, "R"]),
            }
        )
    library_table = pd.DataFrame(rows)
    save_table(
        library_table,
        APP_TABLE,
        "runtime_vs_library_size_components_table",
        number_formats,
    )
    path = APP_TABLE / "runtime_vs_library_size_components_table.tex"
    note = (
        "\n\\par\\smallskip\n"
        "\\begin{minipage}{\\linewidth}\\footnotesize\n"
        "Note: Runtime as a function of $|\\mathcal{I}|$ reflects both "
        "the number of working families and heterogeneous family-specific "
        "local-scoring costs. It should not be interpreted as a controlled "
        "test of strict linear wall-clock scaling in "
        "$|\\mathcal{I}|$.\n"
        "\\end{minipage}\n"
    )
    path.write_text(path.read_text(encoding="utf-8") + note, encoding="utf-8")


def write_runtime_log_audit(main_summary: pd.DataFrame) -> None:
    dedicated_raw = pd.read_csv(
        EXTENDED_RESULTS / "runtime/runtime_per_replication.csv"
    )
    dedicated_summary = pd.read_csv(
        EXTENDED_RESULTS / "runtime/runtime_summary.csv"
    )
    n_values = sorted(
        main_summary.loc[
            main_summary["runtime_sweep"] == "N", "x_value"
        ].astype(int).unique()
    )
    d_values = sorted(
        main_summary.loc[
            main_summary["runtime_sweep"] == "d", "x_value"
        ].astype(int).unique()
    )
    library_values = sorted(
        dedicated_raw.loc[
            dedicated_raw["runtime_sweep"] == "library_size", "x_value"
        ].astype(int).unique()
    )
    text = f"""# Runtime log audit

This audit uses saved CSV files only. No simulation, estimator, baseline, or
new timing diagnostic was run.

## Available sweeps

- Runtime versus N: available in
  `ptsem_six_family_final_results/experiment2_N_sweep/experiment2_N_sweep_raw.csv`.
  Values: {n_values}. The final component figure/table uses common-regime
  LibraryDP timings, with 100 saved replications per N.
- Runtime versus d: available in
  `ptsem_six_family_final_results/experiment1_d_sweep/experiment1_d_sweep_raw.csv`.
  Values: {d_values}. The final component figure/table uses common-regime
  LibraryDP timings, with 100 saved replications per d. A separate isolated
  timing log also contains 20 repetitions per d.
- Runtime versus working-library size |I|: available in
  `ptsem_extended_results/runtime/runtime_per_replication.csv`.
  Values: {library_values}, with 20 saved repetitions per library size.

## Available components

- Precompute/local-score evaluation: available as
  `local_score_runtime_sec` in the main sensitivity logs and `local_score` in
  the isolated timing log.
- Order-DP recursion / DP search: available as `search_runtime_sec` in the
  main sensitivity logs and `dp_search` in the isolated timing log.
- Total runtime: available as `runtime_sec` / `total`, but deliberately
  excluded from the final component figure and LaTeX runtime tables because
  it nearly coincides with local-score runtime.

## Available summaries

The saved per-replication logs permit median, IQR, mean, sample SD, and SE.
The isolated `runtime_summary.csv` already stores all five statistics
(together with confidence intervals). The final figure uses median and IQR
consistently in every panel.

## Reproducibility confirmation

No new timing run was performed. All displayed and tabulated values are
post-processed from the saved logs listed above.
"""
    (META / "runtime_log_audit.md").write_text(text, encoding="utf-8")


def _compact_summary_table(
    summary: pd.DataFrame,
    metric_definition: pd.Series | None = None,
) -> pd.DataFrame:
    selected = summary[
        (summary["sweep"] == "N-sweep")
        & (summary["x_value"].isin([100, 800, 3200, 10000]))
    ].copy()
    if metric_definition is not None:
        selected["Metric convention"] = metric_definition.loc[selected.index]
    columns = [
        "regime",
        "x_value",
        "method",
        "mean",
        "sd",
        "se",
        "ci95_low",
        "ci95_high",
        "N_nonNA",
    ]
    if "Metric convention" in selected:
        columns.insert(3, "Metric convention")
    selected = selected[columns].rename(
        columns={
            "regime": "Regime",
            "x_value": "N",
            "method": "Method",
            "mean": "Mean",
            "sd": "SD",
            "se": "SE",
            "ci95_low": "CI95 low",
            "ci95_high": "CI95 high",
            "N_nonNA": "Replications",
        }
    )
    selected["N"] = selected["N"].astype(int)
    return selected.sort_values(["Regime", "N", "Method"]).reset_index(
        drop=True
    )


def make_appendix_diagnostic_tables(raw: pd.DataFrame) -> None:
    internal_methods = ("LibraryDP", "LibraryGreedy", "OracleDP")
    conditional_rmse = aggregate_metric(
        raw,
        "alpha_rmse_on_correct_edges",
        "conditional_alpha_rmse_correct_edges",
        methods=internal_methods,
        valid_domain=(0, None),
    )
    save_table(
        _compact_summary_table(conditional_rmse),
        APP_TABLE,
        "conditional_alpha_rmse_summary_table",
        {
            "Mean": "{:.4f}",
            "SD": "{:.4f}",
            "SE": "{:.4f}",
            "CI95 low": "{:.4f}",
            "CI95 high": "{:.4f}",
        },
    )
    penalized = aggregate_metric(
        raw,
        "alpha_rmse_on_true_edges",
        "structure_penalized_alpha_rmse",
        methods=internal_methods,
        valid_domain=(0, None),
    )
    save_table(
        _compact_summary_table(penalized),
        APP_TABLE,
        "structure_penalized_alpha_rmse_summary_table",
        {
            "Mean": "{:.4f}",
            "SD": "{:.4f}",
            "SE": "{:.4f}",
            "CI95 low": "{:.4f}",
            "CI95 high": "{:.4f}",
        },
    )

    coverage_raw = raw.copy()
    coverage_raw["correct_edge_coverage"] = (
        coverage_raw["n_correct_directed_edges"]
        / coverage_raw["n_true_edges"]
    )
    coverage = aggregate_metric(
        coverage_raw,
        "correct_edge_coverage",
        "correct_edge_coverage",
        methods=internal_methods,
        valid_domain=(0, 1),
    )
    save_table(
        _compact_summary_table(coverage),
        APP_TABLE,
        "correct_edge_coverage_summary_table",
        {
            "Mean": "{:.4f}",
            "SD": "{:.4f}",
            "SE": "{:.4f}",
            "CI95 low": "{:.4f}",
            "CI95 high": "{:.4f}",
        },
    )

    shd_raw = pd.read_csv(
        EXTENDED_RESULTS
        / "existing_results_augmented/existing_per_replication_metrics.csv"
    )
    shd_raw["method"] = shd_raw["method"].map(METHOD_MAP)
    shd_raw["alpha_regime"] = shd_raw["coefficient_regime"]
    shd_raw["sweep_type"] = shd_raw["sweep_type"].map(
        {
            "d_sweep": "d-sweep",
            "N_sweep": "N-sweep",
            "density_sweep": "density-sweep",
        }
    )
    mixed_methods = {"PC-RCIT", "PB-SCM", "PB-SCM-PGF"}
    shd_raw["shd_for_table"] = np.where(
        shd_raw["method"].isin(mixed_methods),
        shd_raw["skeleton_shd"],
        shd_raw["directed_shd_rev1"],
    )
    shd_raw["shd_definition"] = np.where(
        shd_raw["method"].isin(mixed_methods),
        "Skeleton SHD",
        "Directed SHD (reversal cost 1)",
    )
    shd = aggregate_metric(
        shd_raw,
        "shd_for_table",
        "shd_diagnostic",
        valid_domain=(0, None),
    )
    definitions = (
        shd_raw[
            [
                "alpha_regime",
                "sweep_type",
                "x_value",
                "method",
                "shd_definition",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "alpha_regime": "regime",
                "sweep_type": "sweep",
            }
        )
    )
    shd = shd.merge(
        definitions,
        on=["regime", "sweep", "x_value", "method"],
        how="left",
        validate="one_to_one",
    )
    shd_table = _compact_summary_table(
        shd, shd["shd_definition"]
    )
    save_table(
        shd_table,
        APP_TABLE,
        "shd_summary_table",
        {
            "Mean": "{:.3f}",
            "SD": "{:.3f}",
            "SE": "{:.3f}",
            "CI95 low": "{:.3f}",
            "CI95 high": "{:.3f}",
        },
    )


def make_table1() -> pd.DataFrame:
    rows = [
        ("True exogenous families", "Poisson, NB, ZIP, Geom, Binomial, Bernoulli"),
        ("Working family library", "Poisson, NB, ZIP, Geom, Binomial, Bernoulli"),
        ("Family assignment", "Independent uniform node-wise assignment"),
        ("Common coefficients", "alpha ~ Uniform(0.15, 0.85)"),
        ("Expanded coefficients", "alpha ~ Uniform(0.2, 2.0)"),
        ("DAG generation", "Exact-edge ordered DAG"),
        ("Maximum parents", "5"),
        ("Replications", "R = 100 per cell"),
        ("d-sweep", "d = 4,...,10; N = 3200; average indegree = 1.5"),
        ("N-sweep", "N = 100,200,400,800,1600,3200,6400,10000; d = 8; average indegree = 1.5"),
        ("Density-sweep", "Average indegree = 1.0,1.5,2.0,2.5,3.0; d = 8; N = 3200"),
        ("Common methods", "LibraryDP, LibraryGreedy, OracleDP, PC-RCIT, ODS, PB-SCM, PB-SCM-PGF"),
        ("Expanded methods", "LibraryDP, LibraryGreedy, OracleDP, PC-RCIT, ODS"),
        ("Base seed", "20260622"),
    ]
    frame = pd.DataFrame(rows, columns=["Setting", "Value"])
    save_table(frame, MAIN_TABLE, "table1_simulation_settings")
    return frame


def load_nba_sources() -> tuple[pd.DataFrame, ...]:
    optimization = BASKETBALL / "outputs/expanded_candidate_study/optimization/foul_leaves_5_team"
    baselines = BASKETBALL / "outputs/expanded_candidate_study/baselines/foul_leaves_5_team"
    graph_summary = pd.read_csv(optimization / "summary.csv")
    baseline_summary = pd.read_csv(baselines / "method_summary.csv")
    coefficients = pd.read_csv(
        BASKETBALL / "outputs/expanded_candidate_study/final_five_node_coefficients.csv"
    )
    selected = pd.read_csv(
        EXTENDED_RESULTS / "nba/nba_selected_families_and_exogenous_parameters.csv"
    )
    sanity = pd.read_csv(EXTENDED_RESULTS / "nba/nba_ft_percent_sanity_check.csv")
    bootstrap = pd.read_csv(EXTENDED_RESULTS / "nba/nba_bootstrap_summary.csv")
    edge_stability = pd.read_csv(
        EXTENDED_RESULTS / "nba/nba_bootstrap_edge_selection_frequency.csv"
    )
    return (
        graph_summary,
        baseline_summary,
        coefficients,
        selected,
        sanity,
        bootstrap,
        edge_stability,
    )


def make_nba_tables() -> None:
    (
        graph_summary,
        baseline_summary,
        coefficients,
        selected,
        sanity,
        bootstrap,
        edge_stability,
    ) = load_nba_sources()

    ptsem = pd.DataFrame(
        [
            {
                "Method": "LibraryDP",
                "Skeleton precision": graph_summary.iloc[0]["mean_skeleton_precision"],
                "Skeleton recall": graph_summary.iloc[0]["mean_skeleton_recall"],
                "Skeleton F1": graph_summary.iloc[0]["mean_skeleton_f1"],
                "Directed precision": graph_summary.iloc[0]["mean_directed_precision"],
                "Directed recall": graph_summary.iloc[0]["mean_directed_recall"],
                "Directed F1": graph_summary.iloc[0]["mean_directed_f1"],
                "Exact recovery count": int(graph_summary.iloc[0]["exact_count"]),
                "Model applicable": "Yes",
            }
        ]
    )
    baseline_names = {
        "Poisson DAG (ODS-style)": "ODS",
        "PC (RCIT)": "PC-RCIT",
        "PB-SCM (Cumulant)": "PB-SCM",
        "PB-SCM (PGF)": "PB-SCM-PGF",
    }
    base = baseline_summary.copy()
    base["Method"] = base["method"].map(baseline_names)
    base_table = pd.DataFrame(
        {
            "Method": base["Method"],
            "Skeleton precision": base["mean_skeleton_precision"],
            "Skeleton recall": base["mean_skeleton_recall"],
            "Skeleton F1": base["mean_skeleton_f1"],
            "Directed precision": base["mean_directed_precision"],
            "Directed recall": base["mean_directed_recall"],
            "Directed F1": base["mean_directed_f1"],
            "Exact recovery count": base["exact_recovery_count"].astype(int),
            "Model applicable": np.where(
                base["model_applicable_all_seasons"], "Yes", "No"
            ),
        }
    )
    table2 = pd.concat([ptsem, base_table], ignore_index=True)
    table2["Method"] = pd.Categorical(
        table2["Method"],
        categories=("LibraryDP", "ODS", "PC-RCIT", "PB-SCM", "PB-SCM-PGF"),
        ordered=True,
    )
    table2 = table2.sort_values("Method").reset_index(drop=True)
    table2["Method"] = table2["Method"].astype(str)
    save_table(
        table2,
        NBA_TABLE,
        "nba_table2_graph_recovery",
        {
            "Skeleton precision": "{:.3f}",
            "Skeleton recall": "{:.3f}",
            "Skeleton F1": "{:.3f}",
            "Directed precision": "{:.3f}",
            "Directed recall": "{:.3f}",
            "Directed F1": "{:.3f}",
            "Exact recovery count": "{:.0f}",
        },
    )

    coefficient_columns = (
        "FOUL->FTA",
        "FTA->FTM",
        "FOUL->PERS_FOUL_DRAWN",
        "FOUL->LOOSE_BALL_FOUL_DRAWN",
    )
    coefficient_labels = {
        "FOUL->FTA": "FOUL->FTA",
        "FTA->FTM": "FTA->FTM",
        "FOUL->PERS_FOUL_DRAWN": "FOUL->PERS",
        "FOUL->LOOSE_BALL_FOUL_DRAWN": "FOUL->LOOSE",
    }
    season_values = coefficients[coefficients["season"] != "mean"].copy()
    table3_rows = []
    for column in coefficient_columns:
        values = pd.to_numeric(season_values[column], errors="coerce")
        table3_rows.append(
            {
                "Edge": coefficient_labels[column],
                "Six-season mean": values.mean(),
                "Minimum": values.min(),
                "Maximum": values.max(),
                "Range": f"[{values.min():.3f}, {values.max():.3f}]",
            }
        )
    table3 = pd.DataFrame(table3_rows)
    save_table(
        table3,
        NBA_TABLE,
        "nba_table3_coefficients",
        {
            "Six-season mean": "{:.3f}",
            "Minimum": "{:.3f}",
            "Maximum": "{:.3f}",
        },
    )

    table4 = selected[["Season", "FOUL", "FTA", "FTM", "PERS", "LOOSE"]].copy()
    save_table(table4, NBA_TABLE, "nba_table4_selected_families")

    table5 = sanity[
        [
            "Season",
            "empirical_FT_percent",
            "fitted_FTA_to_FTM_coefficient",
            "estimated_FTM_exogenous_mean",
            "estimated_FTM_exogenous_variance",
            "difference_alpha_minus_empirical_FT_percent",
        ]
    ].rename(
        columns={
            "empirical_FT_percent": "Empirical FTM/FTA",
            "fitted_FTA_to_FTM_coefficient": "Fitted FTA->FTM",
            "estimated_FTM_exogenous_mean": "FTM exogenous mean",
            "estimated_FTM_exogenous_variance": "FTM exogenous variance",
            "difference_alpha_minus_empirical_FT_percent": "Fitted - empirical",
        }
    )
    save_table(
        table5,
        NBA_TABLE,
        "nba_table5_ft_sanity",
        {
            "Empirical FTM/FTA": "{:.4f}",
            "Fitted FTA->FTM": "{:.4f}",
            "FTM exogenous mean": "{:.2e}",
            "FTM exogenous variance": "{:.2e}",
            "Fitted - empirical": "{:.2e}",
        },
    )

    bootstrap_table = bootstrap[
        [
            "Season",
            "B_total",
            "N_success",
            "failure_rate",
            "complete_reference_graph_recovery_frequency",
            "mean_extra_edges",
        ]
    ].rename(
        columns={
            "B_total": "B",
            "N_success": "Successful",
            "failure_rate": "Failure rate",
            "complete_reference_graph_recovery_frequency": "Exact graph frequency",
            "mean_extra_edges": "Mean extra edges",
        }
    )
    save_table(
        bootstrap_table,
        NBA_TABLE,
        "nba_appendix_block_bootstrap_stability",
        {
            "B": "{:.0f}",
            "Successful": "{:.0f}",
            "Failure rate": "{:.3f}",
            "Exact graph frequency": "{:.3f}",
            "Mean extra edges": "{:.2f}",
        },
    )
    atomic_csv(
        edge_stability,
        NBA_TABLE / "nba_appendix_edge_selection_frequency.csv",
    )


def make_nba_graph() -> None:
    positions = {
        "FOUL": (0.12, 0.52),
        "FTA": (0.46, 0.72),
        "FTM": (0.82, 0.72),
        "PERS": (0.48, 0.30),
        "LOOSE": (0.82, 0.30),
    }
    edges = (
        ("FOUL", "FTA"),
        ("FTA", "FTM"),
        ("FOUL", "PERS"),
        ("FOUL", "LOOSE"),
    )
    fig, axis = plt.subplots(figsize=(7.2, 3.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    for node, (x, y) in positions.items():
        box = FancyBboxPatch(
            (x - 0.075, y - 0.075),
            0.15,
            0.15,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor="#F7F7F7",
            edgecolor="#333333",
            linewidth=1.2,
            zorder=3,
        )
        axis.add_patch(box)
        axis.text(x, y, node, ha="center", va="center", fontsize=10, zorder=4)
    for source, target in edges:
        start = np.array(positions[source])
        end = np.array(positions[target])
        direction = end - start
        length = np.linalg.norm(direction)
        unit = direction / length
        arrow = FancyArrowPatch(
            start + 0.095 * unit,
            end - 0.095 * unit,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#333333",
            connectionstyle="arc3,rad=0.0",
            zorder=2,
        )
        axis.add_patch(arrow)
    axis.set_title("Rule-implied NBA benchmark graph", pad=8)
    fig.tight_layout()
    graph_data = pd.DataFrame(edges, columns=["source", "target"])
    save_figure(
        fig,
        NBA_FIG,
        "fig5_rule_implied_nba_graph_final",
        graph_data,
        {
            "graph_role": "Rule-implied five-node benchmark used for Section 6 recovery metrics.",
            "edges": [f"{source}->{target}" for source, target in edges],
        },
    )


def make_nba_diagnostics_figure() -> pd.DataFrame:
    (
        _graph_summary,
        _baseline_summary,
        coefficients,
        selected,
        _sanity,
        _bootstrap,
        _edge_stability,
    ) = load_nba_sources()
    seasons = selected["Season"].astype(str).tolist()
    x = np.arange(len(seasons))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 3.75),
        gridspec_kw={"width_ratios": [1.15, 1.30]},
    )
    plot_rows: list[dict[str, object]] = []

    # Panel (a): season-specific coefficients.
    axis = axes[0]
    coefficient_frame = (
        coefficients[coefficients["season"] != "mean"]
        .set_index("season")
        .reindex(seasons)
    )
    coefficient_specs = (
        ("FOUL->FTA", "FOUL->FTA", "#CC3366", "o", "-"),
        ("FTA->FTM", "FTA->FTM", "#0072B2", "s", "--"),
        (
            "FOUL->PERS_FOUL_DRAWN",
            "FOUL->PERS",
            "#009E73",
            "^",
            "-.",
        ),
        (
            "FOUL->LOOSE_BALL_FOUL_DRAWN",
            "FOUL->LOOSE",
            "#E69F00",
            "D",
            ":",
        ),
    )
    for column, label, color, marker, linestyle in coefficient_specs:
        values = coefficient_frame[column].to_numpy(float)
        axis.plot(
            x,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.9,
            markersize=4.2,
            label=label,
        )
        for season, value in zip(seasons, values):
            plot_rows.append(
                {
                    "panel": "a_coefficients",
                    "Season": season,
                    "series": label,
                    "value": float(value),
                }
            )
    axis.set_ylim(0, 1.38)
    axis.set_ylabel("Fitted coefficient")
    axis.legend(
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
    )
    axis.set_xticks(x, seasons, rotation=35, ha="right")
    configure_axis(axis)

    # Panel (b): selected exogenous families.
    axis = axes[1]
    nodes = ("FOUL", "FTA", "FTM", "PERS", "LOOSE")
    family_colors = {
        "Poisson": "#4C78A8",
        "NB": "#F58518",
        "ZIP": "#54A24B",
        "Geom": "#E45756",
        "Binomial": "#B279A2",
        "Bernoulli": "#72B7B2",
    }
    family_to_code = {family: index for index, family in enumerate(FAMILIES)}
    selected_by_season = selected.set_index("Season").reindex(seasons)
    matrix = np.array(
        [
            [
                family_to_code[str(selected_by_season.loc[season, node])]
                for node in nodes
            ]
            for season in seasons
        ],
        dtype=int,
    )
    cmap = ListedColormap([family_colors[family] for family in FAMILIES])
    axis.imshow(
        matrix,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(FAMILIES) - 0.5,
        aspect="auto",
    )
    for row_index, season in enumerate(seasons):
        for column_index, node in enumerate(nodes):
            family = str(selected_by_season.loc[season, node])
            axis.text(
                column_index,
                row_index,
                family,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if family in {"Poisson", "Geom", "Binomial"} else "#222222",
            )
            plot_rows.append(
                {
                    "panel": "b_selected_working_families",
                    "Season": season,
                    "node": node,
                    "selected_working_family": family,
                    "family_code": family_to_code[family],
                }
            )
    axis.set_xticks(range(len(nodes)), nodes)
    axis.set_yticks(range(len(seasons)), seasons)
    family_handles = [
        Patch(facecolor=family_colors[family], edgecolor="none", label=family)
        for family in FAMILIES
    ]
    axis.legend(
        handles=family_handles,
        frameon=False,
        fontsize=6.5,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.78,
        bottom=0.25,
        wspace=0.34,
    )
    plotdata = pd.DataFrame(plot_rows)
    save_figure(
        fig,
        NBA_DIAGNOSTIC_FIG,
        "fig6_nba_diagnostics_final",
        plotdata,
        {
            "source_files": [
                str(
                    (
                        BASKETBALL
                        / "outputs/expanded_candidate_study/final_five_node_coefficients.csv"
                    ).resolve()
                ),
                str(
                    (
                        EXTENDED_RESULTS
                        / "nba/nba_selected_families_and_exogenous_parameters.csv"
                    ).resolve()
                ),
            ],
            "ft_sanity_table": "nba_tables/nba_table5_ft_sanity.csv and nba_tables/nba_table5_ft_sanity.tex",
            "ft_interpretation": "The removed FT% sanity check is retained as a numerical table. The fitted FTA->FTM coefficient is interpreted as a mean-scale marginal effect; its agreement with empirical FTM/FTA is discussed in the text, not shown as a separate figure panel.",
            "family_interpretation": "Family labels are interpreted as working choices for BIC scoring, not as uniquely identifiable distributional labels.",
            "bootstrap_included": False,
            "caption": "NBA diagnostic summaries. Panel (a) reports season-specific PT-SEM coefficients on the rule-implied reference edges. Panel (b) shows selected node-wise working families by season. Working-family labels are interpreted as scoring choices rather than uniquely identifiable distributional labels.",
            "ci_convention": "Not applicable; Figure 6 displays six completed season-level fits.",
        },
    )
    return plotdata


def make_nba_bootstrap_figure() -> pd.DataFrame:
    (
        _graph_summary,
        _baseline_summary,
        _coefficients,
        _selected,
        _sanity,
        bootstrap,
        edge_stability,
    ) = load_nba_sources()
    edge_labels = {
        "FOUL->FTA": "FOUL->FTA",
        "FTA->FTM": "FTA->FTM",
        "FOUL->PERS_FOUL_DRAWN": "FOUL->PERS",
        "FOUL->LOOSE_BALL_FOUL_DRAWN": "FOUL->LOOSE",
    }
    reference = edge_stability[
        edge_stability["is_reference_edge"].astype(bool)
    ].copy()
    reference["edge_label"] = reference["edge"].map(edge_labels)
    reference = reference[reference["edge_label"].notna()].copy()
    seasons = bootstrap["Season"].astype(str).tolist()
    edge_order = ("FOUL->FTA", "FTA->FTM", "FOUL->PERS", "FOUL->LOOSE")
    matrix = (
        reference.pivot(
            index="Season",
            columns="edge_label",
            values="selection_frequency",
        )
        .reindex(index=seasons, columns=edge_order)
        .to_numpy(float)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.65),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    image = axes[0].imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for i in range(len(seasons)):
        for j in range(len(edge_order)):
            axes[0].text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if matrix[i, j] >= 0.55 else "#222222",
            )
    axes[0].set_xticks(range(len(edge_order)), edge_order, rotation=30, ha="right")
    axes[0].set_yticks(range(len(seasons)), seasons)
    axes[0].set_title("(a) Reference-edge selection")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    colorbar.set_label("Selection frequency")

    x = np.arange(len(seasons))
    recovery = bootstrap[
        "complete_reference_graph_recovery_frequency"
    ].to_numpy(float)
    axes[1].bar(x, recovery, color="#7A5195", width=0.68)
    axes[1].set_xticks(x, seasons, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Recovery frequency")
    axes[1].set_title("(b) Complete reference graph")
    configure_axis(axes[1])
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.25,
        wspace=0.40,
    )

    edge_plotdata = reference[
        [
            "Season",
            "edge_label",
            "selection_frequency",
            "N_success",
        ]
    ].rename(columns={"edge_label": "edge"})
    edge_plotdata.insert(0, "panel", "a_reference_edge_selection")
    recovery_plotdata = bootstrap[
        [
            "Season",
            "complete_reference_graph_recovery_frequency",
            "B_total",
            "N_success",
            "failure_rate",
            "mean_extra_edges",
        ]
    ].copy()
    recovery_plotdata.insert(0, "panel", "b_complete_reference_graph")
    plotdata = pd.concat(
        [edge_plotdata, recovery_plotdata],
        ignore_index=True,
        sort=False,
    )
    save_figure(
        fig,
        APP_FIG,
        "figA2_nba_bootstrap_stability_final",
        plotdata,
        {
            "source_files": [
                str(
                    (
                        EXTENDED_RESULTS
                        / "nba/nba_bootstrap_edge_selection_frequency.csv"
                    ).resolve()
                ),
                str(
                    (
                        EXTENDED_RESULTS
                        / "nba/nba_bootstrap_summary.csv"
                    ).resolve()
                ),
            ],
            "role": "Appendix-only stability diagnostic; not a main positive benchmark claim.",
            "caption": "Game-level block-bootstrap diagnostics. Core reference edges are stable in most seasons, while complete graph recovery is less stable in some seasons due to extra-edge selection. These results are reported as stability diagnostics rather than as a main positive benchmark claim.",
            "ci_convention": "Not applicable; cells and bars are saved bootstrap frequencies.",
        },
    )
    return plotdata


def write_metadata_documents() -> None:
    readme = """# Final PT-SEM paper figures and tables

This directory is the final paper-production package. It was generated only
from completed result CSV files; no simulation, estimator, or baseline was
rerun.

## Main numerical figures

1. Figure 1 uses Directed F1 as the primary directed-graph recovery metric.
2. Figure 2 uses conditional alpha-MAPE (%) only on true directed edges
   recovered with the correct orientation within each replication. It is not
   conditioned on exact graph recovery and excludes missed or wrongly
   oriented true edges. Replications with no correctly recovered true
   directed edge are NA rather than zero and are excluded from that cell's
   mean. Simulated coefficients are bounded away from zero, so the relative
   error is well-defined.
3. Figure 3 reports working-family diagnostics. Labels are working choices
   for finite-family scoring, not uniquely identifiable distribution labels
   in all overlapping/boundary cases. Working-family accuracy is pooled over
   all nodes and all replications in a setting and is not conditioned on exact
   graph recovery.
4. Figure 4 is the all-Poisson additive-overlap/library-cost sensitivity
   analysis, shown as a single Directed-F1 panel. The paired profiling-cost
   gap remains available numerically from the same shared replications but is
   not drawn in the main-text figure.

## NBA figures

- Figure 5 is the rule-implied five-node NBA benchmark graph.
- Figure 6 reports season-specific PT-SEM coefficients and selected node-wise
  working families. Working-family labels are BIC scoring choices rather than
  uniquely identifiable distributional labels. The FT% sanity check remains
  in NBA Table 5 and is discussed numerically in the text.

## Appendix figures

- Figure A1 is appendix-only runtime component diagnostics versus N and d.
  It separates precompute/local-score evaluation from order-DP recursion,
  uses panel-specific linear y-axes, and reports medians with IQR bands from
  saved common-regime LibraryDP sensitivity logs. Both N and d use linear
  x-axes so runtime growth is assessed on the original scale. Total runtime
  is not shown.
- SHD, conditional alpha-RMSE, structure-penalized alpha-RMSE, and
  correct-edge coverage are supplied as compact appendix tables at
  representative N-sweep values rather than repeated 2x3 figures.
- Runtime versus working-library size is supplied as a table only and should
  not be interpreted as strict empirical linear wall-clock scaling in
  working-library size.
- Figure A2 reports existing game-level block-bootstrap stability diagnostics.
  It is appendix-only and is not presented as a main positive benchmark claim.

## Uncertainty

All mean-curve uncertainty uses pointwise 95% Monte Carlo confidence
intervals: mean +/- 1.96 SE, with sample SD computed using ddof=1 and
SE=SD/sqrt(N_nonNA). No SD bands are used. Plotting CSVs retain unclipped
mean, SD, SE, and CI endpoints; clipped endpoints are stored separately for
bounded display domains.

## Deliberate exclusions

Out-of-library misspecification results are deliberately excluded from all
final paper figures, appendix figures, captions, and tables. They remain
internal diagnostics and are not part of the finite-library theory-driven
experimental claim. No out-of-library numerical result is contained here.

## NBA benchmark

The fitted FTA->FTM coefficient is interpreted as the marginal increase in
expected made free throws per additional free-throw attempt. Its agreement
with empirical FTM/FTA, together with near-zero FTM exogenous mean, provides
an external sanity check on the mean-scale interpretation. It is not a
theoretical identity.

NBA family labels are interpreted as working choices for BIC scoring, not as
uniquely identifiable distributional labels.

PB-SCM and PB-SCM-PGF are marked model-inapplicable for the five-node NBA raw
counts because positive reference-edge coefficients can exceed one. Their
reported graph metrics are retained as diagnostics and are not presented as
fully applicable model comparisons.
"""
    (META / "final_figure_readme.md").write_text(readme, encoding="utf-8")

    definitions = """# Metric definitions

## Directed F1
F1 of the true and estimated correctly oriented directed-edge sets.
Unresolved PC-RCIT edges are not counted as two directions.

## Conditional alpha-MAPE
Within each replication, 100 times the mean absolute relative alpha error
over true directed edges selected with the correct orientation. It is
expressed in percent, is not restricted to replications with exact graph
recovery, and excludes missed or wrongly oriented true edges. If no such edge
exists, the replication value is NA rather than zero. Cell summaries use
non-NA replications only. Simulated coefficients are bounded away from zero.

## Conditional alpha-RMSE
The former absolute-scale conditional coefficient diagnostic is retained as
a compact appendix table and uses the same correctly oriented-edge and NA
rules as conditional alpha-MAPE.

## Structure-penalized alpha-RMSE
RMSE over all true directed edges. A missed or wrongly oriented true edge has
estimated coefficient zero.

## Correct-edge coverage
Number of true directed edges recovered with correct orientation divided by
the number of true directed edges.

## SHD
Primary directed SHD counts a missing edge, extra edge, or reversal as one
operation and counts each unordered pair once. PC-RCIT and the PB-SCM
variants are represented by skeleton SHD because their saved outputs can
contain unresolved directions.

## Working-family accuracy
Fraction of nodes whose selected working family exactly equals the generated
family, pooled over all nodes and all replications in a setting. It is not
conditioned on exact graph recovery. It is a scoring diagnostic, subject to
family overlap/boundary ambiguities.

## Runtime
Figure A1 and the N/d component tables use medians and IQRs over 100 saved
common-regime LibraryDP repetitions per setting from the completed main
sensitivity experiments. The working-library-size table uses 20 saved
isolated timing repetitions per setting. Mean, sample SD, and SE remain in
the Figure A1 plotting CSV.
"""
    (META / "metric_definitions.md").write_text(definitions, encoding="utf-8")

    warnings = """# Warnings and missing metrics

- No requested synthetic metric was missing. Conditional alpha-MAPE,
  conditional alpha-RMSE, and correct-edge coverage were read or reconstructed
  exactly from saved per-replication fields.
- Coefficient RMSE is restricted to LibraryDP, LibraryGreedy, and OracleDP
  because external-baseline coefficients are not PT-SEM thinning parameters.
- Mixed directed SHD is not forced from partially directed output; the SHD
  appendix table uses skeleton SHD for PC-RCIT and both PB-SCM variants and
  labels the convention explicitly.
- PB-SCM and PB-SCM-PGF are omitted from the expanded regime because alpha can
  exceed one.
- PB-SCM and PB-SCM-PGF are model-inapplicable to the final five-node NBA raw
  benchmark under their binomial-thinning coefficient domain; Table 2 marks
  this explicitly.
- The runtime-vs-library-size experiment is not plotted in the final package
  because family-specific local-score costs make it unsuitable as evidence of
  strict linear wall-clock scaling.
- Runtime-vs-N and runtime-vs-d component outputs use existing common-regime
  LibraryDP sensitivity logs. The library-size component table uses the
  existing isolated timing log. No new runtime value was generated.
- NBA bootstrap outputs are appendix-only stability diagnostics. Complete
  graph recovery is not uniformly stable across seasons and is not presented
  as a main positive benchmark claim.
- Out-of-library diagnostics are intentionally absent.
"""
    (META / "warnings_or_missing_metrics.md").write_text(
        warnings, encoding="utf-8"
    )


def write_manifest() -> None:
    files = sorted(
        path.relative_to(OUT).as_posix()
        for path in OUT.rglob("*")
        if path.is_file()
    )
    forbidden = [
        path
        for path in files
        if "out_of_library" in path.lower()
        or "poisson_mixture" in path.lower()
        or "zi_logseries" in path.lower()
    ]
    if forbidden:
        raise AssertionError(f"Forbidden internal-diagnostic files: {forbidden}")
    input_sources = [
        MAIN_RESULTS
        / "experiment1_d_sweep/experiment1_d_sweep_raw.csv",
        MAIN_RESULTS
        / "experiment2_N_sweep/experiment2_N_sweep_raw.csv",
        MAIN_RESULTS
        / "experiment3_kin_sweep/experiment3_kin_sweep_raw.csv",
        EXTENDED_RESULTS / "synthetic_per_replication.csv",
        EXTENDED_RESULTS
        / "existing_results_augmented/existing_per_replication_metrics.csv",
        EXTENDED_RESULTS / "runtime/runtime_per_replication.csv",
        EXTENDED_RESULTS / "runtime/runtime_summary.csv",
        BASKETBALL
        / "outputs/expanded_candidate_study/optimization/foul_leaves_5_team/summary.csv",
        BASKETBALL
        / "outputs/expanded_candidate_study/baselines/foul_leaves_5_team/method_summary.csv",
        BASKETBALL
        / "outputs/expanded_candidate_study/final_five_node_coefficients.csv",
        EXTENDED_RESULTS
        / "nba/nba_selected_families_and_exogenous_parameters.csv",
        EXTENDED_RESULTS / "nba/nba_ft_percent_sanity_check.csv",
        EXTENDED_RESULTS / "nba/nba_bootstrap_summary.csv",
        EXTENDED_RESULTS / "nba/nba_bootstrap_edge_selection_frequency.csv",
    ]
    manifest = {
        "simulation_rerun": False,
        "ci_convention": "pointwise mean +/- 1.96*SE; sample SD ddof=1; no SD bands",
        "method_labels": list(METHOD_ORDER),
        "out_of_library_numeric_results_included": False,
        "input_sources": [str(path.resolve()) for path in input_sources],
        "files": files,
    }
    (META / "plotting_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


DISPLAY_METHOD = {
    "LibraryDP": "Proposed DP-BIC",
    "LibraryGreedy": "Greedy-BIC",
    "OracleDP": "Oracle DP",
    "Fixed-Poisson DP": "Poisson-only DP",
    "ODS": "ODS",
    "PC-RCIT": "PC-RCIT",
    "PB-SCM": "PB-SCM",
    "PB-SCM-PGF": "PB-SCM-PGF",
}

DISPLAY_SWEEP = {
    "d-sweep": "Dimension sweep",
    "N-sweep": "Sample-size sweep",
    "density-sweep": "Average-indegree sweep",
}


def _six_panel_from_saved_summary(
    summary: pd.DataFrame,
    metric_label: str,
    restricted_methods: Sequence[str],
    extended_methods: Sequence[str],
    y_limits_by_panel: dict[tuple[str, str], tuple[float, float]],
) -> plt.Figure:
    """Draw a six-panel paper figure from an existing summary CSV only."""
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.25), sharey=False)
    handles: dict[str, object] = {}
    letters = iter("abcdef")
    for row, regime in enumerate(("common", "expanded")):
        methods = restricted_methods if regime == "common" else extended_methods
        for col, sweep in enumerate(SWEEP_ORDER):
            axis = axes[row, col]
            panel = summary[
                (summary["regime"] == regime) & (summary["sweep"] == sweep)
            ]
            for method in methods:
                handle = plot_summary_line(
                    axis,
                    panel,
                    method,
                    band=True,
                    label=DISPLAY_METHOD[method],
                )
                if handle is not None:
                    handles[method] = handle
            axis.set_title(
                f"({next(letters)}) {DISPLAY_SWEEP[sweep]}", fontsize=9.5
            )
            axis.set_xlabel(SWEEP_XLABEL[sweep])
            if col == 0:
                regime_label = "Restricted" if regime == "common" else "Extended"
                axis.set_ylabel(f"{regime_label}\n{metric_label}")
            axis.set_ylim(*y_limits_by_panel[(regime, sweep)])
            if sweep == "d-sweep":
                axis.set_xticks(sorted(panel["x_value"].unique().astype(int)))
            elif sweep == "density-sweep":
                axis.set_xticks(sorted(panel["x_value"].unique()))
            configure_axis(axis, sweep)
    ordered = [method for method in METHOD_ORDER if method in handles]
    fig.legend(
        [handles[method] for method in ordered],
        [DISPLAY_METHOD[method] for method in ordered],
        loc="lower center",
        ncol=min(7, len(ordered)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def redraw_figure1_from_saved_plotdata() -> Path:
    summary = pd.read_csv(MAIN_FIG / "fig1_directed_f1_final_plotdata.csv")
    restricted = (
        "LibraryDP", "LibraryGreedy", "OracleDP", "ODS", "PC-RCIT",
        "PB-SCM", "PB-SCM-PGF",
    )
    extended = restricted[:5]
    limits = {
        (regime, sweep): (0.0, 1.02)
        for regime in ("common", "expanded")
        for sweep in SWEEP_ORDER
    }
    fig = _six_panel_from_saved_summary(summary, "F1", restricted, extended, limits)
    path = MAIN_FIG / "fig1_directed_f1_final.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def redraw_figure2_from_saved_plotdata() -> Path:
    summary = pd.read_csv(
        MAIN_FIG / "fig2_conditional_alpha_mape_final_plotdata.csv"
    )
    methods = ("LibraryDP", "LibraryGreedy", "OracleDP")
    limits: dict[tuple[str, str], tuple[float, float]] = {}
    for regime in ("common", "expanded"):
        for sweep in SWEEP_ORDER:
            panel = summary[
                (summary["regime"] == regime) & (summary["sweep"] == sweep)
            ]
            limits[(regime, sweep)] = (
                0.0,
                max(1e-6, 1.12 * float(np.nanmax(panel["plot_ci95_high"]))),
            )
    fig = _six_panel_from_saved_summary(
        summary,
        r"Thinning-coefficient MAPE (%)",
        methods,
        methods,
        limits,
    )
    path = MAIN_FIG / "fig2_conditional_alpha_mape_final.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def redraw_figure4_from_saved_plotdata() -> Path:
    plotdata = pd.read_csv(
        MAIN_FIG / "fig4_all_poisson_library_cost_final_plotdata.csv"
    )
    summary = plotdata[plotdata["panel"] == "main_directed_f1"].copy()
    methods = ("Fixed-Poisson DP", "LibraryDP", "LibraryGreedy", "ODS", "PC-RCIT")
    fig, axis = plt.subplots(figsize=(6.7, 4.6))
    handles = []
    for method in methods:
        label = "Poisson DAG (ODS)" if method == "ODS" else DISPLAY_METHOD[method]
        handle = plot_summary_line(axis, summary, method, band=True, label=label)
        if handle is not None:
            handles.append(handle)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Sample size $N$")
    axis.set_ylabel("F1")
    axis.set_title("All-Poisson setting")
    configure_axis(axis, "N-sweep")
    axis.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.12, right=0.76, bottom=0.13, top=0.90)
    path = MAIN_FIG / "fig4_all_poisson_library_cost_final.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_saved_figure3_accuracy(axis: plt.Axes, accuracy: pd.DataFrame) -> None:
    """Draw Figure 3(a) verbatim from its saved plotting-data CSV."""
    labels = {"LibraryDP": "Proposed DP-BIC", "LibraryGreedy": "Greedy-BIC"}
    regimes = {"common": "restricted", "expanded": "extended"}
    for method in ("LibraryDP", "LibraryGreedy"):
        for regime, linestyle in (("common", "-"), ("expanded", "--")):
            panel = accuracy[
                (accuracy["method"] == method) & (accuracy["regime"] == regime)
            ].sort_values("x_value")
            style = STYLE[method]
            x = panel["x_value"].to_numpy(float)
            axis.plot(
                x,
                panel["mean"].to_numpy(float),
                color=style["color"],
                marker=style["marker"],
                linestyle=linestyle,
                linewidth=2,
                markersize=4.2,
                label=f"{labels[method]} - {regimes[regime]}",
            )
            axis.fill_between(
                x,
                panel["plot_ci95_low"].to_numpy(float),
                panel["plot_ci95_high"].to_numpy(float),
                color=style["color"],
                alpha=0.10,
                linewidth=0,
            )
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Sample size $N$")
    axis.set_ylabel("Family-selection accuracy")
    configure_axis(axis, "N-sweep")
    axis.legend(
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0,
    )


def _plot_saved_figure3_confusion(
    axis: plt.Axes, frame: pd.DataFrame
) -> matplotlib.image.AxesImage:
    """Draw a Figure 3 confusion matrix from saved row proportions."""
    matrix = (
        frame.pivot(
            index="true_family",
            columns="selected_working_family",
            values="row_proportion",
        )
        .reindex(index=FAMILIES, columns=FAMILIES)
        .to_numpy(float)
    )
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(6):
        for j in range(6):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] >= 0.5 else "#222222",
            )
    axis.set_xticks(range(6), FAMILIES, rotation=45, ha="right")
    axis.set_yticks(range(6), FAMILIES)
    axis.set_xlabel("Selected exogenous family")
    axis.set_ylabel("Generating exogenous family")
    return image


def redraw_figure3_from_saved_plotdata() -> list[Path]:
    """Export title-free Figure 3 panels without reading experiment-level data."""
    accuracy = pd.read_csv(
        MAIN_FIG / "fig3_working_family_diagnostics_final_plotdata.csv"
    )
    restricted = pd.read_csv(MAIN_FIG / "fig3_confusion_common_plotdata.csv")
    extended = pd.read_csv(MAIN_FIG / "fig3_confusion_expanded_plotdata.csv")
    outputs: list[Path] = []

    fig, axis = plt.subplots(figsize=(4.6, 3.65))
    _plot_saved_figure3_accuracy(axis, accuracy)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.76)
    path = MAIN_FIG / "fig3a_family_selection_accuracy_panel.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    for stem, frame, show_colorbar in (
        ("fig3b_restricted_exogenous_family_panel", restricted, False),
        ("fig3c_extended_exogenous_family_panel", extended, True),
    ):
        fig, axis = plt.subplots(figsize=(4.25, 3.75))
        image = _plot_saved_figure3_confusion(axis, frame)
        if show_colorbar:
            fig.colorbar(
                image, ax=axis, label="Row proportion", fraction=0.046, pad=0.04
            )
        fig.tight_layout()
        path = MAIN_FIG / f"{stem}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    fig, axes = plt.subplots(
        1, 3, figsize=(11.8, 3.6), gridspec_kw={"width_ratios": [1.25, 1, 1]}
    )
    _plot_saved_figure3_accuracy(axes[0], accuracy)
    image = _plot_saved_figure3_confusion(axes[1], restricted)
    _plot_saved_figure3_confusion(axes[2], extended)
    fig.subplots_adjust(left=0.065, right=0.90, top=0.78, bottom=0.22, wspace=0.42)
    color_axis = fig.add_axes([0.925, 0.26, 0.014, 0.52])
    fig.colorbar(image, cax=color_axis, label="Row proportion")
    path = MAIN_FIG / "fig3_working_family_diagnostics_final.pdf"
    fig.savefig(path, bbox_inches="tight")
    outputs.append(path)
    preview = MAIN_FIG / "fig3_exogenous_family_diagnostics_preview.png"
    fig.savefig(preview, dpi=400, bbox_inches="tight")
    outputs.append(preview)
    plt.close(fig)
    return outputs


def _saved_figure6_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    plotdata = pd.read_csv(NBA_DIAGNOSTIC_FIG / "fig6_nba_diagnostics_final_plotdata.csv")
    coefficients = plotdata[plotdata["panel"] == "a_coefficients"].copy()
    selected = plotdata[plotdata["panel"] == "b_selected_working_families"].copy()
    seasons = coefficients["Season"].drop_duplicates().astype(str).tolist()
    return coefficients, selected, seasons


def _plot_saved_figure6_coefficients(
    axis: plt.Axes, coefficients: pd.DataFrame, seasons: list[str]
) -> None:
    specs = {
        "FOUL->FTA": ("#CC3366", "o", "-"),
        "FTA->FTM": ("#0072B2", "s", "--"),
        "FOUL->PERS": ("#009E73", "^", "-."),
        "FOUL->LOOSE": ("#E69F00", "D", ":"),
    }
    x = np.arange(len(seasons))
    for label, (color, marker, linestyle) in specs.items():
        frame = coefficients[coefficients["series"] == label].set_index("Season")
        values = frame.reindex(seasons)["value"].to_numpy(float)
        axis.plot(
            x, values, color=color, marker=marker, linestyle=linestyle,
            linewidth=1.9, markersize=4.2, label=label,
        )
    axis.set_ylim(0, 1.38)
    axis.set_ylabel("Fitted coefficient")
    axis.set_xticks(x, seasons, rotation=35, ha="right")
    configure_axis(axis)
    axis.legend(
        frameon=False, fontsize=7.2, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, 1.03), borderaxespad=0,
    )


def _plot_saved_figure6_families(
    axis: plt.Axes, selected: pd.DataFrame, seasons: list[str]
) -> None:
    nodes = ("FOUL", "FTA", "FTM", "PERS", "LOOSE")
    colors = {
        "Poisson": "#4C78A8", "NB": "#F58518", "ZIP": "#54A24B",
        "Geom": "#E45756", "Binomial": "#B279A2", "Bernoulli": "#72B7B2",
    }
    codes = {family: index for index, family in enumerate(FAMILIES)}
    frame = selected.pivot(index="Season", columns="node", values="selected_working_family")
    frame = frame.reindex(index=seasons, columns=nodes)
    matrix = np.array([[codes[str(value)] for value in row] for row in frame.to_numpy()])
    axis.imshow(
        matrix, cmap=ListedColormap([colors[f] for f in FAMILIES]),
        vmin=-0.5, vmax=len(FAMILIES) - 0.5, aspect="auto",
    )
    for i, season in enumerate(seasons):
        for j, node in enumerate(nodes):
            family = str(frame.loc[season, node])
            axis.text(
                j, i, family, ha="center", va="center", fontsize=6.5,
                color="white" if family in {"Poisson", "Geom", "Binomial"} else "#222222",
            )
    axis.set_xticks(range(len(nodes)), nodes)
    axis.set_yticks(range(len(seasons)), seasons)
    axis.set_xlabel("Selected exogenous family")
    handles = [Patch(facecolor=colors[f], edgecolor="none", label=f) for f in FAMILIES]
    axis.legend(
        handles=handles, frameon=False, fontsize=6.5, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, -0.20), borderaxespad=0,
    )


def redraw_figure6_from_saved_plotdata() -> list[Path]:
    """Export title-free Figure 6 panels without loading NBA fit inputs."""
    coefficients, selected, seasons = _saved_figure6_frames()
    outputs: list[Path] = []

    fig, axis = plt.subplots(figsize=(4.6, 3.75))
    _plot_saved_figure6_coefficients(axis, coefficients, seasons)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.21, top=0.77)
    path = NBA_DIAGNOSTIC_FIG / "fig6a_nba_coefficients_panel.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, axis = plt.subplots(figsize=(5.0, 3.75))
    _plot_saved_figure6_families(axis, selected, seasons)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.25)
    path = NBA_DIAGNOSTIC_FIG / "fig6b_selected_exogenous_family_panel.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(
        1, 2, figsize=(9.6, 3.75), gridspec_kw={"width_ratios": [1.15, 1.30]}
    )
    _plot_saved_figure6_coefficients(axes[0], coefficients, seasons)
    _plot_saved_figure6_families(axes[1], selected, seasons)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.77, bottom=0.25, wspace=0.34)
    path = NBA_DIAGNOSTIC_FIG / "fig6_nba_diagnostics_final.pdf"
    fig.savefig(path, bbox_inches="tight")
    outputs.append(path)
    preview = NBA_DIAGNOSTIC_FIG / "fig6_nba_diagnostics_preview.png"
    fig.savefig(preview, dpi=400, bbox_inches="tight")
    outputs.append(preview)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redraw-paper-figures-from-saved",
        action="store_true",
        help="Redraw Figures 1, 2, 3, 4, and 6 from saved plotdata CSV files only.",
    )
    args = parser.parse_args()
    setup_style()
    if args.redraw_paper_figures_from_saved:
        outputs = [redraw_figure1_from_saved_plotdata()]
        outputs.append(redraw_figure2_from_saved_plotdata())
        outputs.extend(redraw_figure3_from_saved_plotdata())
        outputs.append(redraw_figure4_from_saved_plotdata())
        outputs.extend(redraw_figure6_from_saved_plotdata())
        print("Redrew paper figures from saved plotting data only:")
        for path in outputs:
            print(path)
        return
    if OUT.exists():
        shutil.rmtree(OUT)
    for directory in (
        MAIN_FIG,
        APP_FIG,
        APP_TABLE,
        MAIN_TABLE,
        NBA_FIG,
        NBA_DIAGNOSTIC_FIG,
        NBA_TABLE,
        META,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    raw = load_main_raw()
    make_table1()
    make_figure1(raw)
    make_figure2(raw)
    make_figure3(raw)
    make_figure4()
    make_appendix_diagnostic_tables(raw)
    make_runtime_component_figure()
    make_nba_graph()
    make_nba_diagnostics_figure()
    make_nba_bootstrap_figure()
    make_nba_tables()
    write_metadata_documents()
    write_manifest()
    print(f"Created {OUT}")


if __name__ == "__main__":
    main()
