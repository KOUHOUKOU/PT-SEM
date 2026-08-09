"""Shared plotting and output helpers for PT-SEM sensitivity sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import d


PLOT_METRICS: Tuple[Tuple[str, str, str, str, bool], ...] = (
    ("mean_directed_f1", "se_directed_f1", "directed F1", "directed_f1", False),
    ("mean_skeleton_f1", "se_skeleton_f1", "skeleton F1", "skeleton_f1", False),
    ("mean_exact_dag", "se_exact_dag", "exact DAG recovery rate", "exact_dag", False),
    (
        "mean_family_accuracy",
        "se_family_accuracy",
        "node-wise family recovery accuracy",
        "family_accuracy",
        False,
    ),
    (
        "mean_alpha_rmse_if_exact",
        "se_alpha_rmse_if_exact",
        "alpha RMSE given exact DAG",
        "alpha_rmse_if_exact",
        False,
    ),
    (
        "mean_alpha_rmse_on_correct_edges",
        "se_alpha_rmse_on_correct_edges",
        "alpha RMSE on correctly directed edges",
        "alpha_rmse_on_correct_edges",
        False,
    ),
    (
        "mean_alpha_mape_on_correct_edges_pct",
        "se_alpha_mape_on_correct_edges_pct",
        "alpha MAPE on correctly directed edges (%)",
        "alpha_mape_on_correct_edges",
        False,
    ),
    (
        "mean_runtime_sec",
        "se_runtime_sec",
        "total runtime (seconds, log scale)",
        "runtime",
        True,
    ),
    (
        "mean_local_score_runtime_sec",
        "se_local_score_runtime_sec",
        "local-score runtime (seconds, log scale)",
        "local_score_runtime",
        True,
    ),
    (
        "mean_search_runtime_sec",
        "se_search_runtime_sec",
        "graph-search runtime (seconds, log scale)",
        "search_runtime",
        True,
    ),
)


def plot_sweep_summary(
    summary: pd.DataFrame,
    x_column: str,
    xlabel: str,
    suffix: str,
    outdir: Path,
    regimes: Sequence[str],
    log_x: bool = False,
    xticks: Optional[Sequence[float]] = None,
) -> None:
    """Create one scientifically interpretable plot per regime and metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for regime in regimes:
        regime_dir = outdir / "figures" / regime
        regime_dir.mkdir(parents=True, exist_ok=True)
        regime_frame = summary.loc[summary["alpha_regime"] == regime]
        for mean_column, se_column, ylabel, stem, log_y in PLOT_METRICS:
            figure, axis = plt.subplots(figsize=(7.0, 4.5))
            plotted = False
            for method in d.MAIN_PLOT_METHODS:
                subset = regime_frame.loc[
                    regime_frame["method"] == method
                ].sort_values(x_column)
                subset = subset.loc[subset[mean_column].notna()]
                if subset.empty:
                    continue
                x_values = subset[x_column].to_numpy(dtype=float)
                means = subset[mean_column].to_numpy(dtype=float)
                errors = subset[se_column].fillna(0.0).to_numpy(dtype=float)
                if log_y:
                    positive = means > 0.0
                    x_values = x_values[positive]
                    means = means[positive]
                    errors = errors[positive]
                    errors = np.minimum(errors, 0.95 * means)
                if not len(means):
                    continue
                axis.errorbar(
                    x_values,
                    means,
                    yerr=errors,
                    marker="o",
                    linewidth=1.7,
                    capsize=3,
                    label=method,
                )
                plotted = True
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            if log_x:
                axis.set_xscale("log")
            if log_y:
                axis.set_yscale("log")
            if xticks is not None:
                axis.set_xticks(list(xticks))
                if log_x:
                    axis.get_xaxis().set_major_formatter(
                        matplotlib.ticker.ScalarFormatter()
                    )
            axis.grid(alpha=0.2)
            if plotted:
                axis.legend(fontsize=8)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No applicable results",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            figure.tight_layout()
            figure.savefig(
                regime_dir / f"fig_{stem}_vs_{suffix}.png",
                dpi=300,
            )
            plt.close(figure)


def plot_realized_indegree(
    summary: pd.DataFrame,
    x_column: str,
    xlabel: str,
    outdir: Path,
    regimes: Sequence[str],
) -> None:
    """Plot nominal versus realized density once per alpha regime."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for regime in regimes:
        regime_dir = outdir / "figures" / regime
        regime_dir.mkdir(parents=True, exist_ok=True)
        subset = summary.loc[
            (summary["alpha_regime"] == regime)
            & (summary["method"] == "LibraryDP")
        ].sort_values(x_column)
        if subset.empty:
            continue
        x_values = subset[x_column].to_numpy(dtype=float)
        means = subset["mean_realized_avg_indegree"].to_numpy(dtype=float)
        errors = subset["se_realized_avg_indegree"].fillna(0.0).to_numpy(float)
        figure, axis = plt.subplots(figsize=(6.5, 4.3))
        axis.errorbar(x_values, means, yerr=errors, marker="o", capsize=3)
        axis.plot(x_values, x_values, linestyle="--", color="0.45", label="target")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("realized average indegree")
        axis.set_xticks(x_values)
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(regime_dir / "fig_realized_indegree_vs_kbar.png", dpi=300)
        plt.close(figure)
