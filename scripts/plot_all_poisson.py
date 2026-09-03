#!/usr/bin/env python3
"""Draw Figure 1 from complete, validated all-Poisson summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY: Path | None = None
VALIDATION: Path | None = None
CONFIG = PACKAGE_ROOT / "config/all_poisson.json"
OUT: Path | None = None
PLOTDATA: Path | None = None
STEM = "figure1_all_poisson"

METHOD_ORDER = (
    "Proposed DP-BIC",
    "Greedy-BIC",
    "Poisson-only DP",
    "ODS",
    "PC-RCIT",
    "PB-SCM",
    "PB-SCM-PGF",
)
STYLE = {
    "Proposed DP-BIC": dict(color="#CC3366", marker="o", linestyle="-"),
    "Greedy-BIC": dict(color="#E69F00", marker="s", linestyle="-"),
    "Poisson-only DP": dict(color="#0072B2", marker="o", linestyle="-"),
    "ODS": dict(color="#CC79A7", marker="d", linestyle="-."),
    "PC-RCIT": dict(color="#009E73", marker="^", linestyle="--"),
    "PB-SCM": dict(color="#D55E00", marker="v", linestyle=":"),
    "PB-SCM-PGF": dict(color="#56B4E9", marker="P", linestyle="-."),
}
SWEEPS = ("dimension", "sample_size", "average_in_degree")
TITLES = {
    "dimension": "Dimension sweep",
    "sample_size": "Sample-size sweep",
    "average_in_degree": "Average-in-degree sweep",
}
XLABELS = {
    "dimension": "Number of nodes $d$",
    "sample_size": "Sample size $N$",
    "average_in_degree": "Average in-degree",
}


def configure_run_root(run_root: Path) -> None:
    global SUMMARY, VALIDATION, OUT, PLOTDATA
    resolved = run_root.resolve()
    SUMMARY = resolved / "all_poisson/summaries/all_poisson_summary.csv"
    VALIDATION = resolved / "all_poisson/metadata/validation_report.json"
    OUT = resolved / "paper_objects"
    PLOTDATA = resolved / "all_poisson/plotdata"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 12,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def configure_axis(axis: plt.Axes, sweep: str, values: np.ndarray) -> None:
    axis.grid(True, color="#D9D9D9", linestyle="--", linewidth=0.45, alpha=0.30)
    axis.set_axisbelow(True)
    axis.set_ylim(0.0, 1.02)
    if sweep == "sample_size":
        axis.set_xscale("log")
        axis.set_xticks((100, 1000, 10000))
        axis.set_xticklabels((r"$10^2$", r"$10^3$", r"$10^4$"))
        axis.grid(
            True,
            which="minor",
            color="#E6E6E6",
            linestyle="--",
            linewidth=0.35,
            alpha=0.15,
        )
    elif sweep == "dimension":
        axis.set_xticks(sorted(set(values.astype(int))))
    else:
        axis.set_xticks(sorted(set(values.astype(float))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    configure_run_root(args.run_root)
    assert SUMMARY is not None and VALIDATION is not None and OUT is not None and PLOTDATA is not None
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError(
            "Refusing to plot: All-Poisson validation_report.json is not passed"
        )
    summary = pd.read_csv(SUMMARY)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    required_R = int(config["replications"])
    if summary["R_success"].ne(required_R).any() or summary["mean_F1"].isna().any():
        raise RuntimeError("Refusing to plot incomplete summary cells")
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 5.85), sharey=False)
    handles: dict[str, object] = {}
    for row, regime in enumerate(("restricted", "extended")):
        allowed = METHOD_ORDER if regime == "restricted" else METHOD_ORDER[:5]
        for column, sweep in enumerate(SWEEPS):
            axis = axes[row, column]
            panel = summary[
                summary["regime"].eq(regime) & summary["sweep"].eq(sweep)
            ].copy()
            for method in allowed:
                values = panel[panel["method"].eq(method)].sort_values("sweep_value")
                if values.empty:
                    raise RuntimeError(f"Missing curve: {regime}/{sweep}/{method}")
                style = STYLE[method]
                x = values["sweep_value"].to_numpy(float)
                line = axis.plot(
                    x,
                    values["mean_F1"].to_numpy(float),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    markersize=4.2,
                    label=method,
                    zorder=4 if method in ("Proposed DP-BIC", "Poisson-only DP") else 3,
                )[0]
                axis.fill_between(
                    x,
                    values["plot_ci95_low"].to_numpy(float),
                    values["plot_ci95_high"].to_numpy(float),
                    color=style["color"],
                    alpha=0.11,
                    linewidth=0,
                    zorder=1,
                )
                handles[method] = line
            if row == 0:
                axis.set_title(f"({chr(ord('a') + column)}) {TITLES[sweep]}")
            else:
                axis.set_xlabel(XLABELS[sweep])
            if column == 0:
                axis.set_ylabel(
                    ("Restricted" if regime == "restricted" else "Extended")
                    + "\nDirected F1"
                )
            configure_axis(axis, sweep, panel["sweep_value"].to_numpy(float))
    ordered = [method for method in METHOD_ORDER if method in handles]
    fig.legend(
        [handles[method] for method in ordered],
        ordered,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.175,
        wspace=0.23,
        hspace=0.20,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTDATA.mkdir(parents=True, exist_ok=True)
    plotdata_path = PLOTDATA / f"{STEM}_plotdata.csv"
    summary.to_csv(plotdata_path, index=False)
    pdf = OUT / f"{STEM}.pdf"
    png = OUT / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    metadata = {
        "figure": 4,
        "layout": "2x3",
        "rows": ["Restricted", "Extended"],
        "columns": list(SWEEPS),
        "methods_restricted": list(METHOD_ORDER),
        "methods_extended": list(METHOD_ORDER[:5]),
        "input_summary": str(SUMMARY.relative_to(PACKAGE_ROOT)).replace("\\", "/"),
        "numeric_plotdata": str(plotdata_path.relative_to(PACKAGE_ROOT)).replace("\\", "/"),
        "validation_report": str(VALIDATION.relative_to(PACKAGE_ROOT)).replace("\\", "/"),
        "confidence_interval": "mean +/- 1.96*SE; sample SD ddof=1",
        "output_pdf": str(pdf.relative_to(PACKAGE_ROOT)).replace("\\", "/"),
    }
    (OUT / f"{STEM}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
