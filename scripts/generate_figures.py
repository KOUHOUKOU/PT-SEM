#!/usr/bin/env python3
"""Regenerate paper figures from the committed frozen results."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "figures" / "generated"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulation_figures() -> None:
    module = load_module(
        "ptsem_legacy_figures",
        ROOT / "experiments/simulation/legacy/build_final_paper_figures.py",
    )
    main = GENERATED / "main_figures"
    main.mkdir(parents=True, exist_ok=True)
    module.OUT = GENERATED
    module.MAIN_FIG = main
    module.META = GENERATED / "metadata"
    module.META.mkdir(parents=True, exist_ok=True)
    module.setup_style()
    frames = []
    sources = (
        ("d-sweep", ROOT / "data/simulation/raw/experiment1_d_sweep_raw.csv", "d"),
        ("N-sweep", ROOT / "data/simulation/raw/experiment2_N_sweep_raw.csv", "N"),
        ("density-sweep", ROOT / "data/simulation/raw/experiment3_kin_sweep_raw.csv", "kbar_in"),
    )
    for sweep, path, x_column in sources:
        frame = pd.read_csv(path)
        frame["sweep_type"] = sweep
        frame["x_value"] = frame[x_column]
        frame["method"] = frame["method"].map(module.METHOD_MAP)
        frame["replication"] = frame["rep"]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    module.make_figure1(raw)
    module.make_figure2(raw)
    paper_figure3(raw, module, main)


def paper_figure3(raw, module, destination: Path) -> None:
    """Draw the current-manuscript Figure 3 labels from the frozen raw data."""
    n_raw = raw[
        (raw["sweep_type"] == "N-sweep")
        & (raw["method"].isin(["LibraryDP", "LibraryGreedy"]))
    ]
    accuracy = module.aggregate_metric(
        n_raw,
        "family_accuracy",
        "working_family_accuracy",
        methods=("LibraryDP", "LibraryGreedy"),
        valid_domain=(0, 1),
    )
    common = module.family_confusion(raw, "common")
    expanded = module.family_confusion(raw, "expanded")
    fig, axes = module.plt.subplots(
        1, 3, figsize=(11.8, 3.6), gridspec_kw={"width_ratios": [1.25, 1, 1]}
    )
    axis = axes[0]
    for method in ("LibraryDP", "LibraryGreedy"):
        for regime, linestyle in (("common", "-"), ("expanded", "--")):
            panel = accuracy[
                accuracy["method"].eq(method) & accuracy["regime"].eq(regime)
            ].sort_values("x_value")
            style = module.STYLE[method]
            x = panel["x_value"].to_numpy(float)
            axis.plot(
                x,
                panel["mean"],
                color=style["color"], marker=style["marker"], linestyle=linestyle,
                linewidth=2, markersize=4.2, label=f"{method} — {regime}",
            )
            axis.fill_between(
                x, panel["plot_ci95_low"], panel["plot_ci95_high"],
                color=style["color"], alpha=0.10, linewidth=0,
            )
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Sample size $N$")
    axis.set_ylabel("Working-family accuracy")
    axis.set_title("(a) Working-family accuracy")
    module.configure_axis(axis, "N-sweep")
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", borderaxespad=0.7)

    image = None
    for axis, frame, title in (
        (axes[1], common, "(b) Common"),
        (axes[2], expanded, "(c) Expanded"),
    ):
        matrix = (
            frame.pivot(index="true_family", columns="selected_working_family", values="row_proportion")
            .reindex(index=module.FAMILIES, columns=module.FAMILIES)
            .to_numpy(float)
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(6):
            for j in range(6):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if matrix[i, j] >= 0.5 else "#222222")
        axis.set_xticks(range(6), module.FAMILIES, rotation=45, ha="right")
        axis.set_yticks(range(6), module.FAMILIES)
        axis.set_xlabel("Selected working family")
        axis.set_ylabel("True family")
        axis.set_title(title)
    fig.subplots_adjust(left=0.050, right=0.90, top=0.90, bottom=0.20, wspace=0.40)
    color_axis = fig.add_axes([0.925, 0.22, 0.014, 0.68])
    fig.colorbar(image, cax=color_axis, label="Row proportion")
    fig.savefig(destination / "fig3_working_family_diagnostics_final.pdf", bbox_inches="tight")
    fig.savefig(destination / "fig3_working_family_diagnostics_final.png", dpi=400, bbox_inches="tight")
    module.plt.close(fig)
    accuracy.to_csv(destination / "fig3_working_family_diagnostics_final_plotdata.csv", index=False)
    common.to_csv(destination / "fig3_confusion_common_plotdata.csv", index=False)
    expanded.to_csv(destination / "fig3_confusion_expanded_plotdata.csv", index=False)


def all_poisson_figure() -> None:
    module = load_module(
        "ptsem_all_poisson_figure",
        ROOT / "experiments/simulation/all_poisson_snapshot/scripts/plot_all_poisson_final.py",
    )
    module.SNAPSHOT = ROOT
    module.SUMMARY = ROOT / "data/all_poisson/summaries/all_poisson_summary.csv"
    module.VALIDATION = ROOT / "data/all_poisson/metadata/validation_report.json"
    module.OUT = GENERATED
    module.main()


def nba_figures() -> None:
    module = load_module(
        "ptsem_nba_figures",
        ROOT / "experiments/nba/scripts/make_nba_final_figures.py",
    )
    module.RESULTS = ROOT / "results/nba"
    module.FINAL = GENERATED

    def save_once(fig, stem: str) -> None:
        GENERATED.mkdir(parents=True, exist_ok=True)
        fig.savefig(GENERATED / f"{stem}.pdf", format="pdf")
        fig.savefig(GENERATED / f"{stem}.png", dpi=300, format="png")

    module.save_both = save_once
    module.make_figure5()
    module.make_figure6()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("simulation", "all-poisson", "nba"),
        help="Generate one result family; default generates all six paper figures.",
    )
    args = parser.parse_args()
    GENERATED.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "simulation"):
        simulation_figures()
    if args.only in (None, "all-poisson"):
        all_poisson_figure()
    if args.only in (None, "nba"):
        nba_figures()
    print(f"Generated figures in {GENERATED}")


if __name__ == "__main__":
    main()
