#!/usr/bin/env python3
"""Reproduce the paper's real-data results from the committed NBA inputs.

Deliverables produced by this driver, and nothing else:

* the recovered graph structure for each season,
* the proposed-method row of manuscript Table 3,
* manuscript Figures 5 and 6.

Everything the historical scripts write on the way there (per-family local BIC
tables, optimizer diagnostics, per-start traces, run manifests) is treated as
intermediate and is left under ``<work-dir>/intermediate`` rather than being
presented as a result.

The historical experiment scripts under ``experiments/nba/scripts`` are used
unmodified.  This driver only supplies the directory layout they expect,
because the repository stores the processed inputs under
``data/nba/processed_team_quarter`` while the historical scripts read the
original ``outputs/expanded_candidate_study`` layout.

The four baseline methods in Table 3 are not recomputed here: their sources are
not redistributable (see ``docs/THIRD_PARTY.md``).  Only the proposed-method row
is regenerated and compared.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import check_environment


ROOT = Path(__file__).resolve().parents[1]
NBA_SCRIPTS = ROOT / "experiments/nba/scripts"
COMMITTED_INPUTS = ROOT / "data/nba/processed_team_quarter"
COMMITTED_RESULTS = ROOT / "results/nba"
CORE = ROOT / "experiments/nba/core/d.py"

HYPOTHESIS = "foul_leaves_5_team"
SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2025))
#: Layout the historical estimator expects, relative to its ``--workspace``.
STAGED_INPUT_SUBDIR = Path("outputs/expanded_candidate_study/data_team_loose")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stage_inputs(work: Path, seasons: tuple[str, ...]) -> Path:
    """Copy the committed processed CSVs into the historical input layout."""
    staged = work / "intermediate" / STAGED_INPUT_SUBDIR
    staged.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        source = COMMITTED_INPUTS / f"expanded_team_quarter_{season}.csv"
        if not source.exists():
            raise SystemExit(f"Missing committed input: {source}")
        shutil.copy2(source, staged / source.name)
    return work / "intermediate"


def run_estimator(workspace: Path, fits: Path, seasons: tuple[str, ...],
                  workers: int) -> None:
    command = [
        sys.executable,
        str(NBA_SCRIPTS / "run_expanded_candidate_optimization.py"),
        HYPOTHESIS,
        "--workspace", str(workspace),
        "--core", str(CORE),
        "--outputs-dir", str(fits),
        "--starts", "2",
        "--maxiter", "250",
        "--workers", str(workers),
        "--seasons", *seasons,
    ]
    print("[run] " + " ".join(command[1:]), flush=True)
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(f"Estimator failed with exit code {completed.returncode}")


def collect(fits: Path, seasons: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive structure, coefficients and families from the produced fits.

    Metric definitions and node-name abbreviations are imported from the
    historical summarizer so this driver cannot drift from it.
    """
    sys.path.insert(0, str(NBA_SCRIPTS))
    summarizer = load_module("ptsem_nba_summarizer", NBA_SCRIPTS / "summarize_nba_extension.py")
    optimizer = load_module(
        "ptsem_nba_optimizer", NBA_SCRIPTS / "run_expanded_candidate_optimization.py"
    )

    recovery, coefficients, families = [], [], []
    for season in seasons:
        path = fits / HYPOTHESIS / season / "optimization/graph_result.json"
        if not path.exists():
            raise SystemExit(f"Estimator produced no result for {season}: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        predicted = optimizer.parse_edges(result["edges"])
        recovery.append({
            "season": season,
            "estimated_graph": result["edges"],
            **summarizer.graph_metrics(predicted),
        })
        for fit in result["node_fits"]:
            node = summarizer.SHORT.get(fit["node"], fit["node"])
            families.append({"season": season, "node": node, "family": fit["family"]})
            for parent, alpha in fit["alpha"].items():
                coefficients.append({
                    "season": season,
                    "source": summarizer.SHORT.get(parent, parent),
                    "target": node,
                    "coefficient": alpha,
                    "selected_family": fit["family"],
                })
    return pd.DataFrame(recovery), pd.DataFrame(coefficients), pd.DataFrame(families)


def proposed_table3_row(recovery: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "skeleton_precision", "skeleton_recall", "skeleton_f1",
        "directed_precision", "directed_recall", "directed_f1",
    )
    return pd.DataFrame([{
        "method": "PT-SEM (LibraryDP)",
        "n_seasons": len(recovery),
        **{f"mean_{column}": float(recovery[column].mean()) for column in columns},
    }])


def draw_figures(results_dir: Path, destination: Path) -> None:
    module = load_module("ptsem_nba_figures", NBA_SCRIPTS / "make_nba_final_figures.py")
    module.RESULTS = results_dir
    module.FINAL = destination

    def save_once(fig, stem: str) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{stem}.pdf", format="pdf")
        fig.savefig(destination / f"{stem}.png", dpi=300, format="png")

    module.save_both = save_once
    module.make_figure5()
    module.make_figure6()


def compare_with_committed(recovery: pd.DataFrame, row: pd.DataFrame,
                           seasons: tuple[str, ...]) -> bool:
    """Report whether the rerun matches the committed paper results."""
    print("\n" + "=" * 78)
    print("Comparison with the committed paper results")
    print("=" * 78)

    ok = True
    committed_by_season = pd.read_csv(COMMITTED_RESULTS / "nba_main_graph_recovery_by_season.csv")
    committed_by_season = committed_by_season.set_index("season")
    for entry in recovery.to_dict("records"):
        season = entry["season"]
        if season not in committed_by_season.index:
            print(f"  {season}: not present in the committed table")
            ok = False
            continue
        expected = committed_by_season.loc[season, "estimated_graph"]
        match = expected == entry["estimated_graph"]
        ok &= match
        print(f"  {season}  structure {'MATCH ' if match else 'DIFFER'}  {entry['estimated_graph']}")

    # The committed Table 3 row averages all ten seasons, so a partial rerun
    # has nothing comparable to offer.
    if seasons != SEASONS:
        print(f"\n  [skip] Table 3 averages all ten seasons; this run had {len(seasons)}.")
        return bool(ok)

    committed_table = pd.read_csv(COMMITTED_RESULTS / "tables/table2_nba_graph_recovery.csv")
    proposed = committed_table[committed_table["method"] == "PT-SEM (LibraryDP)"]
    if proposed.empty:
        print("\n  committed Table 3 has no proposed-method row to compare")
        return False
    print("\n  Table 3, proposed-method row:")
    for column in row.columns:
        if not column.startswith("mean_"):
            continue
        produced = float(row.iloc[0][column])
        expected = float(proposed.iloc[0][column])
        delta = abs(produced - expected)
        flag = "MATCH " if delta <= 1e-12 else "DIFFER"
        ok &= delta <= 1e-12
        print(f"    {column:<28} produced={produced:.10f} committed={expected:.10f} "
              f"delta={delta:.3e} {flag}")
    return bool(ok)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "scratch/nba_reproduction")
    parser.add_argument("--seasons", nargs="+", choices=SEASONS, default=list(SEASONS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="run in an unpinned interpreter; results are then not expected to match",
    )
    args = parser.parse_args()

    if args.skip_env_check:
        print("WARNING: environment check skipped; fitted results may not match the paper.")
    else:
        check_environment.require("nba")

    seasons = tuple(args.seasons)
    work = args.work_dir.resolve()
    deliverables = work / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)

    workspace = stage_inputs(work, seasons)
    fits = work / "intermediate" / "fits"
    run_estimator(workspace, fits, seasons, args.workers)

    recovery, coefficients, families = collect(fits, seasons)
    row = proposed_table3_row(recovery)

    recovery.to_csv(deliverables / "nba_graph_recovery_by_season.csv", index=False)
    row.to_csv(deliverables / "table3_proposed_method_row.csv", index=False)

    # Figure 6 validates a complete ten-season grid, so it is only drawn for a
    # full run.  The figure inputs live with the intermediates.
    figure_inputs = work / "intermediate" / "figure_inputs"
    figure_inputs.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(figure_inputs / "nba_coefficients_by_season.csv", index=False)
    families.to_csv(figure_inputs / "nba_working_families_by_season.csv", index=False)
    if seasons == SEASONS:
        draw_figures(figure_inputs, deliverables)
    else:
        print(f"\n[skip] Figures 5-6 need all ten seasons; this run had {len(seasons)}.")

    matched = compare_with_committed(recovery, row, seasons)

    scope = "all ten seasons" if seasons == SEASONS else f"{len(seasons)} of ten seasons"
    print("\n" + "=" * 78)
    print(f"Deliverables: {deliverables}")
    print(f"Intermediates: {work / 'intermediate'}")
    print(f"RESULT ({scope}):",
          "matches the committed paper results"
          if matched else "DOES NOT match the committed paper results")
    print("=" * 78)
    raise SystemExit(0 if matched else 1)


if __name__ == "__main__":
    main()
