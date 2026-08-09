# -*- coding: utf-8 -*-
"""Paper-final six-family PT-SEM sensitivity suite.

One command runs three orthogonal sweeps under two coefficient regimes:

* d-sweep: d=4,...,10 at N=3200 and exact average indegree 1.5;
* N-sweep: N=100,...,10000 at d=8 and exact average indegree 1.5;
* density-sweep: average indegree=1.0,...,3.0 at d=8 and N=3200.

Every node independently draws its true exogenous family uniformly from
Poisson, NB, ZIP, Geom, Binomial, and Bernoulli.  The candidate working
library contains the same six families.  Common and expanded alpha regimes
are paired by seed: DAG edges, node families, and exogenous parameters agree;
only the coefficient range and resulting observations change.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# These variables must be set before importing d.py, including in Windows
# spawn workers.  The values are intentionally locked for the paper suite.
SIX_FAMILIES: Tuple[str, ...] = (
    "Poisson",
    "NB",
    "ZIP",
    "Geom",
    "Binomial",
    "Bernoulli",
)
os.environ["PTSEM_FAMILY_LIBRARY"] = ",".join(SIX_FAMILIES)
os.environ["PTSEM_FAMILY_ASSIGNMENT"] = "iid_uniform"
os.environ["PTSEM_BALANCED_FAMILY_SHUFFLE"] = "0"
os.environ.setdefault("PTSEM_BINOMIAL_N_MIN", "2")
os.environ.setdefault("PTSEM_BINOMIAL_N_MAX", "20")
os.environ.setdefault("PTSEM_BINOMIAL_P_MIN", "0.15")
os.environ.setdefault("PTSEM_BINOMIAL_P_MAX", "0.85")
os.environ.setdefault("PTSEM_BERNOULLI_P_MIN", "0.15")
os.environ.setdefault("PTSEM_BERNOULLI_P_MAX", "0.85")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import d
import setup_external_baselines
from ptsem_experiment1_alpha_regimes import (
    atomic_write_csv,
    atomic_write_json,
    prevent_windows_sleep,
    utc_now,
)
from ptsem_sweep_utils import plot_realized_indegree, plot_sweep_summary


SUITE_VERSION = "six_family_final_v1"
D_VALUES: Tuple[int, ...] = tuple(range(4, 11))
N_VALUES: Tuple[int, ...] = (
    100,
    200,
    400,
    800,
    1600,
    2400,
    3200,
    6400,
    10000,
)
KBAR_VALUES: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
ANCHOR_D = 8
ANCHOR_N = 3200
ANCHOR_KBAR = 1.5
MAX_PARENTS = 5

COMMON_METHODS: Tuple[str, ...] = (
    "LibraryDP",
    "LibraryGreedy",
    "OracleDP",
    "PC-RCIT",
    "PBSCM",
    "PBSCM_PGF",
    "PoissonDAG-ODS",
)
EXPANDED_METHODS: Tuple[str, ...] = (
    "LibraryDP",
    "LibraryGreedy",
    "OracleDP",
    "PC-RCIT",
    "PoissonDAG-ODS",
)
REGIMES: Mapping[str, Mapping[str, object]] = {
    "common": {
        "alpha_low": 0.15,
        "alpha_high": 0.85,
        "methods": COMMON_METHODS,
        "description": "standard alpha<1 comparison including both PB-SCM baselines",
    },
    "expanded": {
        "alpha_low": 0.2,
        "alpha_high": 2.0,
        "methods": EXPANDED_METHODS,
        "description": "alpha>1 extension; PB-SCM excluded outside its model domain",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-final six-family PT-SEM d/N/density suite"
    )
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument(
        "--results-root",
        default=str(HERE / "ptsem_six_family_final_results"),
    )
    parser.add_argument(
        "--regime",
        choices=("both", "common", "expanded"),
        default="both",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=("d", "N", "kbar"),
        default=("d", "N", "kbar"),
        help="Run all three by default; this option is mainly for diagnostics.",
    )
    parser.add_argument("--d-values", nargs="+", type=int, default=D_VALUES)
    parser.add_argument("--N-values", nargs="+", type=int, default=N_VALUES)
    parser.add_argument("--kbar-values", nargs="+", type=float, default=KBAR_VALUES)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="R=1 with endpoint values; writes to the selected results root.",
    )
    args = parser.parse_args(argv)
    if args.R < 1 or args.workers < 1 or args.seed < 0:
        parser.error("R/workers must be positive and seed nonnegative")
    args.sweeps = tuple(dict.fromkeys(args.sweeps))
    args.d_values = tuple(dict.fromkeys(args.d_values))
    args.N_values = tuple(dict.fromkeys(args.N_values))
    args.kbar_values = tuple(dict.fromkeys(args.kbar_values))
    if any(value < 2 for value in args.d_values):
        parser.error("all d values must be at least 2")
    if any(value < 1 for value in args.N_values):
        parser.error("all N values must be positive")
    if any(value < 0.0 for value in args.kbar_values):
        parser.error("all average-indegree values must be nonnegative")
    capacity = sum(min(MAX_PARENTS, child) for child in range(1, ANCHOR_D))
    infeasible = [
        value
        for value in args.kbar_values
        if round(ANCHOR_D * value) > capacity
    ]
    if infeasible:
        parser.error(
            f"kbar values {infeasible} exceed exact-edge capacity "
            f"{capacity / ANCHOR_D:.3f}"
        )
    if args.smoke:
        args.R = 1
        args.workers = min(args.workers, 2)
        args.d_values = (4, 10)
        args.N_values = (100, 300, 800)
        args.kbar_values = (1.0, 3.0)
        args.anchor_N = 300
    else:
        args.anchor_N = ANCHOR_N
    return args


def selected_regimes(name: str) -> Tuple[str, ...]:
    return ("common", "expanded") if name == "both" else (name,)


def dependency_versions() -> Dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pandas", "scipy", "statsmodels", "causal-learn")
    }


def run_preflight(seed: int) -> Dict[str, object]:
    """Fail early on mathematical, assignment, and large-count regressions."""
    if tuple(d.FAMLIB) != SIX_FAMILIES:
        raise RuntimeError(f"six-family activation failed: {d.FAMLIB}")
    if d.FAMILY_ASSIGNMENT_MODE != "iid_uniform":
        raise RuntimeError(
            "paper suite requires iid_uniform true-family assignment; got "
            f"{d.FAMILY_ASSIGNMENT_MODE}"
        )
    from six_family_math_validation import run_checks

    validation = run_checks()
    assignment_rng = np.random.default_rng(seed + 77001)
    labels: List[str] = []
    for _ in range(5000):
        labels.extend(spec.family for spec in d.make_exogenous_specs(8, assignment_rng))
    counts = {family: labels.count(family) for family in SIX_FAMILIES}
    proportions = {family: counts[family] / len(labels) for family in SIX_FAMILIES}
    if max(abs(value - 1.0 / 6.0) for value in proportions.values()) > 0.01:
        raise RuntimeError(f"iid family assignment audit failed: {proportions}")

    stress: Dict[str, Dict[str, float]] = {}
    for label, node_count, kbar in (
        ("expanded_d10", 10, ANCHOR_KBAR),
        ("expanded_kbar3", ANCHOR_D, 3.0),
    ):
        X, A, specs = d.simulate_ptsem(
            node_count,
            400,
            d.derive_seed(seed, node_count, 0),
            kbar,
            MAX_PARENTS,
            0.2,
            2.0,
            "exact_edges",
        )
        if not np.issubdtype(X.dtype, np.integer) or np.any(X < 0):
            raise RuntimeError(f"invalid expanded-alpha counts in {label}")
        stress[label] = {
            "max_count": int(X.max()),
            "mean_count": float(X.mean()),
            "p999_count": float(np.quantile(X, 0.999)),
            "edge_count": int(np.count_nonzero(A)),
            "distinct_true_families": len({spec.family for spec in specs}),
        }
    return {
        "status": "passed",
        "mathematical_validation": validation,
        "iid_assignment_proportions_40000_draws": proportions,
        "expanded_alpha_stress": stress,
    }


def cell_tag(d_value: int, n_value: int, kbar: float) -> str:
    k_text = f"{kbar:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"d_{d_value:02d}_N_{n_value:05d}_kin_{k_text}"


def cell_configuration(
    regime: str,
    d_value: int,
    n_value: int,
    kbar: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    config = REGIMES[regime]
    return {
        "suite_version": SUITE_VERSION,
        "regime": regime,
        "d": d_value,
        "N": n_value,
        "R": args.R,
        "avg_indegree": kbar,
        "max_parents": MAX_PARENTS,
        "dag_generation_mode": "exact_edges",
        "alpha_low": config["alpha_low"],
        "alpha_high": config["alpha_high"],
        "seed": args.seed,
        "methods": list(config["methods"]),
        "candidate_library": list(SIX_FAMILIES),
        "true_family_assignment": "iid categorical uniform with probability 1/6 per node",
    }


def run_cell(
    root: Path,
    regime: str,
    d_value: int,
    n_value: int,
    kbar: float,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = REGIMES[regime]
    methods = tuple(str(value) for value in config["methods"])
    cell = root / "_shared_cells" / regime / cell_tag(d_value, n_value, kbar)
    cell.mkdir(parents=True, exist_ok=True)
    manifest_path = cell / "cell_manifest.json"
    configuration = cell_configuration(regime, d_value, n_value, kbar, args)
    previous: Dict[str, object] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration") not in (None, configuration):
            raise RuntimeError(
                f"Cell {cell} contains another configuration; do not mix results"
            )
    cell_manifest: Dict[str, object] = {
        "status": "running",
        "created_at": previous.get("created_at", utc_now()),
        "last_started_at": utc_now(),
        "configuration": configuration,
    }
    atomic_write_json(manifest_path, cell_manifest)
    try:
        raw, summary = d.run_experiment(
            d_values=(d_value,),
            n=n_value,
            reps=args.R,
            methods=methods,
            max_parents=MAX_PARENTS,
            avg_indegree=kbar,
            seed=args.seed,
            outdir=str(cell),
            workers=args.workers,
            require_all_methods=True,
            alpha_low=float(config["alpha_low"]),
            alpha_high=float(config["alpha_high"]),
            resume=True,
            dag_generation_mode="exact_edges",
        )
        confusion = pd.read_csv(cell / "experiment1_family_confusion.csv")
        expected_rows = args.R * len(methods)
        if len(raw) != expected_rows:
            raise RuntimeError(
                f"strict cell row check failed at {cell}: {len(raw)}/{expected_rows}"
            )
        truth_columns = (
            "seed",
            "true_families",
            "true_exogenous_params",
            "true_alpha_matrix",
            "true_edges",
        )
        for _, group in raw.groupby("rep"):
            if any(group[column].nunique(dropna=False) != 1 for column in truth_columns):
                raise RuntimeError(f"methods did not receive identical truth at {cell}")
        cell_manifest["status"] = "complete"
        cell_manifest["completed_at"] = utc_now()
        cell_manifest["rows"] = len(raw)
        atomic_write_json(manifest_path, cell_manifest)
        return raw, summary, confusion
    except BaseException as exc:
        cell_manifest["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        cell_manifest["stopped_at"] = utc_now()
        cell_manifest["last_error"] = repr(exc)
        atomic_write_json(manifest_path, cell_manifest)
        raise


def annotate(
    frame: pd.DataFrame,
    regime: str,
    sweep: str,
    scan_column: str,
    scan_value: float,
    d_value: int,
    n_value: int,
    kbar: float,
) -> pd.DataFrame:
    output = frame.copy()
    prefix = {
        "alpha_regime": regime,
        "sweep": sweep,
        scan_column: scan_value,
        "anchor_d": d_value,
        "anchor_N": n_value,
        "kbar_in": kbar,
    }
    # Avoid inserting a duplicate native column such as d or N.
    for name, value in reversed(list(prefix.items())):
        if name in output.columns:
            output[name] = value
        else:
            output.insert(0, name, value)
    return output


def family_metrics(
    confusion: pd.DataFrame, scan_column: str
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_columns = ["alpha_regime", scan_column, "method"]
    for keys, group in confusion.groupby(group_columns, sort=True):
        regime, scan_value, method = keys
        total = int(group["count"].sum())
        correct = int(
            group.loc[
                group["true_family"] == group["selected_family"], "count"
            ].sum()
        )
        per_family: List[float] = []
        for family in SIX_FAMILIES:
            subset = group.loc[group["true_family"] == family]
            family_total = int(subset["count"].sum())
            if family_total:
                family_correct = int(
                    subset.loc[subset["selected_family"] == family, "count"].sum()
                )
                per_family.append(family_correct / family_total)
        rows.append(
            {
                "alpha_regime": regime,
                scan_column: scan_value,
                "method": method,
                "micro_family_accuracy": correct / total if total else math.nan,
                "macro_family_accuracy": (
                    float(np.mean(per_family)) if per_family else math.nan
                ),
                "represented_true_families": len(per_family),
                "node_predictions": total,
            }
        )
    return pd.DataFrame(rows)


def assignment_audit(raw: pd.DataFrame, scan_column: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    internal = raw.loc[raw["method"] == "LibraryDP"]
    for keys, group in internal.groupby(["alpha_regime", scan_column], sort=True):
        regime, scan_value = keys
        counts = {family: 0 for family in SIX_FAMILIES}
        for serialized in group["true_families"].astype(str):
            for family in serialized.split(","):
                counts[family] += 1
        total = sum(counts.values())
        for family in SIX_FAMILIES:
            proportion = counts[family] / total if total else math.nan
            rows.append(
                {
                    "alpha_regime": regime,
                    scan_column: scan_value,
                    "true_family": family,
                    "count": counts[family],
                    "total_nodes": total,
                    "proportion": proportion,
                    "expected_proportion": 1.0 / 6.0,
                    "deviation_from_one_sixth": proportion - 1.0 / 6.0,
                }
            )
    return pd.DataFrame(rows)


def write_sweep_outputs(
    root: Path,
    experiment_name: str,
    raw_frames: Sequence[pd.DataFrame],
    summary_frames: Sequence[pd.DataFrame],
    confusion_frames: Sequence[pd.DataFrame],
    scan_column: str,
    xlabel: str,
    suffix: str,
    regimes: Sequence[str],
    xticks: Sequence[float],
    log_x: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output = root / experiment_name
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(raw_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    confusion = pd.concat(confusion_frames, ignore_index=True)
    atomic_write_csv(raw, output / f"{experiment_name}_raw.csv")
    atomic_write_csv(summary, output / f"{experiment_name}_summary.csv")
    atomic_write_csv(confusion, output / f"{experiment_name}_family_confusion.csv")
    atomic_write_csv(
        family_metrics(confusion, scan_column),
        output / f"{experiment_name}_family_metrics.csv",
    )
    atomic_write_csv(
        assignment_audit(raw, scan_column),
        output / f"{experiment_name}_family_assignment_audit.csv",
    )
    plot_sweep_summary(
        summary,
        x_column=scan_column,
        xlabel=xlabel,
        suffix=suffix,
        outdir=output,
        regimes=regimes,
        log_x=log_x,
        xticks=xticks,
    )
    if scan_column == "kbar_in":
        plot_realized_indegree(
            summary,
            x_column=scan_column,
            xlabel=xlabel,
            outdir=output,
            regimes=regimes,
        )
    return raw, summary


def normalized_alpha_matrix(serialized: str, low: float, high: float) -> np.ndarray:
    matrix = np.asarray(json.loads(serialized), dtype=float)
    nonzero = matrix > 0.0
    normalized = np.zeros_like(matrix)
    normalized[nonzero] = (matrix[nonzero] - low) / (high - low)
    return normalized


def reproducibility_audit(
    cell_results: Mapping[Tuple[str, int, int, float], pd.DataFrame],
    anchor_n: int,
) -> Dict[str, object]:
    """Verify the intended cross-regime, N, and density pairing exactly."""
    failures: List[str] = []
    checks = 0
    # Common versus expanded: same graph/family/parameters and same underlying
    # uniform coefficient draws after rescaling each alpha interval.
    configurations = sorted({key[1:] for key in cell_results})
    for d_value, n_value, kbar in configurations:
        common = cell_results.get(("common", d_value, n_value, kbar))
        expanded = cell_results.get(("expanded", d_value, n_value, kbar))
        if common is None or expanded is None:
            continue
        left = common.loc[common["method"] == "LibraryDP"].sort_values("rep")
        right = expanded.loc[expanded["method"] == "LibraryDP"].sort_values("rep")
        for lrow, rrow in zip(left.itertuples(), right.itertuples()):
            checks += 1
            if (
                lrow.true_edges != rrow.true_edges
                or lrow.true_families != rrow.true_families
                or lrow.true_exogenous_params != rrow.true_exogenous_params
            ):
                failures.append(
                    f"regime truth mismatch d={d_value},N={n_value},k={kbar},rep={lrow.rep}"
                )
                continue
            common_u = normalized_alpha_matrix(lrow.true_alpha_matrix, 0.15, 0.85)
            expanded_u = normalized_alpha_matrix(rrow.true_alpha_matrix, 0.2, 2.0)
            if not np.allclose(common_u, expanded_u, atol=2e-14, rtol=0.0):
                failures.append(
                    f"regime alpha pairing mismatch d={d_value},N={n_value},k={kbar},rep={lrow.rep}"
                )

    # N-sweep: truth must be identical for each (regime, replicate).
    for regime in ("common", "expanded"):
        frames = [
            frame.loc[frame["method"] == "LibraryDP"]
            for (name, d_value, _, kbar), frame in cell_results.items()
            if name == regime and d_value == ANCHOR_D and abs(kbar - ANCHOR_KBAR) < 1e-12
        ]
        if len(frames) > 1:
            combined = pd.concat(frames, ignore_index=True)
            for rep, group in combined.groupby("rep"):
                checks += 1
                for column in (
                    "true_edges",
                    "true_families",
                    "true_exogenous_params",
                    "true_alpha_matrix",
                ):
                    if group[column].nunique(dropna=False) != 1:
                        failures.append(f"N-pairing mismatch {regime},rep={rep},{column}")

    # Density-sweep: lower graphs must be edge subsets, shared alpha and all
    # exogenous specifications must be unchanged.
    for regime in ("common", "expanded"):
        density_frames = {
            kbar: frame.loc[frame["method"] == "LibraryDP"].set_index("rep")
            for (name, d_value, n_value, kbar), frame in cell_results.items()
            if name == regime and d_value == ANCHOR_D and n_value == anchor_n
        }
        ordered = sorted(density_frames)
        for lower_k, upper_k in zip(ordered, ordered[1:]):
            lower, upper = density_frames[lower_k], density_frames[upper_k]
            for rep in sorted(set(lower.index) & set(upper.index)):
                checks += 1
                lrow, urow = lower.loc[rep], upper.loc[rep]
                if (
                    lrow["true_families"] != urow["true_families"]
                    or lrow["true_exogenous_params"] != urow["true_exogenous_params"]
                ):
                    failures.append(f"density exogenous mismatch {regime},rep={rep}")
                    continue
                low_a = np.asarray(json.loads(lrow["true_alpha_matrix"]), dtype=float)
                high_a = np.asarray(json.loads(urow["true_alpha_matrix"]), dtype=float)
                low_edges = low_a > 0.0
                if np.any(low_edges & ~(high_a > 0.0)) or not np.allclose(
                    low_a[low_edges], high_a[low_edges], atol=2e-14, rtol=0.0
                ):
                    failures.append(
                        f"density nesting mismatch {regime},rep={rep},{lower_k}->{upper_k}"
                    )
    if failures:
        raise RuntimeError("Reproducibility audit failed: " + "; ".join(failures[:20]))
    return {"status": "passed", "pairing_checks": checks, "failures": 0}


def copy_paper_figures(root: Path, experiments: Iterable[str]) -> None:
    destination = root / "PAPER_FIGURES"
    destination.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        source_root = root / experiment / "figures"
        if not source_root.exists():
            continue
        for source in source_root.rglob("*.png"):
            relative = source.relative_to(source_root)
            target_name = f"{experiment}__{'__'.join(relative.parts)}"
            shutil.copy2(source, destination / target_name)
    readme = (
        "Paper-final six-family PT-SEM figures\n"
        "=======================================\n"
        "Each filename records experiment, alpha regime, metric, and scan axis.\n"
        "Main text: common alpha. Expanded alpha is the alpha>1 extension.\n"
        "OracleDP is retained in CSV output as a diagnostic upper bound and is\n"
        "not included in the primary plots.\n"
    )
    (destination / "README_FIRST.txt").write_text(readme, encoding="utf-8")


def suite_configuration(args: argparse.Namespace, preflight: Dict[str, object]) -> Dict[str, object]:
    regimes = selected_regimes(args.regime)
    return json.loads(
        json.dumps(
            {
                "suite_version": SUITE_VERSION,
                "R": args.R,
                "workers": args.workers,
                "seed": args.seed,
                "smoke": args.smoke,
                "sweeps": list(args.sweeps),
                "d_values": list(args.d_values),
                "N_values": list(args.N_values),
                "kbar_values": list(args.kbar_values),
                "anchor": {"d": ANCHOR_D, "N": args.anchor_N, "kbar_in": ANCHOR_KBAR},
                "max_parents": MAX_PARENTS,
                "dag_generation_mode": "exact_edges",
                "candidate_library": list(SIX_FAMILIES),
                "true_family_assignment": {
                    "model": "iid categorical uniform",
                    "probability_per_family": 1.0 / 6.0,
                    "forced_coexistence": False,
                },
                "exogenous_parameter_ranges": {
                    "Poisson_lambda": [2.0, 10.0],
                    "NB_r": [2.0, 10.0],
                    "NB_p": [0.15, 0.85],
                    "ZIP_lambda": [2.0, 10.0],
                    "ZIP_rho": [0.15, 0.85],
                    "Geom_mean": [2.0, 10.0],
                    "Binomial_n_integers": [d.BINOMIAL_N_MIN, d.BINOMIAL_N_MAX],
                    "Binomial_p": [d.BINOMIAL_P_MIN, d.BINOMIAL_P_MAX],
                    "Bernoulli_p": [d.BERNOULLI_P_MIN, d.BERNOULLI_P_MAX],
                },
                "regimes": {name: REGIMES[name] for name in regimes},
                "fixed_family_ablations": "excluded",
                "CPCM": "excluded",
                "dependency_versions": dependency_versions(),
                "preflight": preflight,
            }
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.results_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    preflight = run_preflight(args.seed)
    print(
        "Preflight passed: six likelihoods, moment maps, exact DP, iid family "
        "assignment, and expanded-alpha count generation.",
        flush=True,
    )
    if args.preflight_only:
        atomic_write_json(root / "preflight_report.json", preflight)
        print(f"Preflight report: {root / 'preflight_report.json'}")
        return

    configuration = suite_configuration(args, preflight)
    manifest_path = root / "final_suite_manifest.json"
    previous: Dict[str, object] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration") not in (None, configuration):
            raise RuntimeError(
                "The selected results root contains another configuration. "
                "Use a new --results-root; final experiments must never be mixed."
            )
    manifest: Dict[str, object] = {
        "status": "running",
        "created_at": previous.get("created_at", utc_now()),
        "last_started_at": utc_now(),
        "configuration": configuration,
        "completed_cells": [],
        "sleep_prevention": "ES_CONTINUOUS | ES_SYSTEM_REQUIRED",
    }
    atomic_write_json(manifest_path, manifest)

    regimes = selected_regimes(args.regime)
    cell_results: Dict[Tuple[str, int, int, float], pd.DataFrame] = {}
    completed_cells: List[str] = []
    generated_experiments: List[str] = []
    all_summaries: List[pd.DataFrame] = []
    try:
        with prevent_windows_sleep():
            if not args.skip_setup:
                setup_external_baselines.main()
            for regime in regimes:
                print(
                    f"Starting {regime} alpha regime: "
                    f"[{REGIMES[regime]['alpha_low']}, {REGIMES[regime]['alpha_high']}]",
                    flush=True,
                )
                for sweep in args.sweeps:
                    raw_frames: List[pd.DataFrame] = []
                    summary_frames: List[pd.DataFrame] = []
                    confusion_frames: List[pd.DataFrame] = []
                    if sweep == "d":
                        cells = [
                            (value, args.anchor_N, ANCHOR_KBAR, float(value))
                            for value in args.d_values
                        ]
                        experiment = "experiment1_d_sweep"
                        scan_column, xlabel, suffix = "d", "number of nodes d", "d"
                        xticks, log_x = args.d_values, False
                    elif sweep == "N":
                        cells = [
                            (ANCHOR_D, value, ANCHOR_KBAR, float(value))
                            for value in args.N_values
                        ]
                        experiment = "experiment2_N_sweep"
                        scan_column, xlabel, suffix = "N", "sample size N", "N"
                        xticks, log_x = args.N_values, True
                    else:
                        cells = [
                            (ANCHOR_D, args.anchor_N, value, float(value))
                            for value in args.kbar_values
                        ]
                        experiment = "experiment3_kin_sweep"
                        scan_column, xlabel, suffix = (
                            "kbar_in",
                            "target average indegree",
                            "kbar",
                        )
                        xticks, log_x = args.kbar_values, False

                    for d_value, n_value, kbar, scan_value in cells:
                        key = (regime, int(d_value), int(n_value), float(kbar))
                        if key not in cell_results:
                            print(
                                f"[{regime}/{sweep}] d={d_value}, N={n_value}, "
                                f"kbar={kbar}",
                                flush=True,
                            )
                            raw, summary, confusion = run_cell(
                                root,
                                regime,
                                int(d_value),
                                int(n_value),
                                float(kbar),
                                args,
                            )
                            cell_results[key] = raw
                            cell_name = f"{regime}:{cell_tag(d_value, n_value, kbar)}"
                            if cell_name not in completed_cells:
                                completed_cells.append(cell_name)
                                manifest["completed_cells"] = completed_cells
                                atomic_write_json(manifest_path, manifest)
                        else:
                            raw = cell_results[key]
                            cell = (
                                root
                                / "_shared_cells"
                                / regime
                                / cell_tag(d_value, n_value, kbar)
                            )
                            summary = pd.read_csv(cell / "experiment1_summary.csv")
                            confusion = pd.read_csv(
                                cell / "experiment1_family_confusion.csv"
                            )
                        raw_frames.append(
                            annotate(
                                raw,
                                regime,
                                sweep,
                                scan_column,
                                scan_value,
                                d_value,
                                n_value,
                                kbar,
                            )
                        )
                        summary_frames.append(
                            annotate(
                                summary,
                                regime,
                                sweep,
                                scan_column,
                                scan_value,
                                d_value,
                                n_value,
                                kbar,
                            )
                        )
                        confusion_frames.append(
                            annotate(
                                confusion,
                                regime,
                                sweep,
                                scan_column,
                                scan_value,
                                d_value,
                                n_value,
                                kbar,
                            )
                        )

                    # Merge with an already written other regime so each
                    # experiment remains useful even if the suite stops later.
                    exp_root = root / experiment
                    existing_raw: List[pd.DataFrame] = []
                    existing_summary: List[pd.DataFrame] = []
                    existing_confusion: List[pd.DataFrame] = []
                    raw_path = exp_root / f"{experiment}_raw.csv"
                    summary_path = exp_root / f"{experiment}_summary.csv"
                    confusion_path = exp_root / f"{experiment}_family_confusion.csv"
                    if raw_path.exists():
                        previous_raw = pd.read_csv(raw_path)
                        previous_summary = pd.read_csv(summary_path)
                        previous_confusion = pd.read_csv(confusion_path)
                        keep = previous_raw["alpha_regime"] != regime
                        existing_raw.append(previous_raw.loc[keep])
                        keep = previous_summary["alpha_regime"] != regime
                        existing_summary.append(previous_summary.loc[keep])
                        keep = previous_confusion["alpha_regime"] != regime
                        existing_confusion.append(previous_confusion.loc[keep])
                    combined_raw = [*existing_raw, *raw_frames]
                    combined_summary = [*existing_summary, *summary_frames]
                    combined_confusion = [*existing_confusion, *confusion_frames]
                    _, sweep_summary = write_sweep_outputs(
                        root,
                        experiment,
                        combined_raw,
                        combined_summary,
                        combined_confusion,
                        scan_column,
                        xlabel,
                        suffix,
                        tuple(
                            value
                            for value in ("common", "expanded")
                            if any(
                                frame["alpha_regime"].eq(value).any()
                                for frame in combined_summary
                            )
                        ),
                        xticks,
                        log_x,
                    )
                    if experiment not in generated_experiments:
                        generated_experiments.append(experiment)
                    all_summaries.append(sweep_summary)

        pairing = reproducibility_audit(cell_results, args.anchor_N)
        atomic_write_json(root / "reproducibility_audit.json", pairing)
        # Re-read final summaries so the top-level file contains each row once.
        final_summary_frames = [
            pd.read_csv(root / name / f"{name}_summary.csv")
            for name in generated_experiments
        ]
        paper_summary = pd.concat(final_summary_frames, ignore_index=True)
        atomic_write_csv(paper_summary, root / "paper_final_all_sweeps_summary.csv")
        copy_paper_figures(root, generated_experiments)
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        manifest["reproducibility_audit"] = pairing
        atomic_write_json(manifest_path, manifest)
        print(f"All requested six-family experiments completed: {root}", flush=True)
        print(f"Paper figures: {root / 'PAPER_FIGURES'}", flush=True)
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        manifest["stopped_at"] = utc_now()
        manifest["last_error"] = repr(exc)
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
