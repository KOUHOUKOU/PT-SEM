#!/usr/bin/env python3
"""Run the complete JMLR all-Poisson 2x3 experiment suite."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Prevent numerical-library oversubscription before NumPy/SciPy are imported.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC))

import all_poisson_framework as formal


CONFIG_PATH = PACKAGE_ROOT / "config" / "all_poisson.json"
RESULT_ROOT: Path | None = None
FORMAL_ROOT: Path | None = None
METADATA_ROOT: Path | None = None
LOG_ROOT: Path | None = None
PREFLIGHT_ROOT: Path | None = None


def configure_run_root(run_root: Path) -> None:
    global RESULT_ROOT, FORMAL_ROOT, METADATA_ROOT, LOG_ROOT, PREFLIGHT_ROOT
    RESULT_ROOT = run_root.resolve() / "all_poisson"
    FORMAL_ROOT = RESULT_ROOT / "formal_cells"
    METADATA_ROOT = RESULT_ROOT / "metadata"
    LOG_ROOT = RESULT_ROOT / "logs" / "formal_cells"
    PREFLIGHT_ROOT = RESULT_ROOT / "preflight"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_value(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".15g")


def path_value(value: object) -> str:
    return canonical_value(value).replace("-", "m").replace(".", "p")


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def stable_cell_base_seed(master_seed: int, sweep: str, value: object) -> int:
    """Return the paper-framework seed shared by every Table 2 cell.

    ``sweep`` and ``value`` are accepted deliberately so call sites document
    the cell, but are not mixed into the seed.  The simulation core derives
    data and method streams from this master seed, dimension, replication and
    explicit stream ID.  This preserves cross-regime truth pairing, identical
    truth over the N sweep, and nested graphs over the density sweep.
    """
    del sweep, value
    return int(master_seed)


def load_and_validate_config() -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["version"] != "ptsem_final_all_poisson_v2":
        raise RuntimeError("Unexpected All-Poisson config version")
    if getattr(formal.d, "NUMERICAL_CORE_VERSION", None) != "ptsem_submission_corrected_v3":
        raise RuntimeError("The required simulation core version is not active")
    if config["candidate_library"] != list(formal.SIX_FAMILIES):
        raise RuntimeError("Candidate library differs from the formal six-family suite")
    if config["true_exogenous_family"] != "Poisson":
        raise RuntimeError("True family must be Poisson")
    if config["poisson_lambda"] != {
        "distribution": "Uniform",
        "low": 2.0,
        "high": 10.0,
    }:
        raise RuntimeError("Table 1 Poisson parameter range is not exact")
    expected_restricted = {
        "LibraryDP",
        "LibraryGreedy",
        "OracleDP",
        "PoissonDAG-ODS",
        "PC-RCIT",
        "PBSCM",
        "PBSCM_PGF",
    }
    expected_extended = expected_restricted - {"PBSCM", "PBSCM_PGF"}
    if set(config["regimes"]["restricted"]["methods"]) != expected_restricted:
        raise RuntimeError("Restricted method set is not exact")
    if set(config["regimes"]["extended"]["methods"]) != expected_extended:
        raise RuntimeError("Extended method set is not exact")
    if config["regimes"]["restricted"]["alpha_low"] != 0.15 or config["regimes"]["restricted"]["alpha_high"] != 0.85:
        raise RuntimeError("Restricted coefficient range is not exact")
    if config["regimes"]["extended"]["alpha_low"] != 0.2 or config["regimes"]["extended"]["alpha_high"] != 2.0:
        raise RuntimeError("Extended coefficient range is not exact")
    return config


def build_cells(config: dict[str, object]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    master_seed = int(config["master_seed"])
    for regime in ("restricted", "extended"):
        regime_config = config["regimes"][regime]
        for sweep in ("dimension", "sample_size", "average_in_degree"):
            sweep_config = config["sweeps"][sweep]
            for value in sweep_config["values"]:
                if sweep == "dimension":
                    dimension = int(value)
                    sample_size = int(sweep_config["N"])
                    average_indegree = float(sweep_config["average_indegree"])
                elif sweep == "sample_size":
                    dimension = int(sweep_config["d"])
                    sample_size = int(value)
                    average_indegree = float(sweep_config["average_indegree"])
                else:
                    dimension = int(sweep_config["d"])
                    sample_size = int(sweep_config["N"])
                    average_indegree = float(value)
                cells.append(
                    {
                        "regime": regime,
                        "sweep": sweep,
                        "sweep_value": value,
                        "d": dimension,
                        "N": sample_size,
                        "average_indegree": average_indegree,
                        "max_parents": int(config["dag"]["max_parents"]),
                        "alpha_low": float(regime_config["alpha_low"]),
                        "alpha_high": float(regime_config["alpha_high"]),
                        "methods": list(regime_config["methods"]),
                        "cell_base_seed": stable_cell_base_seed(master_seed, sweep, value),
                        "shared_cell_id": (
                            f"{regime}_d{dimension}_N{sample_size}_"
                            f"k{path_value(average_indegree)}"
                        ),
                    }
                )
    return cells


def unique_physical_cells(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    """Run the shared Table 2 anchor once per coefficient regime."""
    unique: dict[str, dict[str, object]] = {}
    for cell in cells:
        unique.setdefault(str(cell["shared_cell_id"]), cell)
    return list(unique.values())


def cell_directory(cell: dict[str, object]) -> Path:
    return (
        FORMAL_ROOT
        / str(cell["regime"])
        / f"d{cell['d']}_N{cell['N']}_k{path_value(cell['average_indegree'])}"
    )


def parse_alpha_values(serialized: str) -> list[float]:
    matrix = np.asarray(json.loads(serialized), dtype=float)
    return matrix[matrix > 0.0].tolist()


def validate_cell(raw: pd.DataFrame, cell: dict[str, object], reps: int) -> dict[str, object]:
    methods = set(str(value) for value in cell["methods"])
    expected_rows = reps * len(methods)
    if len(raw) != expected_rows:
        raise RuntimeError(f"Cell row count {len(raw)} != {expected_rows}")
    key_counts = raw.groupby(["d", "rep", "method"]).size()
    if not key_counts.eq(1).all():
        raise RuntimeError("Duplicate method rows exist in formal cell")
    if set(raw["method"].astype(str)) != methods:
        raise RuntimeError("Formal cell method set differs from config")
    if set(raw["rep"].astype(int)) != set(range(reps)):
        raise RuntimeError("Formal cell replication IDs are incomplete")
    expected_family = ",".join(["Poisson"] * int(cell["d"]))
    if set(raw["true_families"].astype(str)) != {expected_family}:
        raise RuntimeError("A non-Poisson true family entered the formal cell")
    unique_data = raw.drop_duplicates(["d", "rep"])
    lambdas: list[float] = []
    alphas: list[float] = []
    for serialized in unique_data["true_exogenous_params"].astype(str):
        params = json.loads(serialized)
        if len(params) != int(cell["d"]) or any(set(item) != {"lam"} for item in params):
            raise RuntimeError("Exogenous parameter schema is not all-Poisson")
        lambdas.extend(float(item["lam"]) for item in params)
    for serialized in unique_data["true_alpha_matrix"].astype(str):
        alphas.extend(parse_alpha_values(serialized))
    if not lambdas or min(lambdas) < 2.0 or max(lambdas) > 10.0:
        raise RuntimeError("Poisson lambda is outside Table 1 range")
    if not alphas or min(alphas) < float(cell["alpha_low"]) or max(alphas) > float(cell["alpha_high"]):
        raise RuntimeError("Edge coefficient is outside the configured regime")
    expected_edges = int(round(int(cell["d"]) * float(cell["average_indegree"])))
    if set(unique_data["n_true_edges"].astype(int)) != {expected_edges}:
        raise RuntimeError("Exact-edge DAG construction failed")
    truth_fields = ["seed", "true_families", "true_exogenous_params", "true_alpha_matrix", "true_edges"]
    for _, group in raw.groupby(["d", "rep"], sort=False):
        if any(group[field].astype(str).nunique(dropna=False) != 1 for field in truth_fields):
            raise RuntimeError("Methods did not share identical data/truth")
    if not raw["directed_f1"].between(0.0, 1.0).all():
        raise RuntimeError("Directed F1 is missing or outside [0,1]")
    if str(cell["regime"]) == "extended" and ({"PBSCM", "PBSCM_PGF"} & methods):
        raise RuntimeError("PB-SCM method entered the extended regime")
    return {
        "status": "passed",
        "rows": len(raw),
        "replications": reps,
        "methods": sorted(methods),
        "lambda_min": min(lambdas),
        "lambda_max": max(lambdas),
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "true_edges": expected_edges,
    }


def seed_plan(config: dict[str, object], cells: list[dict[str, object]], reps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell in cells:
        for replication in range(reps):
            base = int(cell["cell_base_seed"])
            data_seed = formal.d.derive_seed(
                base, int(cell["d"]), replication, stream=0
            )
            for method in cell["methods"]:
                stream = int(formal.d.METHOD_STREAM[str(method)])
                rows.append(
                    {
                        "source": "simulation_core",
                        "regime": cell["regime"],
                        "sweep": cell["sweep"],
                        "sweep_value": cell["sweep_value"],
                        "d": cell["d"],
                        "N": cell["N"],
                        "average_in_degree": cell["average_indegree"],
                        "replication": replication,
                        "method_internal": method,
                        "method_display": config["display_names"][str(method)],
                        "method_stream": stream,
                        "cell_base_seed": base,
                        "data_seed": data_seed,
                        "method_seed": formal.d.derive_seed(
                            base,
                            int(cell["d"]),
                            replication,
                            stream=stream,
                        ),
                        "paired_across_regimes": True,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["regime", "sweep", "sweep_value", "replication", "method_internal"],
        kind="stable",
    )
    return frame.reset_index(drop=True)


def compare_deterministic_outputs(first: pd.DataFrame, second: pd.DataFrame) -> float:
    ignored = {
        "runtime_sec",
        "local_score_runtime_sec",
        "search_runtime_sec",
        "parallel_workers",
    }
    columns = [column for column in first.columns if column not in ignored]
    order = ["d", "rep", "method"]
    left = first[columns].sort_values(order).reset_index(drop=True)
    right = second[columns].sort_values(order).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        raise RuntimeError("Determinism comparison schema or row count differs")
    maximum_float_difference = 0.0
    for column in columns:
        left_values = left[column]
        right_values = right[column]
        if pd.api.types.is_numeric_dtype(left_values) and pd.api.types.is_numeric_dtype(right_values):
            left_numeric = pd.to_numeric(left_values).to_numpy(float)
            right_numeric = pd.to_numeric(right_values).to_numpy(float)
            finite = np.isfinite(left_numeric) & np.isfinite(right_numeric)
            if finite.any():
                maximum_float_difference = max(
                    maximum_float_difference,
                    float(np.max(np.abs(left_numeric[finite] - right_numeric[finite]))),
                )
            if not np.allclose(
                left_numeric,
                right_numeric,
                rtol=0.0,
                atol=5e-15,
                equal_nan=True,
            ):
                raise RuntimeError(f"Deterministic numeric field differs: {column}")
        elif not left_values.fillna("").astype(str).equals(
            right_values.fillna("").astype(str)
        ):
            raise RuntimeError(f"Deterministic discrete field differs: {column}")
    return maximum_float_difference


def run_preflight(config: dict[str, object]) -> dict[str, object]:
    started = utc_now()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = PREFLIGHT_ROOT / stamp
    root.mkdir(parents=True, exist_ok=False)
    audit = formal.framework_provenance()
    if audit.get("numerical_core_version") != "ptsem_submission_corrected_v3":
        raise RuntimeError("Preflight refused: required numerical core is absent")

    X, A, specs = formal.d.simulate_ptsem(4, 80, 918273645, 1.5, 5, 0.2, 2.0, "exact_edges")
    if not np.issubdtype(X.dtype, np.integer) or (X < 0).any():
        raise RuntimeError("Formal All-Poisson sample is not nonnegative integer-valued")
    if [spec.family for spec in specs] != ["Poisson"] * 4:
        raise RuntimeError("Formal specialization did not force every true family to Poisson")
    lambdas = [float(spec.params["lam"]) for spec in specs]
    if min(lambdas) < 2.0 or max(lambdas) > 10.0:
        raise RuntimeError("Preflight lambda range failed")
    if len(formal.d.edges_from_A(A)) != round(4 * 1.5):
        raise RuntimeError("Preflight exact-edge count failed")

    # In an All-Poisson truth, the formal OracleDP path is exactly the
    # Poisson-only family restriction.  Verify every admissible local score,
    # rather than merely comparing the final graph on one data set.
    true_families = ["Poisson"] * 4
    oracle_scores, _ = formal.d.precompute_local_scores(
        X, family_options=[("Poisson",)] * X.shape[1]
    )
    mean, covariance = formal.d.empirical_moments(X)
    oracle_score_checks = 0
    for child in range(4):
        for parent_mask in range(1 << 4):
            if (parent_mask >> child) & 1:
                continue
            direct = formal.d.local_score(
                X, mean, covariance, child, parent_mask, ("Poisson",)
            ).bic
            if oracle_scores[child][parent_mask] != direct:
                raise RuntimeError("OracleDP differs from the Poisson-only local score")
            oracle_score_checks += 1

    test_methods = ("LibraryDP", "LibraryGreedy", "OracleDP")
    common_kwargs = dict(
        d_values=(4,),
        n=80,
        reps=2,
        methods=test_methods,
        max_parents=5,
        avg_indegree=1.5,
        seed=77665544,
        require_all_methods=True,
        alpha_low=0.15,
        alpha_high=0.85,
        resume=False,
        dag_generation_mode="exact_edges",
    )
    single, _ = formal.run_experiment(outdir=str(root / "single_worker"), workers=1, **common_kwargs)
    multi, _ = formal.run_experiment(outdir=str(root / "two_workers"), workers=2, **common_kwargs)
    worker_max_difference = compare_deterministic_outputs(single, multi)

    # Exercise the exact checkpoint path used by an interrupted overnight run:
    # retain rep=0 from an R=1 run, resume to R=2, and compare with the fresh R=2
    # reference above.
    resume_kwargs = dict(common_kwargs)
    resume_kwargs["reps"] = 1
    formal.run_experiment(
        outdir=str(root / "resume_worker_change"),
        workers=1,
        **resume_kwargs,
    )
    resume_kwargs["reps"] = 2
    resume_kwargs["resume"] = True
    resumed, _ = formal.run_experiment(
        outdir=str(root / "resume_worker_change"),
        workers=2,
        **resume_kwargs,
    )
    resume_max_difference = compare_deterministic_outputs(multi, resumed)

    smoke_methods = tuple(config["regimes"]["restricted"]["methods"])
    smoke, _ = formal.run_experiment(
        d_values=(4,),
        n=100,
        reps=1,
        methods=smoke_methods,
        max_parents=5,
        avg_indegree=1.5,
        seed=88776655,
        outdir=str(root / "all_methods_smoke"),
        workers=1,
        require_all_methods=True,
        alpha_low=0.15,
        alpha_high=0.85,
        resume=False,
        dag_generation_mode="exact_edges",
    )
    smoke_cell = {
        "regime": "restricted",
        "d": 4,
        "average_indegree": 1.5,
        "alpha_low": 0.15,
        "alpha_high": 0.85,
        "methods": list(smoke_methods),
    }
    smoke_validation = validate_cell(smoke, smoke_cell, 1)
    report = {
        "status": "passed",
        "started_at": started,
        "completed_at": utc_now(),
        "framework": audit,
        "config_path": str(CONFIG_PATH),
        "specialization_check": {
            "families": [spec.family for spec in specs],
            "lambdas": lambdas,
            "edge_count": len(formal.d.edges_from_A(A)),
            "nonnegative_integer_sample": True,
        },
        "single_vs_two_worker_max_nonruntime_float_difference": worker_max_difference,
        "resume_from_R1_to_R2_max_nonruntime_float_difference": resume_max_difference,
        "determinism_float_tolerance": 5e-15,
        "determinism_discrete_fields": "exact string equality",
        "oracle_dp_poisson_only_local_score_checks": oracle_score_checks,
        "all_required_methods_smoke": smoke_validation,
        "output_root": str(root),
    }
    atomic_json(report, METADATA_ROOT / "framework_validation.json")
    return report


def run_cell(cell: dict[str, object], reps: int, workers: int) -> dict[str, object]:
    output = cell_directory(cell)
    output.mkdir(parents=True, exist_ok=True)
    log_path = (
        LOG_ROOT
        / str(cell["regime"])
        / str(cell["sweep"])
        / f"value_{path_value(cell['sweep_value'])}.json"
    )
    metadata = {
        **cell,
        "R": reps,
        "workers": workers,
        "output_path": str(output.relative_to(PACKAGE_ROOT)).replace("\\", "/"),
        "started_at": utc_now(),
        "status": "running",
        "exception": "",
    }
    atomic_json(metadata, log_path)
    started = time.perf_counter()
    try:
        raw, _ = formal.run_experiment(
            d_values=(int(cell["d"]),),
            n=int(cell["N"]),
            reps=reps,
            methods=tuple(cell["methods"]),
            max_parents=int(cell["max_parents"]),
            avg_indegree=float(cell["average_indegree"]),
            seed=int(cell["cell_base_seed"]),
            outdir=str(output),
            workers=workers,
            require_all_methods=True,
            alpha_low=float(cell["alpha_low"]),
            alpha_high=float(cell["alpha_high"]),
            resume=True,
            dag_generation_mode="exact_edges",
        )
        validation = validate_cell(raw, cell, reps)
        metadata.update(
            status="success",
            completed_at=utc_now(),
            runtime_sec=time.perf_counter() - started,
            validation=validation,
        )
        atomic_json(metadata, log_path)
        return metadata
    except BaseException as exc:
        metadata.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            completed_at=utc_now(),
            runtime_sec=time.perf_counter() - started,
            exception=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        atomic_json(metadata, log_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal-framework JMLR All-Poisson missing-sweep runner"
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_run_root(args.run_root)
    assert FORMAL_ROOT is not None
    assert METADATA_ROOT is not None
    assert LOG_ROOT is not None
    assert PREFLIGHT_ROOT is not None
    config = load_and_validate_config()
    workers = int(args.workers or config["workers"])
    reps = int(config["replications"])
    if workers < 1 or reps < 1:
        raise SystemExit("--workers and configured replications must be positive")
    cells = build_cells(config)
    physical_cells = unique_physical_cells(cells)
    atomic_csv(seed_plan(config, cells, reps), METADATA_ROOT / "seeds.csv")
    plan = {
        "config": str(CONFIG_PATH),
        "framework": formal.framework_provenance(),
        "workers": workers,
        "R": reps,
        "panel_cells": cells,
        "cells": physical_cells,
        "panel_cell_count": len(cells),
        "unique_physical_cell_count": len(physical_cells),
        "new_replications": len(physical_cells) * reps,
        "restricted_sample_size_reused": False,
        "historical_results_consumed": False,
    }
    atomic_json(plan, METADATA_ROOT / "formal_run_plan.json")
    print(json.dumps({"panel_cells": len(cells), "unique_physical_cells": len(physical_cells), "new_replications": len(physical_cells) * reps, "workers": workers}, indent=2), flush=True)
    if args.plan_only:
        return
    report = run_preflight(config)
    print(f"Formal framework preflight: {report['status']}", flush=True)
    if args.preflight_only:
        return
    for index, cell in enumerate(physical_cells, start=1):
        print(
            f"[{index}/{len(physical_cells)}] {cell['regime']} {cell['sweep']}={cell['sweep_value']} "
            f"d={cell['d']} N={cell['N']} k={cell['average_indegree']}",
            flush=True,
        )
        run_cell(cell, reps, workers)
    print("All formal All-Poisson cells passed strict validation.", flush=True)


if __name__ == "__main__":
    main()
