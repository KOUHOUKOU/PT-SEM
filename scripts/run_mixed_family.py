#!/usr/bin/env python3
"""Run the heterogeneous six-family Table 2 experiment cells.

The runner writes only inside the requested result root. Complete cells are
resume-safe and the
common anchor cell is physically run once per regime, then referenced by all
three sweeps during summarization.
"""

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

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC))

import mixed_family_framework as formal

CONFIG_PATH = PACKAGE_ROOT / "config" / "mixed_family.json"
RESULT_ROOT: Path | None = None
CELL_ROOT: Path | None = None
LOG_ROOT: Path | None = None
METADATA_ROOT: Path | None = None


def configure_run_root(run_root: Path) -> None:
    global RESULT_ROOT, CELL_ROOT, LOG_ROOT, METADATA_ROOT
    RESULT_ROOT = run_root.resolve() / "mixed_family"
    CELL_ROOT = RESULT_ROOT / "formal_cells"
    LOG_ROOT = RESULT_ROOT / "logs" / "formal_cells"
    METADATA_ROOT = RESULT_ROOT / "metadata"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def canonical(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".15g")


def path_value(value: object) -> str:
    return canonical(value).replace("-", "m").replace(".", "p")


def load_config() -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version") != "ptsem_final_mixed_family_v3":
        raise RuntimeError("Unexpected mixed-family config version")
    if config.get("numerical_core_version") != formal.d.NUMERICAL_CORE_VERSION:
        raise RuntimeError("Config/core numerical version mismatch")
    if tuple(config["family_library"]) != formal.SIX_FAMILIES:
        raise RuntimeError("Six-family library mismatch")
    if config.get("true_family_assignment") != "iid_uniform":
        raise RuntimeError("True families must be assigned iid uniformly")
    if config["sweeps"]["sample_size"]["values"] != [100, 200, 400, 800, 1600, 3200, 6400, 10000]:
        raise RuntimeError("Sample-size sweep differs from paper Table 2")
    if 2400 in config["sweeps"]["sample_size"]["values"]:
        raise RuntimeError("N=2400 is outside the specified sample-size sweep")
    restricted = set(config["regimes"]["restricted"]["methods"])
    extended = set(config["regimes"]["extended"]["methods"])
    expected = {"LibraryDP", "LibraryGreedy", "OracleDP", "PoissonDAG-ODS", "PC-RCIT", "PBSCM", "PBSCM_PGF"}
    if restricted != expected or extended != expected - {"PBSCM", "PBSCM_PGF"}:
        raise RuntimeError("Regime method sets are not the paper method sets")
    return config


def panel_cells(config: dict[str, object]) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for regime in ("restricted", "extended"):
        regime_config = config["regimes"][regime]
        for sweep in ("dimension", "sample_size", "average_in_degree"):
            spec = config["sweeps"][sweep]
            for value in spec["values"]:
                if sweep == "dimension":
                    d_value, n_value, k_value = int(value), int(spec["N"]), float(spec["average_indegree"])
                elif sweep == "sample_size":
                    d_value, n_value, k_value = int(spec["d"]), int(value), float(spec["average_indegree"])
                else:
                    d_value, n_value, k_value = int(spec["d"]), int(spec["N"]), float(value)
                shared_id = f"{regime}_d{d_value}_N{n_value}_k{path_value(k_value)}"
                cells.append({
                    "regime": regime, "sweep": sweep, "sweep_value": value,
                    "d": d_value, "N": n_value, "average_indegree": k_value,
                    "alpha_low": float(regime_config["alpha_low"]),
                    "alpha_high": float(regime_config["alpha_high"]),
                    "methods": list(regime_config["methods"]),
                    "max_parents": int(config["dag"]["max_parents"]),
                    "base_seed": int(config["master_seed"]),
                    "shared_cell_id": shared_id,
                })
    return cells


def unique_cells(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[str, dict[str, object]] = {}
    for cell in cells:
        unique.setdefault(str(cell["shared_cell_id"]), cell)
    return list(unique.values())


def cell_directory(cell: dict[str, object]) -> Path:
    if CELL_ROOT is None:
        raise RuntimeError("configure_run_root() must be called first")
    return CELL_ROOT / str(cell["regime"]) / f"d{cell['d']}_N{cell['N']}_k{path_value(cell['average_indegree'])}"


def parse_params_and_check(family: str, params: dict[str, object]) -> None:
    def between(value: object, low: float, high: float) -> bool:
        return low <= float(value) <= high
    if family == "Poisson" and set(params) == {"lam"} and between(params["lam"], 2, 10):
        return
    if family == "NB" and set(params) == {"p", "r"} and between(params["r"], 2, 10) and between(params["p"], .15, .85):
        return
    if family == "ZIP" and set(params) == {"lam", "rho"} and between(params["lam"], 2, 10) and between(params["rho"], .15, .85):
        return
    if family == "Geom" and set(params) == {"mean", "p"} and between(params["mean"], 2, 10) and abs(float(params["p"]) - 1/(1+float(params["mean"]))) <= 1e-12:
        return
    if family == "Binomial" and set(params) == {"n", "p"} and 2 <= int(params["n"]) <= 20 and between(params["p"], .15, .85):
        return
    if family == "Bernoulli" and set(params) == {"p"} and between(params["p"], .15, .85):
        return
    raise RuntimeError(f"Table 1 parameter violation: family={family}, params={params}")


def validate_cell(raw: pd.DataFrame, cell: dict[str, object], reps: int) -> dict[str, object]:
    methods = set(cell["methods"])
    if len(raw) != reps * len(methods):
        raise RuntimeError(f"Incomplete cell rows: {len(raw)}/{reps * len(methods)}")
    if set(raw["method"].astype(str)) != methods or set(raw["rep"].astype(int)) != set(range(reps)):
        raise RuntimeError("Cell method or replication set is incomplete")
    if not raw.groupby(["rep", "method"]).size().eq(1).all():
        raise RuntimeError("Duplicate replication/method rows")
    reference = raw.loc[raw["method"] == "LibraryDP"].sort_values("rep")
    allowed = set(formal.SIX_FAMILIES)
    family_counts = {family: 0 for family in formal.SIX_FAMILIES}
    for row in reference.itertuples(index=False):
        families = str(row.true_families).split(",")
        params = json.loads(str(row.true_exogenous_params))
        if len(families) != int(cell["d"]) or len(params) != int(cell["d"]) or not set(families).issubset(allowed):
            raise RuntimeError("Generated family vector is invalid")
        for family, parameter in zip(families, params):
            family_counts[family] += 1
            parse_params_and_check(family, parameter)
        alpha = np.asarray(json.loads(str(row.true_alpha_matrix)), dtype=float)
        nonzero = alpha[alpha > 0]
        if len(nonzero) != round(int(cell["d"]) * float(cell["average_indegree"])):
            raise RuntimeError("Exact-edge DAG count failed")
        if len(nonzero) and (nonzero.min() < float(cell["alpha_low"]) or nonzero.max() > float(cell["alpha_high"])):
            raise RuntimeError("Coefficient range failed")
    truth_fields = ["seed", "true_families", "true_exogenous_params", "true_alpha_matrix", "true_edges"]
    for _, group in raw.groupby("rep", sort=False):
        if any(group[field].astype(str).nunique(dropna=False) != 1 for field in truth_fields):
            raise RuntimeError("Methods did not share identical data truth")
    if str(cell["regime"]) == "extended" and ({"PBSCM", "PBSCM_PGF"} & methods):
        raise RuntimeError("PB-SCM entered extended regime")
    if not raw["directed_f1"].between(0, 1).all():
        raise RuntimeError("Directed F1 is missing or outside [0,1]")
    return {"status": "passed", "rows": len(raw), "replications": reps, "family_counts": family_counts}


def seed_plan(config: dict[str, object], cells: list[dict[str, object]], reps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell in cells:
        for rep in range(reps):
            for method in cell["methods"]:
                stream = int(formal.d.METHOD_STREAM[str(method)])
                rows.append({
                    "regime": cell["regime"], "sweep": cell["sweep"], "sweep_value": cell["sweep_value"],
                    "shared_cell_id": cell["shared_cell_id"], "d": cell["d"], "N": cell["N"],
                    "average_in_degree": cell["average_indegree"], "replication": rep,
                    "method_internal": method, "method_display": config["display_names"][method],
                    "master_seed": cell["base_seed"],
                    "data_seed": formal.d.derive_seed(int(cell["base_seed"]), int(cell["d"]), rep, stream=0),
                    "method_seed": formal.d.derive_seed(int(cell["base_seed"]), int(cell["d"]), rep, stream=stream),
                })
    return pd.DataFrame(rows)


def run_cell(cell: dict[str, object], reps: int, workers: int) -> dict[str, object]:
    output = cell_directory(cell)
    log_path = LOG_ROOT / str(cell["regime"]) / f"{cell['shared_cell_id']}.json"
    record = {**cell, "R": reps, "workers": workers, "output_path": str(output.relative_to(PACKAGE_ROOT)).replace("\\", "/"), "started_at": utc_now(), "status": "running", "exception": ""}
    atomic_json(record, log_path)
    started = time.perf_counter()
    try:
        raw, _ = formal.run_experiment(
            d_values=(int(cell["d"]),), n=int(cell["N"]), reps=reps,
            methods=tuple(cell["methods"]), max_parents=int(cell["max_parents"]),
            avg_indegree=float(cell["average_indegree"]), seed=int(cell["base_seed"]),
            outdir=str(output), workers=workers, require_all_methods=True,
            alpha_low=float(cell["alpha_low"]), alpha_high=float(cell["alpha_high"]),
            resume=True, dag_generation_mode="exact_edges",
        )
        record.update(status="success", completed_at=utc_now(), runtime_sec=time.perf_counter()-started, validation=validate_cell(raw, cell, reps))
        atomic_json(record, log_path)
        return record
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", completed_at=utc_now(), runtime_sec=time.perf_counter()-started, exception=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        atomic_json(record, log_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed-family Table 2 runner")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    configure_run_root(args.run_root)
    assert METADATA_ROOT is not None and LOG_ROOT is not None
    config = load_config()
    workers = int(args.workers or config["workers"])
    reps = int(config["replications"])
    memberships = panel_cells(config)
    physical = unique_cells(memberships)
    atomic_csv(seed_plan(config, memberships, reps), METADATA_ROOT / "seeds.csv")
    plan = {
        "status": "plan_only" if args.plan_only else "running", "created_at": utc_now(),
        "config": str(CONFIG_PATH.relative_to(PACKAGE_ROOT)).replace("\\", "/"),
        "framework": formal.framework_provenance(), "R": reps, "workers": workers,
        "panel_cells": len(memberships), "unique_physical_cells": len(physical),
        "datasets": len(physical) * reps,
        "method_jobs": sum(len(cell["methods"]) * reps for cell in physical),
        "historical_results_consumed": False, "cells": physical,
    }
    atomic_json(plan, METADATA_ROOT / "formal_run_plan.json")
    print(json.dumps({key: plan[key] for key in ("panel_cells", "unique_physical_cells", "datasets", "method_jobs", "workers")}, indent=2))
    if args.plan_only:
        return
    completed = []
    for index, cell in enumerate(physical, 1):
        print(f"[{index}/{len(physical)}] {cell['shared_cell_id']}", flush=True)
        completed.append(run_cell(cell, reps, workers))
    plan.update(status="complete", completed_at=utc_now(), completed_cells=len(completed))
    atomic_json(plan, METADATA_ROOT / "formal_run_plan.json")


if __name__ == "__main__":
    main()
