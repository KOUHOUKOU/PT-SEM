#!/usr/bin/env python3
"""Strictly validate and summarize mixed-family raw replications."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC))

import mixed_family_framework as formal
import run_mixed_family as runner

CONFIG_PATH = PACKAGE_ROOT / "config" / "mixed_family.json"
RESULT_ROOT: Path | None = None
RAW_ROOT: Path | None = None
SUMMARY_ROOT: Path | None = None


def configure_run_root(run_root: Path) -> None:
    global RESULT_ROOT, RAW_ROOT, SUMMARY_ROOT
    runner.configure_run_root(run_root)
    RESULT_ROOT = run_root.resolve() / "mixed_family"
    RAW_ROOT = RESULT_ROOT / "raw"
    SUMMARY_ROOT = RESULT_ROOT / "summaries"


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


def parse_edges(value: object) -> set[tuple[int, int]]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return set()
    edges: set[tuple[int, int]] = set()
    for item in text.split(";"):
        parent, child = item.split("->", 1)
        edges.add((int(parent), int(child)))
    return edges


def directed_f1(truth: set[tuple[int, int]], estimate: set[tuple[int, int]]) -> float:
    if not truth and not estimate:
        return 1.0
    tp = len(truth & estimate)
    precision = tp / len(estimate) if estimate else 0.0
    recall = tp / len(truth) if truth else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def recompute_metrics(row: pd.Series) -> tuple[float, float, float]:
    truth = parse_edges(row["true_edges"])
    estimate = parse_edges(row["estimated_directed_edges"])
    f1 = directed_f1(truth, estimate)
    # Alpha MAPE is defined only for the three likelihood-based methods that
    # return a complete fitted coefficient matrix.  External graph baselines
    # return edges only; their blank coefficient field must not be fabricated
    # or parsed as JSON.
    mape = math.nan
    if row["method"] in {"LibraryDP", "LibraryGreedy", "OracleDP"}:
        true_alpha = np.asarray(json.loads(str(row["true_alpha_matrix"])), dtype=float)
        estimated_alpha = np.asarray(json.loads(str(row["estimated_alpha_matrix"])), dtype=float)
        if true_alpha.shape != estimated_alpha.shape:
            raise RuntimeError("True and estimated alpha matrices differ in shape")
        correct = truth & estimate
        if correct:
            errors = [abs((estimated_alpha[child, parent] - true_alpha[child, parent]) / true_alpha[child, parent]) for parent, child in correct]
            mape = 100.0 * float(np.mean(errors))
    selected = str(row["selected_families"]).split(",")
    truth_families = str(row["true_families"]).split(",")
    if row["method"] in {"LibraryDP", "LibraryGreedy"}:
        if len(selected) != len(truth_families) or any(not item for item in selected):
            raise RuntimeError("Internal method lacks a complete selected-family vector")
        family_accuracy = float(np.mean(np.asarray(selected) == np.asarray(truth_families)))
    else:
        family_accuracy = math.nan
    return f1, mape, family_accuracy


def annotate_membership(raw: pd.DataFrame, cell: dict[str, object]) -> pd.DataFrame:
    output = raw.copy()
    output.insert(0, "regime", cell["regime"])
    output.insert(1, "sweep", cell["sweep"])
    output.insert(2, "sweep_value", cell["sweep_value"])
    output.insert(3, "shared_cell_id", cell["shared_cell_id"])
    output["average_in_degree"] = float(cell["average_indegree"])
    return output


def aggregate(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = ["regime", "sweep", "sweep_value", "d", "N", "average_in_degree", "method"]
    rows = []
    for values, group in raw.groupby(keys, sort=False, dropna=False):
        data = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
        total = len(group)
        count = len(data)
        mean = float(np.mean(data)) if count else math.nan
        sd = float(np.std(data, ddof=1)) if count > 1 else math.nan
        se = sd / math.sqrt(count) if count > 1 else math.nan
        rows.append(dict(zip(keys, values)) | {
            "metric": metric, "N_total": total, "N_nonNA": count,
            "NA_rate": 1.0 - count / total, "mean": mean, "sd": sd, "se": se,
            "ci95_low": mean - 1.96 * se if count > 1 else math.nan,
            "ci95_high": mean + 1.96 * se if count > 1 else math.nan,
        })
    return pd.DataFrame(rows)


def pairing_audit(physical: dict[str, pd.DataFrame], config: dict[str, object]) -> dict[str, object]:
    failures: list[str] = []
    checks = 0
    by_tuple: dict[tuple[str, int, int, float], pd.DataFrame] = {}
    cells = runner.unique_cells(runner.panel_cells(config))
    for cell in cells:
        by_tuple[(str(cell["regime"]), int(cell["d"]), int(cell["N"]), float(cell["average_indegree"]))] = physical[str(cell["shared_cell_id"])].loc[lambda x: x["method"] == "LibraryDP"].set_index("rep")
    for d_value, n_value, k_value in sorted({key[1:] for key in by_tuple}):
        left = by_tuple.get(("restricted", d_value, n_value, k_value))
        right = by_tuple.get(("extended", d_value, n_value, k_value))
        if left is None or right is None:
            continue
        for rep in range(int(config["replications"])):
            checks += 1
            lrow, rrow = left.loc[rep], right.loc[rep]
            if any(lrow[field] != rrow[field] for field in ("true_edges", "true_families", "true_exogenous_params")):
                failures.append(f"regime truth mismatch d={d_value},N={n_value},k={k_value},rep={rep}")
            la = np.asarray(json.loads(lrow["true_alpha_matrix"]), dtype=float)
            ra = np.asarray(json.loads(rrow["true_alpha_matrix"]), dtype=float)
            ln = np.where(la > 0, (la-.15)/(.85-.15), 0)
            rn = np.where(ra > 0, (ra-.2)/(2.0-.2), 0)
            if not np.allclose(ln, rn, rtol=0, atol=2e-14):
                failures.append(f"regime alpha mismatch d={d_value},N={n_value},k={k_value},rep={rep}")
    for regime in ("restricted", "extended"):
        n_cells = [(key, frame) for key, frame in by_tuple.items() if key[0] == regime and key[1] == 8 and abs(key[3]-1.5) < 1e-12]
        for rep in range(int(config["replications"])):
            checks += 1
            rows = [frame.loc[rep] for key, frame in n_cells]
            for field in ("true_edges", "true_families", "true_exogenous_params", "true_alpha_matrix"):
                if len({str(row[field]) for row in rows}) != 1:
                    failures.append(f"N sweep truth mismatch {regime},rep={rep},{field}")
        density = sorted([(key[3], frame) for key, frame in by_tuple.items() if key[0] == regime and key[1] == 8 and key[2] == 3200])
        for (lower_k, lower), (upper_k, upper) in zip(density, density[1:]):
            for rep in range(int(config["replications"])):
                checks += 1
                lrow, urow = lower.loc[rep], upper.loc[rep]
                if lrow["true_families"] != urow["true_families"] or lrow["true_exogenous_params"] != urow["true_exogenous_params"]:
                    failures.append(f"density exogenous mismatch {regime},rep={rep}")
                la = np.asarray(json.loads(lrow["true_alpha_matrix"]), dtype=float)
                ua = np.asarray(json.loads(urow["true_alpha_matrix"]), dtype=float)
                mask = la > 0
                if np.any(mask & ~(ua > 0)) or not np.allclose(la[mask], ua[mask], rtol=0, atol=2e-14):
                    failures.append(f"density nesting mismatch {regime},{lower_k}->{upper_k},rep={rep}")
    if failures:
        raise RuntimeError("Pairing audit failed: " + "; ".join(failures[:20]))
    return {"status": "passed", "checks": checks, "failures": 0}


def confusion_table(raw: pd.DataFrame) -> pd.DataFrame:
    anchor = raw.loc[(raw["sweep"] == "sample_size") & (raw["sweep_value"].astype(float) == 3200) & (raw["method"] == "LibraryDP")]
    counts: dict[tuple[str, str, str], int] = {}
    for row in anchor.itertuples(index=False):
        for truth, selected in zip(str(row.true_families).split(","), str(row.selected_families).split(",")):
            key = (str(row.regime), truth, selected)
            counts[key] = counts.get(key, 0) + 1
    rows = []
    for (regime, truth, selected), count in counts.items():
        denominator = sum(value for (r, t, _), value in counts.items() if r == regime and t == truth)
        rows.append({"regime": regime, "true_family": truth, "selected_family": selected, "count": count, "row_total": denominator, "row_proportion": count/denominator})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    configure_run_root(args.run_root)
    assert RAW_ROOT is not None and SUMMARY_ROOT is not None
    config = runner.load_config()
    plan_path = RESULT_ROOT / "metadata/formal_run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    reused_unaffected = bool(plan.get("prior_run_unaffected_rows_consumed", False))
    if reused_unaffected and plan.get(
        "historical_incorrect_affected_internal_rows_consumed"
    ) is not False:
        raise RuntimeError("Run plan permits unsupported internal-row reuse")
    reps = int(config["replications"])
    memberships = runner.panel_cells(config)
    physical: dict[str, pd.DataFrame] = {}
    for cell in runner.unique_cells(memberships):
        path = runner.cell_directory(cell) / "experiment1_raw_results.csv"
        if not path.exists():
            raise SystemExit(f"Missing formal raw cell: {path}")
        frame = pd.read_csv(path)
        runner.validate_cell(frame, cell, reps)
        if "estimated_alpha_matrix" not in frame:
            raise RuntimeError("Raw schema cannot independently reproduce alpha MAPE")
        physical[str(cell["shared_cell_id"])] = frame
    pairing = pairing_audit(physical, config)
    frames = [annotate_membership(physical[str(cell["shared_cell_id"])], cell) for cell in memberships]
    raw = pd.concat(frames, ignore_index=True)
    recomputed = raw.apply(recompute_metrics, axis=1, result_type="expand")
    recomputed.columns = ["directed_f1_recomputed", "alpha_mape_recomputed", "family_accuracy_recomputed"]
    raw = pd.concat([raw.reset_index(drop=True), recomputed], axis=1)
    if not np.allclose(raw["directed_f1"], raw["directed_f1_recomputed"], rtol=0, atol=1e-12):
        raise RuntimeError("Stored directed F1 differs from edge-set recomputation")
    stored_mape = pd.to_numeric(raw["alpha_mape_on_correct_edges_pct"], errors="coerce").to_numpy(float)
    new_mape = raw["alpha_mape_recomputed"].to_numpy(float)
    if not np.allclose(stored_mape, new_mape, rtol=0, atol=1e-10, equal_nan=True):
        raise RuntimeError("Stored alpha MAPE differs from coefficient recomputation")
    internal = raw["method"].isin(["LibraryDP", "LibraryGreedy"])
    if not np.allclose(raw.loc[internal, "family_accuracy"], raw.loc[internal, "family_accuracy_recomputed"], rtol=0, atol=1e-12):
        raise RuntimeError("Stored family accuracy differs from label recomputation")
    summaries = pd.concat([
        aggregate(raw, "directed_f1_recomputed"),
        aggregate(raw.loc[raw["method"].isin(["LibraryDP", "LibraryGreedy", "OracleDP"])], "alpha_mape_recomputed"),
        aggregate(raw.loc[internal], "family_accuracy_recomputed"),
    ], ignore_index=True)
    confusion = confusion_table(raw)
    atomic_csv(raw, RAW_ROOT / "mixed_family_all_sweeps_raw.csv")
    atomic_csv(summaries, SUMMARY_ROOT / "mixed_family_all_metrics_summary.csv")
    atomic_csv(confusion, SUMMARY_ROOT / "mixed_family_anchor_confusion.csv")
    report = {
        "status": "passed", "created_at": utc_now(), "R": reps,
        "raw_rows_with_panel_membership": len(raw), "unique_physical_cells": len(physical),
        "pairing": pairing,
        "historical_results_consumed": reused_unaffected,
        "prior_run_unaffected_rows_consumed": reused_unaffected,
        "historical_incorrect_affected_internal_rows_consumed": False,
        "directed_f1_recomputed": True, "alpha_mape_recomputed": True,
        "family_accuracy_recomputed": True,
    }
    atomic_json(report, SUMMARY_ROOT / "validation_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
