#!/usr/bin/env python3
"""Normalize, validate, and summarize the formal JMLR All-Poisson results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SNAPSHOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    SNAPSHOT
    / "frozen_configs/all_poisson/jmlr_all_poisson_formal_design_v2.json"
)
RESULT_ROOT = SNAPSHOT / "new_results/all_poisson"
REUSED_PATH = RESULT_ROOT / "raw/reused_restricted_sample_size.csv"
RAW_OUTPUT = RESULT_ROOT / "raw/all_poisson_per_replication.csv"
SUMMARY_OUTPUT = RESULT_ROOT / "summaries/all_poisson_summary.csv"
VALIDATION_OUTPUT = RESULT_ROOT / "metadata/validation_report.json"
SEED_PATH = RESULT_ROOT / "metadata/seeds.csv"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_edges(value: object) -> set[tuple[int, int]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    output: set[tuple[int, int]] = set()
    for part in text.replace(",", ";").split(";"):
        if not part.strip():
            continue
        source, target = part.strip().split("->", 1)
        output.add((int(source), int(target)))
    return output


def serialize_adjacency(edges: set[tuple[int, int]], dimension: int) -> str:
    matrix = np.zeros((dimension, dimension), dtype=int)
    for source, target in edges:
        matrix[source, target] = 1
    return json.dumps(matrix.tolist(), separators=(",", ":"))


def directed_f1(truth: set[tuple[int, int]], estimate: set[tuple[int, int]]) -> float:
    if not truth and not estimate:
        return 1.0
    if not truth or not estimate:
        return 0.0
    true_positive = len(truth & estimate)
    precision = true_positive / len(estimate)
    recall = true_positive / len(truth)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalize_reused() -> pd.DataFrame:
    if not REUSED_PATH.exists():
        raise FileNotFoundError(f"Missing reused restricted sample-size raw: {REUSED_PATH}")
    frame = pd.read_csv(REUSED_PATH)
    frame["raw_file"] = str(REUSED_PATH.relative_to(SNAPSHOT)).replace("\\", "/")
    frame["dag_generation_mode"] = "exact_edges"
    frame["realized_avg_indegree"] = frame["n_true_edges"] / frame["d"]
    frame["parallel_workers"] = np.nan
    frame["selected_families"] = ""
    frame["family_accuracy"] = np.nan
    # Preserve seeds losslessly across the historical int64 and formal uint64
    # sources.  Concatenating them as numeric columns would coerce to float64.
    frame["seed"] = frame["seed"].map(lambda value: str(int(value)))
    frame["method_seed"] = frame["method_seed"].map(lambda value: str(int(value)))
    return frame


def normalize_formal_cell(log_path: Path, display_names: Mapping[str, str]) -> pd.DataFrame:
    metadata = json.loads(log_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "success":
        raise RuntimeError(f"Formal cell is not successful: {log_path}")
    raw_path = Path(metadata["output_path"]) / "experiment1_raw_results.csv"
    source = pd.read_csv(raw_path)
    mapped = source["method"].map(display_names)
    if mapped.isna().any():
        raise RuntimeError(f"Unmapped formal method in {raw_path}")
    truth = source["true_edges"].map(parse_edges)
    estimates = source["estimated_directed_edges"].map(parse_edges)
    output = pd.DataFrame(
        {
            "setting": "all_poisson",
            "regime": metadata["regime"],
            "sweep": metadata["sweep"],
            "sweep_value": metadata["sweep_value"],
            "d": source["d"],
            "N": source["N"],
            "average_in_degree": source["avg_indegree"],
            "max_parents": source["max_parents"],
            "alpha_low": source["alpha_low"],
            "alpha_high": source["alpha_high"],
            "replication": source["rep"],
            "seed": source["seed"].map(lambda value: str(int(value))),
            "method": mapped,
            "method_internal": source["method"],
            "method_seed": source["method_seed"].map(lambda value: str(int(value))),
            "start_time_utc": metadata["started_at"],
            "end_time_utc": metadata["completed_at"],
            "status": "success",
            "error_type": "",
            "error_message": "",
            "runtime_sec": source["runtime_sec"],
            "runtime_local_score_sec": source["local_score_runtime_sec"],
            "runtime_search_sec": source["search_runtime_sec"],
            "n_local_scores_evaluated": source["n_local_scores_evaluated"],
            "true_exogenous_families": source["true_families"].str.replace(",", ";", regex=False),
            "true_exogenous_params": source["true_exogenous_params"],
            "true_alpha_matrix": source["true_alpha_matrix"],
            "true_directed_edge_set": source["true_edges"],
            "estimated_directed_edge_set": source["estimated_directed_edges"],
            "estimated_undirected_edge_set": source["estimated_undirected_edges"],
            "true_adjacency": [serialize_adjacency(value, int(d)) for value, d in zip(truth, source["d"])],
            "estimated_adjacency": [serialize_adjacency(value, int(d)) for value, d in zip(estimates, source["d"])],
            "F1": source["directed_f1"],
            "skeleton_F1": source["skeleton_f1"],
            "exact_recovery": source["exact_dag"],
            "n_true_edges": source["n_true_edges"],
            "n_estimated_directed": source["n_est_directed_edges"],
            "n_estimated_undirected": source["n_est_undirected_edges"],
            "source": "formal_M_setting_framework_v2",
            "raw_file": str(raw_path.relative_to(SNAPSHOT)).replace("\\", "/"),
            "dag_generation_mode": source["dag_generation_mode"],
            "realized_avg_indegree": source["realized_avg_indegree"],
            "parallel_workers": source["parallel_workers"],
            "selected_families": source["selected_families"],
            "family_accuracy": source["family_accuracy"],
        }
    )
    return output


def load_raw(config: Mapping[str, Any]) -> tuple[pd.DataFrame, int, list[Path]]:
    frames = [normalize_reused()]
    logs = sorted((RESULT_ROOT / "logs/formal_cells").glob("**/*.json"))
    for path in logs:
        frames.append(normalize_formal_cell(path, config["display_names"]))
    raw = pd.concat(frames, ignore_index=True, sort=False)
    key = ["regime", "sweep", "sweep_value", "replication", "method"]
    raw["sweep_value"] = pd.to_numeric(raw["sweep_value"])
    raw = raw.sort_values(key, kind="stable").reset_index(drop=True)
    duplicate_rows = int(raw.duplicated(key, keep=False).sum())
    return raw, duplicate_rows, logs


def expected_cells(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime, regime_config in config["regimes"].items():
        for sweep, sweep_config in config["sweeps"].items():
            for value in sweep_config["values"]:
                for internal in regime_config["methods"]:
                    rows.append(
                        {
                            "regime": regime,
                            "sweep": sweep,
                            "sweep_value": float(value),
                            "method": config["display_names"][internal],
                            "expected_R": int(config["replications"]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["regime", "sweep", "sweep_value", "method"]
    for key, group in raw.groupby(keys, sort=True):
        values = pd.to_numeric(
            group.loc[group["status"].eq("success"), "F1"], errors="coerce"
        ).dropna().to_numpy(float)
        mean = float(values.mean()) if len(values) else math.nan
        sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        se = sd / math.sqrt(len(values)) if len(values) > 1 else math.nan
        rows.append(
            {
                "regime": key[0],
                "sweep": key[1],
                "sweep_value": float(key[2]),
                "method": key[3],
                "R_total": int(group["replication"].nunique()),
                "R_success": int(len(values)),
                "R_failed": int(group["replication"].nunique() - len(values)),
                "mean_F1": mean,
                "sd_F1": sd,
                "se_F1": se,
                "ci95_low": mean - 1.96 * se,
                "ci95_high": mean + 1.96 * se,
                "plot_ci95_low": max(0.0, mean - 1.96 * se),
                "plot_ci95_high": min(1.0, mean + 1.96 * se),
                "mean_runtime_sec": float(pd.to_numeric(group["runtime_sec"], errors="coerce").mean()),
            }
        )
    actual = pd.DataFrame(rows)
    expected = expected_cells(config)
    output = expected.merge(
        actual,
        on=["regime", "sweep", "sweep_value", "method"],
        how="left",
        validate="one_to_one",
    )
    output.insert(0, "setting", "all_poisson")
    return output


def validate_against_original_figure4(summary: pd.DataFrame) -> dict[str, object]:
    original = pd.read_csv(
        SNAPSHOT
        / "frozen_inputs/summary_results/final_paper_figures/main_figures/fig4_all_poisson_library_cost_final_plotdata.csv"
    )
    original = original[original["panel"].eq("main_directed_f1")].copy()
    original["method"] = original["method"].map(
        {
            "Fixed-Poisson DP": "Poisson-only DP",
            "LibraryDP": "Proposed DP-BIC",
            "LibraryGreedy": "Greedy-BIC",
            "ODS": "ODS",
            "PC-RCIT": "PC-RCIT",
        }
    )
    current = summary[
        summary["regime"].eq("restricted")
        & summary["sweep"].eq("sample_size")
        & summary["method"].isin(original["method"])
    ]
    merged = original.merge(
        current,
        left_on=["x_value", "method"],
        right_on=["sweep_value", "method"],
        validate="one_to_one",
    )
    comparisons = {
        "mean": ("mean", "mean_F1"),
        "sd": ("sd", "sd_F1"),
        "se": ("se", "se_F1"),
        "ci95_low": ("ci95_low_x", "ci95_low_y"),
        "ci95_high": ("ci95_high_x", "ci95_high_y"),
    }
    differences = {
        name: float(np.nanmax(np.abs(merged[left] - merged[right])))
        for name, (left, right) in comparisons.items()
    }
    return {
        "passed": len(merged) == 40 and all(value <= 1e-12 for value in differences.values()),
        "matched_rows": int(len(merged)),
        "expected_rows": 40,
        "max_absolute_differences": differences,
    }


def standardized_alpha(serialized: str, low: float, high: float) -> np.ndarray:
    matrix = np.asarray(json.loads(serialized), dtype=float)
    return np.where(matrix > 0.0, (matrix - low) / (high - low), 0.0)


def validation_report(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    duplicate_rows: int,
    formal_logs: list[Path],
) -> dict[str, object]:
    failures: list[str] = []
    warnings: list[str] = []
    dataset_key = ["regime", "sweep", "sweep_value", "replication"]
    raw_key = dataset_key + ["method"]
    expected = expected_cells(config)
    incomplete = summary[
        summary["R_success"].fillna(0).astype(int).ne(summary["expected_R"].astype(int))
    ]
    if len(incomplete):
        failures.append(f"{len(incomplete)} method/setting cells do not have exactly R=100 successes")
    if duplicate_rows:
        failures.append(f"{duplicate_rows} duplicate raw-key rows")
    failed_rows = raw[raw["status"].ne("success")]
    if len(failed_rows):
        failures.append(f"{len(failed_rows)} raw method rows are not successful")
    nan_f1 = int(pd.to_numeric(raw["F1"], errors="coerce").isna().sum())
    if nan_f1:
        failures.append(f"{nan_f1} raw F1 values are NaN/non-numeric")

    seeds = pd.read_csv(SEED_PATH, dtype={"data_seed": "string"})
    seed_coordinate = ["regime", "sweep", "sweep_value", "replication"]
    duplicate_seed_coordinates = int(seeds.duplicated(seed_coordinate, keep=False).sum())
    if duplicate_seed_coordinates:
        failures.append(f"{duplicate_seed_coordinates} duplicate seed coordinates")
    expected_dataset_rows = (
        len(config["regimes"])
        * sum(len(value["values"]) for value in config["sweeps"].values())
        * int(config["replications"])
    )
    if len(seeds) != expected_dataset_rows:
        failures.append(f"seed table has {len(seeds)}/{expected_dataset_rows} rows")
    raw_datasets = raw.drop_duplicates(dataset_key)
    seed_join = raw_datasets[dataset_key + ["seed"]].merge(
        seeds[seed_coordinate + ["data_seed"]],
        on=seed_coordinate,
        validate="one_to_one",
    )
    seed_mismatches = int(
        (seed_join["seed"].astype(str) != seed_join["data_seed"].astype(str)).sum()
    )
    if seed_mismatches:
        failures.append(f"{seed_mismatches} raw dataset seeds differ from seeds.csv")
    within_regime_seed_collisions = 0
    for _, group in seeds.groupby("regime"):
        within_regime_seed_collisions += int(group.duplicated("data_seed", keep=False).sum())
    if within_regime_seed_collisions:
        failures.append(f"{within_regime_seed_collisions} unintended within-regime seed collisions")

    if int(raw.groupby(dataset_key)["seed"].nunique(dropna=False).max()) != 1:
        failures.append("Methods within a replication do not share one data seed")
    truth_columns = [
        "true_exogenous_families",
        "true_exogenous_params",
        "true_alpha_matrix",
        "true_directed_edge_set",
    ]
    truth_nunique = {
        column: int(raw.groupby(dataset_key)[column].nunique(dropna=False).max())
        for column in truth_columns
    }
    if any(value != 1 for value in truth_nunique.values()):
        failures.append("Methods within a replication do not share identical truth")

    method_failures = 0
    expected_methods = {
        regime: {config["display_names"][name] for name in details["methods"]}
        for regime, details in config["regimes"].items()
    }
    for key, group in raw.groupby(dataset_key, sort=False):
        if set(group["method"].astype(str)) != expected_methods[str(key[0])]:
            method_failures += 1
    if method_failures:
        failures.append(f"{method_failures} replications have an incorrect method set")
    pb_extended = int(
        raw[
            raw["regime"].eq("extended")
            & raw["method"].isin(["PB-SCM", "PB-SCM-PGF"])
        ].shape[0]
    )
    if pb_extended:
        failures.append("PB-SCM rows exist in extended")
    if raw["method"].astype(str).str.contains("Oracle", case=False).any():
        failures.append("Oracle DP appears as a duplicate display method")

    family_failures = lambda_failures = alpha_failures = edge_rule_failures = 0
    for row in raw_datasets.itertuples(index=False):
        families = str(row.true_exogenous_families).replace(",", ";").split(";")
        if len(families) != int(row.d) or set(families) != {"Poisson"}:
            family_failures += 1
        params = json.loads(str(row.true_exogenous_params))
        if len(params) != int(row.d) or any(
            set(item) != {"lam"} or not 2.0 <= float(item["lam"]) <= 10.0
            for item in params
        ):
            lambda_failures += 1
        alpha = np.asarray(json.loads(str(row.true_alpha_matrix)), dtype=float)
        nonzero = alpha[alpha > 0.0]
        if len(nonzero) and (
            nonzero.min() < float(row.alpha_low) - 1e-12
            or nonzero.max() > float(row.alpha_high) + 1e-12
        ):
            alpha_failures += 1
        if len(nonzero) != round(int(row.d) * float(row.average_in_degree)):
            edge_rule_failures += 1
    for count, message in (
        (family_failures, "datasets are not entirely Poisson"),
        (lambda_failures, "datasets violate lambda Uniform(2,10) support"),
        (alpha_failures, "datasets violate their coefficient regime"),
        (edge_rule_failures, "datasets violate exact-edge construction"),
    ):
        if count:
            failures.append(f"{count} {message}")

    f1_mismatches = 0
    maximum_f1_difference = 0.0
    for row in raw.itertuples(index=False):
        recomputed = directed_f1(
            parse_edges(row.true_directed_edge_set),
            parse_edges(row.estimated_directed_edge_set),
        )
        difference = abs(recomputed - float(row.F1))
        maximum_f1_difference = max(maximum_f1_difference, difference)
        f1_mismatches += difference > 1e-12
    if f1_mismatches:
        failures.append(f"{f1_mismatches} F1 values are not directed-edge F1")

    paired_failures = 0
    paired_checks = 0
    for sweep in ("dimension", "average_in_degree"):
        restricted = raw_datasets[
            raw_datasets["regime"].eq("restricted") & raw_datasets["sweep"].eq(sweep)
        ]
        extended = raw_datasets[
            raw_datasets["regime"].eq("extended") & raw_datasets["sweep"].eq(sweep)
        ]
        paired = restricted.merge(
            extended,
            on=["sweep", "sweep_value", "replication", "d", "N", "average_in_degree"],
            suffixes=("_restricted", "_extended"),
            validate="one_to_one",
        )
        for row in paired.itertuples(index=False):
            paired_checks += 1
            same = (
                int(row.seed_restricted) == int(row.seed_extended)
                and row.true_directed_edge_set_restricted == row.true_directed_edge_set_extended
                and row.true_exogenous_params_restricted == row.true_exogenous_params_extended
            )
            left = standardized_alpha(row.true_alpha_matrix_restricted, 0.15, 0.85)
            right = standardized_alpha(row.true_alpha_matrix_extended, 0.2, 2.0)
            if not same or not np.allclose(left, right, rtol=0.0, atol=2e-15):
                paired_failures += 1
    if paired_failures:
        failures.append(f"{paired_failures}/{paired_checks} cross-regime pairs are inconsistent")

    formal_log_statuses = [
        json.loads(path.read_text(encoding="utf-8")).get("status") for path in formal_logs
    ]
    if len(formal_logs) != 32 or set(formal_log_statuses) != {"success"}:
        failures.append("Formal cell logs are not exactly 32 successful cells")
    original_match = validate_against_original_figure4(summary)
    if not original_match["passed"]:
        failures.append("Reused restricted sample-size summary differs from original Figure 4")

    trend_rows: list[dict[str, object]] = []
    trend = summary[
        summary["sweep"].eq("sample_size")
        & summary["method"].isin(["Proposed DP-BIC", "Poisson-only DP"])
    ].sort_values("sweep_value")
    for (regime, method), group in trend.groupby(["regime", "method"]):
        first, last = group.iloc[0], group.iloc[-1]
        reasonable = float(last.mean_F1) >= float(first.mean_F1) - 0.05
        if not reasonable:
            warnings.append(f"Non-improving sample-size trend: {regime}/{method}")
        trend_rows.append(
            {
                "regime": regime,
                "method": method,
                "first_N": first.sweep_value,
                "first_mean_F1": first.mean_F1,
                "last_N": last.sweep_value,
                "last_mean_F1": last.mean_F1,
                "reasonable": reasonable,
            }
        )

    expected_raw_rows = int(expected["expected_R"].sum())
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "warnings": warnings,
        "raw_rows": int(len(raw)),
        "expected_raw_rows": expected_raw_rows,
        "dataset_replications_present": int(raw[dataset_key].drop_duplicates().shape[0]),
        "expected_dataset_replications": expected_dataset_rows,
        "method_setting_cells_expected": int(len(expected)),
        "method_setting_cells_incomplete": int(len(incomplete)),
        "formal_cell_logs": len(formal_logs),
        "formal_cell_status_counts": pd.Series(formal_log_statuses).value_counts().to_dict(),
        "failed_raw_rows": int(len(failed_rows)),
        "nan_F1_rows": nan_f1,
        "duplicate_raw_rows": duplicate_rows,
        "seed_rows": int(len(seeds)),
        "duplicate_seed_coordinates": duplicate_seed_coordinates,
        "within_regime_seed_collisions": within_regime_seed_collisions,
        "seed_mismatches_against_raw": seed_mismatches,
        "truth_nunique_max_within_replication": truth_nunique,
        "method_set_failures": method_failures,
        "PB_rows_in_extended": pb_extended,
        "family_failures": family_failures,
        "lambda_support_failures": lambda_failures,
        "coefficient_range_failures": alpha_failures,
        "exact_edge_rule_failures": edge_rule_failures,
        "directed_F1_mismatches": f1_mismatches,
        "max_directed_F1_absolute_difference": maximum_f1_difference,
        "cross_regime_pair_checks": paired_checks,
        "cross_regime_pair_failures": paired_failures,
        "original_figure4_sample_size_match": original_match,
        "large_sample_trend_check": trend_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw, duplicate_rows, formal_logs = load_raw(config)
    summary = summarize(raw, config)
    report = validation_report(raw, summary, config, duplicate_rows, formal_logs)
    atomic_csv(raw, RAW_OUTPUT)
    atomic_csv(summary, SUMMARY_OUTPUT)
    atomic_json(report, VALIDATION_OUTPUT)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"Raw: {RAW_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Validation: {VALIDATION_OUTPUT}")
    if report["status"] != "passed" and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
