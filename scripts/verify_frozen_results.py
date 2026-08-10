#!/usr/bin/env python3
"""Verify artifact digests and the structural invariants of each experiment line."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEASONS = (
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_hashes() -> None:
    manifest = ROOT / "manifests/SHA256SUMS.csv"
    require(manifest.exists(), "Run scripts/build_manifests.py first")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) >= 90, "Publication manifest is unexpectedly small")
    for row in rows:
        path = ROOT / row["path"]
        require(path.is_file(), f"Missing frozen file: {row['path']}")
        require(path.stat().st_size == int(row["size_bytes"]), f"Size mismatch: {row['path']}")
        require(digest(path) == row["sha256"], f"SHA-256 mismatch: {row['path']}")
    print(f"PASS hashes ({len(rows)} files)")


CORRECTED_CORE = "70da3d96d4013270de44e0dce1653cc27e6e2ab6582fa4b124ac962e1ebdb020"
HISTORICAL_CORE = "d14c13f5984ad90d16b9444bed1d4e7f9ad361161b63e402614f093449a65fdb"
FAMILY_LIBRARY = ["Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli"]
N_GRID = [100, 200, 400, 800, 1600, 3200, 6400, 10000]


def verify_cores() -> None:
    """Each experiment line must keep the core recorded in its results."""
    require(digest(ROOT / "src/d.py") == CORRECTED_CORE,
            "Simulation core is not the corrected nb_exact_v2 build")
    require(digest(ROOT / "experiments/nba/core/d.py") == HISTORICAL_CORE,
            "NBA core is not the historical build recorded in data/nba/graph_json")
    for path in sorted((ROOT / "data/nba/graph_json").glob("*.json")):
        recorded = json.loads(path.read_text(encoding="utf-8"))["core_sha256"]
        require(recorded == HISTORICAL_CORE, f"{path.name} was fitted with a different core")
    print("PASS numerical cores (simulation nb_exact_v2, NBA historical)")


def verify_simulation() -> None:
    report = json.loads((ROOT / "results/mixed_family/summaries/validation_report.json").read_text())
    require(report["status"] == "passed", "Mixed-family validation did not pass")
    require(report["R"] == 100, "Mixed-family R mismatch")
    require(report["unique_physical_cells"] == 36, "Mixed-family cell count mismatch")
    require(report["pairing"] == {"checks": 2800, "failures": 0, "status": "passed"},
            "Mixed-family pairing audit mismatch")
    require(report["historical_results_consumed"] is False,
            "Mixed-family results must be a fresh run, not a repackaged snapshot")

    plan = json.loads((ROOT / "results/mixed_family/metadata/formal_run_plan.json").read_text())
    require(plan["framework"]["core_d_sha256"] == CORRECTED_CORE, "Mixed-family core mismatch")
    require(plan["framework"]["family_library"] == FAMILY_LIBRARY, "Family library mismatch")

    raw = pd.read_csv(ROOT / "results/mixed_family/raw/mixed_family_all_sweeps_raw.csv",
                      usecols=["regime", "sweep", "N", "rep", "method", "directed_f1"])
    require(len(raw) == 24000, f"Mixed-family row count is {len(raw)}, expected 24000")
    require(raw["rep"].between(0, 99).all(), "Mixed-family replication range failure")
    require(raw["directed_f1"].dropna().between(0, 1).all(), "Mixed-family F1 range failure")
    executed = sorted(raw.loc[raw["sweep"].eq("sample_size"), "N"].unique().astype(int))
    require(executed == N_GRID, f"Executed N grid {executed} does not match Table 2")
    print("PASS mixed-family 24,000 rows / 36 cells / Table 2 sweep grid")


def verify_all_poisson() -> None:
    report = json.loads((ROOT / "results/all_poisson/metadata/validation_report.json").read_text())
    require(report.get("status") == "passed", "All-Poisson validation did not pass")
    require(report["raw_rows"] == report["expected_raw_rows"] == 24000, "All-Poisson row count mismatch")
    require(report["method_setting_cells_incomplete"] == 0, "All-Poisson has incomplete cells")
    require(report["historical_results_consumed"] is False,
            "All-Poisson results must be a fresh run, not a repackaged snapshot")

    plan = json.loads((ROOT / "results/all_poisson/metadata/formal_run_plan.json").read_text())
    require(plan["framework"]["candidate_library"] == FAMILY_LIBRARY, "Family library mismatch")

    raw = pd.read_csv(ROOT / "results/all_poisson/raw/all_poisson_per_replication.csv",
                      usecols=["setting", "regime", "method", "replication", "F1"])
    require(len(raw) == 24000, f"All-Poisson row count is {len(raw)}, expected 24000")
    require(raw["F1"].between(0, 1).all(), "All-Poisson F1 range failure")
    summary = pd.read_csv(ROOT / "results/all_poisson/summaries/all_poisson_summary.csv")
    require(len(summary) == 240, "All-Poisson summary must have 240 cells")
    require(summary["R_success"].eq(100).all(), "All-Poisson cell completion failure")
    print("PASS all-Poisson 24,000 rows / 240 validated cells")


def assert_close(actual: float, expected: float, label: str) -> None:
    require(math.isclose(float(actual), expected, rel_tol=0, abs_tol=5e-4), f"{label}: {actual} != {expected}")


def verify_nba() -> None:
    total = 0
    for season in SEASONS:
        path = ROOT / f"data/nba/processed_team_quarter/expanded_team_quarter_{season}.csv"
        frame = pd.read_csv(path)
        require(set(frame["season"].astype(str)) == {season}, f"Season label mismatch: {season}")
        require(frame["period"].between(1, 4).all(), f"Overtime present: {season}")
        count_columns = ["FOUL", "FTA", "FTM", "PERS_FOUL_DRAWN", "LOOSE_BALL_FOUL_DRAWN"]
        require((frame[count_columns] >= 0).all().all(), f"Negative counts: {season}")
        require((frame["FTM"] <= frame["FTA"]).all(), f"FTM > FTA: {season}")
        total += len(frame)
    require(total == 95808, f"NBA total rows {total} != 95808")

    proposed = pd.read_csv(ROOT / "results/nba/nba_main_graph_recovery_summary.csv").iloc[0]
    assert_close(proposed["mean_skeleton_f1"], 1.000, "Proposed skeleton F1")
    assert_close(proposed["mean_directed_f1"], 0.975, "Proposed directed F1")
    require(proposed["exact_recovery"] == "9/10", "Proposed exact-recovery count mismatch")

    table = pd.read_csv(ROOT / "results/nba/tables/table2_nba_graph_recovery.csv")
    expected = {
        "PT-SEM (LibraryDP)": (1.000, 1.000, 1.000, 0.975, 0.975, 0.975),
        "Poisson DAG (ODS-style)": (0.400, 1.000, 0.571, 0.300, 0.750, 0.429),
        "PC (RCIT)": (0.619, 1.000, 0.764, 0.148, 0.200, 0.169),
        "PB-SCM (Cumulant)": (0.900, 0.500, 0.630, 0.750, 0.275, 0.393),
        "PB-SCM (PGF)": (0.719, 0.925, 0.806, 0.423, 0.450, 0.423),
    }
    columns = ("mean_skeleton_precision", "mean_skeleton_recall", "mean_skeleton_f1", "mean_directed_precision", "mean_directed_recall", "mean_directed_f1")
    for method, values in expected.items():
        row = table.loc[table["method"].eq(method)]
        require(len(row) == 1, f"Missing Table 3 method: {method}")
        for column, value in zip(columns, values):
            assert_close(row.iloc[0][column], value, f"Table 3 {method}/{column}")
    by_season = pd.read_csv(ROOT / "results/nba/nba_main_graph_recovery_by_season.csv")
    require(by_season["exact_recovery"].sum() == 9, "NBA exact directed recovery must be 9/10")
    require(set(by_season.loc[~by_season["exact_recovery"], "season"]) == {"2022-23"}, "Unexpected non-exact season")
    print("PASS NBA 95,808 rows, ten seasons, Figures 5-6 inputs, and Table 3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section",
                        choices=("hashes", "cores", "simulation", "all-poisson", "nba"))
    args = parser.parse_args()
    checks = {
        "hashes": verify_hashes,
        "cores": verify_cores,
        "simulation": verify_simulation,
        "all-poisson": verify_all_poisson,
        "nba": verify_nba,
    }
    selected = (args.section,) if args.section else tuple(checks)
    try:
        for name in selected:
            checks[name]()
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("ALL SELECTED CHECKS PASSED")


if __name__ == "__main__":
    main()
