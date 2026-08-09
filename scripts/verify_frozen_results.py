#!/usr/bin/env python3
"""Verify hashes and paper-facing invariants of the frozen repository."""

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


def verify_simulation() -> None:
    manifest = json.loads((ROOT / "data/simulation/summaries/final_suite_manifest.json").read_text())
    config = manifest["configuration"]
    require(manifest["status"] == "complete", "Mixed-family suite is not complete")
    require(config["R"] == 100 and config["seed"] == 20260622, "Formal R/seed mismatch")
    require(config["candidate_library"] == ["Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli"], "Family library mismatch")
    require(config["N_values"] == [100, 200, 400, 800, 1600, 2400, 3200, 6400, 10000], "Executed N grid mismatch")
    require(manifest["reproducibility_audit"] == {"status": "passed", "pairing_checks": 2900, "failures": 0}, "Pairing audit mismatch")
    for name in ("experiment1_d_sweep_raw.csv", "experiment2_N_sweep_raw.csv", "experiment3_kin_sweep_raw.csv"):
        frame = pd.read_csv(ROOT / "data/simulation/raw" / name, usecols=["alpha_regime", "rep", "method", "seed", "directed_f1"])
        require(frame["rep"].between(0, 99).all(), f"Replication range failure in {name}")
        require(frame["directed_f1"].dropna().between(0, 1).all(), f"F1 range failure in {name}")
    print("PASS mixed-family formal manifest and consolidated raw results")


def verify_all_poisson() -> None:
    report = json.loads((ROOT / "data/all_poisson/metadata/validation_report.json").read_text())
    require(report.get("status") == "passed", "All-Poisson validation did not pass")
    raw = pd.read_csv(ROOT / "data/all_poisson/raw/all_poisson_per_replication.csv", usecols=["setting", "regime", "method", "replication", "F1"])
    require(len(raw) == 24000, f"All-Poisson row count is {len(raw)}, expected 24000")
    require(raw["F1"].between(0, 1).all(), "All-Poisson F1 range failure")
    summary = pd.read_csv(ROOT / "data/all_poisson/summaries/all_poisson_summary.csv")
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
    print("PASS NBA 95,808 rows, ten seasons, Figures 5-6 inputs, and manuscript Table 3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("hashes", "simulation", "all-poisson", "nba"))
    args = parser.parse_args()
    checks = {
        "hashes": verify_hashes,
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
