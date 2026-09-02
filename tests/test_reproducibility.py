"""Integrity and release-layout tests for the public package."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "PAPER_OBJECTS.csv"
SIMULATION_CORE_SHA256 = (
    "e61e9c58477f9ffc50552d1f538f4507756a464edcf7f5cfd16e79fe6d78df12"
)
NBA_CORE_SHA256 = (
    "d65ee02ecdf78fbc634353f88b42aab71290d1ff3fdedfb2ad9119db450cfa3f"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_sha256(path: Path) -> str:
    page = PdfReader(str(path)).pages[0]
    return hashlib.sha256(page.get_contents().get_data()).hexdigest()


def pins(name: str) -> dict[str, str]:
    result = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            package, version = line.split("==", 1)
            result[package.strip()] = version.strip()
    return result


class ReleaseIntegrityTests(unittest.TestCase):
    def test_exact_scientific_cores(self) -> None:
        self.assertEqual(sha256(ROOT / "src/simulation_core.py"), SIMULATION_CORE_SHA256)
        self.assertEqual(sha256(ROOT / "src/nba_core.py"), NBA_CORE_SHA256)

    def test_seven_paper_objects(self) -> None:
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 7)
        self.assertEqual(
            {record["path"] for record in records},
            {
                "outputs/figures/figure1_all_poisson.pdf",
                "outputs/figures/figure2_dag_recovery_f1.pdf",
                "outputs/figures/figure3_coefficient_mape.pdf",
                "outputs/figures/figure4_family_selection.pdf",
                "outputs/figures/figure5_nba_reference_dag.pdf",
                "outputs/figures/figure6_nba_season_estimates.pdf",
                "outputs/tables/nba_structural_recovery.csv",
            },
        )
        for record in records:
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), path)
            if record["kind"] == "figure":
                self.assertEqual(
                    stream_sha256(path), record["content_stream_sha256"], path
                )
            else:
                self.assertEqual(sha256(path), record["sha256"], path)

    def test_outputs_contain_only_the_seven_objects(self) -> None:
        delivered = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "outputs").rglob("*")
            if path.is_file()
        }
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            expected = {record["path"] for record in csv.DictReader(handle)}
        self.assertEqual(delivered, expected)

    def test_nba_inputs_match_manifest(self) -> None:
        expected = json.loads(
            (ROOT / "results/nba/input_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(expected), 10)
        for season, digest in expected.items():
            path = (
                ROOT
                / "data/nba/processed_team_quarter"
                / f"expanded_team_quarter_{season}.csv"
            )
            self.assertEqual(sha256(path), digest, path)

    def test_nba_proposed_result_is_complete_in_every_season(self) -> None:
        detail = pd.read_csv(
            ROOT / "results/nba/summaries/nba_all_methods_by_season.csv"
        )
        proposed = detail[detail["method"] == "PT-SEM (full-MLE DP-BIC)"]
        self.assertEqual(len(proposed), 10)
        self.assertTrue(proposed["exact_recovery"].all())
        for column in (
            "skeleton_precision",
            "skeleton_recall",
            "skeleton_f1",
            "directed_precision",
            "directed_recall",
            "directed_f1",
        ):
            self.assertTrue(proposed[column].eq(1.0).all(), column)

    def test_environment_profiles_differ_only_in_numpy(self) -> None:
        simulation = pins("requirements-simulation.txt")
        nba = pins("requirements-nba.txt")
        self.assertEqual(set(simulation), set(nba))
        self.assertEqual(
            {name for name in simulation if simulation[name] != nba[name]},
            {"numpy"},
        )


if __name__ == "__main__":
    unittest.main()
