"""Integrity tests for the shipped artifacts.

These run without the pinned scientific environments: they check committed
bytes and structural invariants, not refitted numbers.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from scripts import verify_frozen_results as verify


ROOT = Path(__file__).resolve().parents[1]


class FrozenResultTests(unittest.TestCase):
    def test_manifest_hashes(self) -> None:
        verify.verify_hashes()

    def test_numerical_cores(self) -> None:
        verify.verify_cores()

    def test_mixed_family(self) -> None:
        verify.verify_simulation()

    def test_all_poisson(self) -> None:
        verify.verify_all_poisson()

    def test_nba(self) -> None:
        verify.verify_nba()


class PaperObjectTests(unittest.TestCase):
    def test_every_paper_object_verifies(self) -> None:
        """Each delivered object must match its expected digest."""
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_paper_objects.py"), "--check-only"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("ALL PAPER OBJECTS VERIFIED", completed.stdout)


class HygieneTests(unittest.TestCase):
    def test_outputs_contains_only_paper_objects(self) -> None:
        """outputs/ carries the paper's objects and the generated report."""
        delivered = sorted(
            path.relative_to(ROOT / "outputs").as_posix()
            for path in (ROOT / "outputs").rglob("*")
            if path.is_file()
        )
        self.assertEqual(delivered, [
            "REPRODUCTION_REPORT.md",
            "figures/figure1_all_poisson.pdf",
            "figures/figure2_dag_recovery_f1.pdf",
            "figures/figure3_coefficient_mape.pdf",
            "figures/figure4_family_selection.pdf",
            "figures/figure5_nba_reference_dag.pdf",
            "figures/figure6_nba_season_estimates.pdf",
            "tables/table3_nba_structural_recovery.csv",
        ])

    def test_the_two_environments_differ_only_in_numpy(self) -> None:
        def pins(name: str) -> dict[str, str]:
            out = {}
            for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "==" in line:
                    key, value = line.split("==", 1)
                    out[key.strip()] = value.strip()
            return out

        simulation = pins("requirements-simulation.txt")
        nba = pins("requirements-nba.txt")
        self.assertEqual(set(simulation), set(nba))
        differing = {k for k in simulation if simulation[k] != nba[k]}
        self.assertEqual(differing, {"numpy"})


if __name__ == "__main__":
    unittest.main()
