#!/usr/bin/env python3
"""Rebuild the six paper figures and NBA recovery table from committed results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from scripts.summarize_nba import summarize


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WORK = ROOT / "work" / "reproduce"
GENERATED = WORK / "paper_objects"
OUTPUTS = ROOT / "outputs"
MANIFEST = ROOT / "manifests" / "PAPER_OBJECTS.csv"


def run(interpreter: Path, program: Path) -> None:
    subprocess.run(
        [str(interpreter), "-B", str(program), "--run-root", str(WORK)],
        cwd=ROOT,
        check=True,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_sha256(path: Path) -> str:
    page = PdfReader(str(path)).pages[0]
    return hashlib.sha256(page.get_contents().get_data()).hexdigest()


def prepare_working_results() -> None:
    if WORK.exists():
        if WORK.resolve().parent != (ROOT / "work").resolve():
            raise RuntimeError(f"Refusing to remove unexpected path: {WORK}")
        shutil.rmtree(WORK)
    copies = (
        (RESULTS / "all_poisson" / "summaries", WORK / "all_poisson" / "summaries"),
        (RESULTS / "all_poisson" / "metadata", WORK / "all_poisson" / "metadata"),
        (RESULTS / "mixed_family" / "summaries", WORK / "mixed_family" / "summaries"),
        (RESULTS / "nba" / "summaries", WORK / "nba" / "summaries"),
    )
    for source, destination in copies:
        shutil.copytree(source, destination)


def verify_outputs() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 7:
        raise RuntimeError("Paper-object manifest must contain exactly seven records")
    for record in records:
        path = ROOT / record["path"]
        if record["kind"] == "figure":
            actual = stream_sha256(path)
            expected = record["content_stream_sha256"]
        else:
            actual = file_sha256(path)
            expected = record["sha256"]
        if actual != expected:
            raise RuntimeError(f"Integrity check failed for {record['paper_object']}: {path}")
        print(f"{record['paper_object']} = PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nba-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter from the pinned NBA environment",
    )
    args = parser.parse_args()

    prepare_working_results()
    try:
        run(Path(sys.executable), ROOT / "scripts" / "plot_all_poisson.py")
        run(Path(sys.executable), ROOT / "scripts" / "plot_mixed_family.py")
        run(
            args.nba_python,
            ROOT / "experiments" / "nba" / "scripts" / "plot_nba_figures.py",
        )

        detail = pd.read_csv(
            WORK / "nba" / "summaries" / "nba_all_methods_by_season.csv"
        )
        table = summarize(detail)
        GENERATED.mkdir(parents=True, exist_ok=True)
        table.to_csv(GENERATED / "nba_structural_recovery.csv", index=False)

        for name in (
            "figure1_all_poisson.pdf",
            "figure2_dag_recovery_f1.pdf",
            "figure3_coefficient_mape.pdf",
            "figure4_family_selection.pdf",
            "figure5_nba_reference_dag.pdf",
            "figure6_nba_season_estimates.pdf",
        ):
            destination = OUTPUTS / "figures" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(GENERATED / name, destination)
        table_destination = OUTPUTS / "tables" / "nba_structural_recovery.csv"
        table_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(GENERATED / "nba_structural_recovery.csv", table_destination)
        verify_outputs()
    finally:
        if WORK.exists():
            shutil.rmtree(WORK)


if __name__ == "__main__":
    main()
