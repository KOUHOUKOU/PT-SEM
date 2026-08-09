#!/usr/bin/env python3
"""Prepare a portable copy of the historical all-Poisson snapshot and optionally run it."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(destination: Path) -> Path:
    source = ROOT / "experiments/simulation/all_poisson_snapshot"
    for path in (source / "scripts").glob("*.py"):
        copy_file(path, destination / "scripts" / path.name)
    for path in (source / "frozen_methods").glob("*"):
        if path.is_file():
            copy_file(path, destination / "frozen_code/methods" / path.name)
    copy_file(
        ROOT / "experiments/simulation/legacy/setup_external_baselines.py",
        destination / "frozen_code/methods/setup_external_baselines.py",
    )
    copy_file(
        source / "frozen_configs/all_poisson/jmlr_all_poisson_formal_design_v2.json",
        destination / "frozen_configs/all_poisson/jmlr_all_poisson_formal_design_v2.json",
    )
    copy_file(
        ROOT / "data/all_poisson/raw/reused_restricted_sample_size.csv",
        destination / "new_results/all_poisson/raw/reused_restricted_sample_size.csv",
    )
    return destination / "scripts/run_all_poisson_missing_sweeps.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT / "scratch/all_poisson_rerun")
    parser.add_argument("--setup-external", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run after preparing; otherwise only print the command.")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the historical runner.")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    runner = prepare(workspace)
    if args.setup_external:
        subprocess.run(
            [sys.executable, str(workspace / "frozen_code/methods/setup_external_baselines.py")],
            cwd=workspace / "frozen_code/methods",
            check=True,
        )
    forwarded = list(args.runner_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    command = [sys.executable, str(runner), *forwarded]
    print("Prepared:", workspace)
    print("Command:", " ".join(command))
    if args.execute:
        subprocess.run(command, cwd=workspace, check=True)
    else:
        print("Preparation only. Add --execute to start the computation.")


if __name__ == "__main__":
    main()
