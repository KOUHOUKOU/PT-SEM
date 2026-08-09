"""Run ODS, PC-RCIT, and both PB-SCM baselines for a selected hypothesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_four_var_baseline_comparison as baseline
from scan_expanded_candidates_moment import HYPOTHESES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hypothesis", choices=sorted(HYPOTHESES))
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs/expanded_candidate_study/baselines"),
    )
    args = parser.parse_args()
    spec = HYPOTHESES[args.hypothesis]
    baseline.VARIABLES = list(spec["variables"])
    baseline.REFERENCE_EDGES = set(spec["edges"])
    is_team = args.hypothesis.endswith("_team")
    is_loose = args.hypothesis in {
        "loose_foul_4_team",
        "foul_leaves_5_team",
    }
    if is_team:
        input_dir = (
            Path("outputs/expanded_candidate_study/data_team_loose")
            if is_loose
            else Path("outputs/expanded_candidate_study/data_team")
        )
        pattern = "expanded_team_quarter_{season}.csv"
    else:
        input_dir = Path("outputs/expanded_candidate_study/data")
        pattern = "expanded_game_quarter_{season}.csv"
    sys.argv = [
        sys.argv[0],
        "--input-dir",
        str(input_dir),
        "--pattern",
        pattern,
        "--outputs-dir",
        str(args.outputs_root / args.hypothesis),
    ]
    baseline.main()


if __name__ == "__main__":
    main()
