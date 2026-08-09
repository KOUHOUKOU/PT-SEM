"""Combine completed optimized candidate evaluations into one ranking."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/expanded_candidate_study/optimization"),
    )
    args = parser.parse_args()
    summaries = []
    details = []
    for path in sorted(args.root.glob("*/summary.csv")):
        summaries.append(pd.read_csv(path))
    for path in sorted(args.root.glob("*/evaluation.csv")):
        details.append(pd.read_csv(path))
    if not summaries:
        raise RuntimeError("no completed optimization summaries")
    summary = pd.concat(summaries, ignore_index=True).sort_values(
        ["exact_count", "mean_directed_f1", "mean_skeleton_f1"],
        ascending=False,
    )
    detail = pd.concat(details, ignore_index=True)
    summary.to_csv(args.root / "candidate_ranking.csv", index=False)
    detail.to_csv(args.root / "candidate_details.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
