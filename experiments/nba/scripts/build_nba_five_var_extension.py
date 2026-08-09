"""Build the paper-final five-variable NBA team-quarter data without refitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_expanded_team_quarter_fouls import SEASONS, loose_ball_counts
from build_quarter_counts import DEFAULT_DATASET_DIR, clean_game_id, clean_team_id
from run_four_var_candidate_scan import collect_candidate_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/expanded_candidate_study/data_team_loose"),
    )
    parser.add_argument("--seasons", nargs="+", choices=SEASONS, default=list(SEASONS))
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for season in args.seasons:
        year = int(season[:4])
        raw = args.dataset_dir / f"nbastats_{year}.csv"
        base = args.processed_dir / f"quarter_counts_{season}_drawn.csv"
        frame = pd.read_csv(base)
        personal = collect_candidate_counts(raw, "PERS_FOUL_DRAWN", args.chunksize)
        loose = loose_ball_counts(season)
        keys = zip(frame["game_id"], frame["period"], frame["team_id"])
        normalized = [
            (clean_game_id(game), int(period), clean_team_id(team))
            for game, period, team in keys
        ]
        frame["PERS_FOUL_DRAWN"] = [personal[key] for key in normalized]
        frame["LOOSE_BALL_FOUL_DRAWN"] = [loose[key] for key in normalized]
        output = args.output_dir / f"expanded_team_quarter_{season}.csv"
        frame[
            [
                "season", "game_id", "period", "team_id",
                "FOUL", "FTA", "FTM", "PERS_FOUL_DRAWN",
                "LOOSE_BALL_FOUL_DRAWN",
            ]
        ].to_csv(output, index=False)
        print(season, len(frame), output, flush=True)


if __name__ == "__main__":
    main()
