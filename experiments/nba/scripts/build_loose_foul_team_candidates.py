"""Add loose-ball fouls to a separate copy of the team-quarter candidate data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_expanded_team_quarter_fouls import (
    SEASONS,
    loose_ball_counts,
)
from build_quarter_counts import clean_game_id, clean_team_id


SOURCE = Path("outputs/expanded_candidate_study/data_team")
OUTPUT = Path("outputs/expanded_candidate_study/data_team_loose")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        frame = pd.read_csv(SOURCE / f"expanded_team_quarter_{season}.csv")
        counts = loose_ball_counts(season)
        frame["LOOSE_BALL_FOUL_DRAWN"] = [
            counts[(clean_game_id(game), int(period), clean_team_id(team))]
            for game, period, team in zip(
                frame["game_id"], frame["period"], frame["team_id"]
            )
        ]
        path = OUTPUT / f"expanded_team_quarter_{season}.csv"
        frame.to_csv(path, index=False)
        print(season, len(frame), path)


if __name__ == "__main__":
    main()
