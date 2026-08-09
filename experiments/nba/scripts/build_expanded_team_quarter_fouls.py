"""Merge existing team-quarter foul-component candidate count files."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import pandas as pd

from build_quarter_counts import DEFAULT_DATASET_DIR, SEASONS as YEAR_MAP, clean_game_id, clean_team_id


SEASONS = (
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
)
ROOT = Path("outputs/four_var_candidate_scan")
OUT = Path("outputs/expanded_candidate_study/data_team")


def loose_ball_counts(season: str) -> dict[tuple[str, int, str], int]:
    path = DEFAULT_DATASET_DIR / f"nbastats_{YEAR_MAP[season]}.csv"
    counts: dict[tuple[str, int, str], int] = defaultdict(int)
    columns = [
        "GAME_ID",
        "EVENTMSGTYPE",
        "EVENTMSGACTIONTYPE",
        "PERIOD",
        "PLAYER2_TEAM_ID",
    ]
    for chunk in pd.read_csv(
        path, usecols=columns, chunksize=100_000, low_memory=False
    ):
        chunk = chunk[
            chunk["PERIOD"].isin([1, 2, 3, 4])
            & chunk["EVENTMSGTYPE"].eq(6)
            & chunk["EVENTMSGACTIONTYPE"].eq(3)
        ]
        for row in chunk.itertuples(index=False):
            team = clean_team_id(row.PLAYER2_TEAM_ID)
            if team and team != "0":
                counts[(clean_game_id(row.GAME_ID), int(row.PERIOD), team)] += 1
    return counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["season", "game_id", "period", "team_id"]
    for season in SEASONS:
        shot = pd.read_csv(
            ROOT
            / "SHOT_FOUL_DRAWN"
            / f"quarter_counts_{season}_drawn_SHOT_FOUL_DRAWN.csv"
        )
        personal = pd.read_csv(
            ROOT
            / "PERS_FOUL_DRAWN"
            / f"quarter_counts_{season}_drawn_PERS_FOUL_DRAWN.csv"
        )
        frame = shot.merge(
            personal[keys + ["PERS_FOUL_DRAWN"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        loose = loose_ball_counts(season)
        frame["LOOSE_BALL_FOUL_DRAWN"] = [
            loose[(clean_game_id(game), int(period), clean_team_id(team))]
            for game, period, team in zip(
                frame["game_id"], frame["period"], frame["team_id"]
            )
        ]
        frame["MISS_FT"] = frame["FTA"] - frame["FTM"]
        output = OUT / f"expanded_team_quarter_{season}.csv"
        frame.to_csv(output, index=False)
        print(season, len(frame), output)


if __name__ == "__main__":
    main()
