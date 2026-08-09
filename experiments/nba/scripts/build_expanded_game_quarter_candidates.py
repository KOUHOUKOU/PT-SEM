"""Build a wide game-quarter count table for interpretable candidate scans."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from build_quarter_counts import DEFAULT_DATASET_DIR, SEASONS, clean_game_id


EVENT_COLUMNS = [
    "GAME_ID",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "PLAYER2_ID",
    "PLAYER3_ID",
]


def present(value: object) -> bool:
    if pd.isna(value):
        return False
    try:
        return int(float(value)) != 0
    except (TypeError, ValueError):
        return bool(str(value).strip())


def collect(path: Path, chunksize: int) -> dict[tuple[str, int], dict[str, int]]:
    counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for chunk in pd.read_csv(
        path, usecols=EVENT_COLUMNS, chunksize=chunksize, low_memory=False
    ):
        chunk = chunk[chunk["PERIOD"].isin([1, 2, 3, 4])]
        for row in chunk.itertuples(index=False):
            key = (clean_game_id(row.GAME_ID), int(row.PERIOD))
            event = int(row.EVENTMSGTYPE)
            action = (
                int(row.EVENTMSGACTIONTYPE)
                if not pd.isna(row.EVENTMSGACTIONTYPE)
                else -1
            )
            out = counts[key]
            if event in (1, 2):
                out["FGA"] += 1
            if event == 1:
                out["FGM"] += 1
                if present(row.PLAYER2_ID):
                    out["AST"] += 1
            elif event == 2:
                out["MISS_FG"] += 1
                if present(row.PLAYER3_ID):
                    out["BLK"] += 1
            elif event == 4:
                out["REB"] += 1
            elif event == 5:
                out["TOV"] += 1
                if present(row.PLAYER2_ID):
                    out["STL"] += 1
            elif event == 6:
                if action in {2, 29}:
                    out["SHOT_FOUL_DRAWN"] += 1
                elif action in {1, 27, 28}:
                    out["PERS_FOUL_DRAWN"] += 1
                elif action in {4, 26}:
                    out["OFF_FOUL_DRAWN"] += 1
            elif event == 7:
                out["VIOL"] += 1
            elif event == 8:
                out["SUB"] += 1
            elif event == 9:
                out["TIMEOUT"] += 1
            elif event == 10:
                out["JUMP_BALL"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/rebound_control_study/game-quarter"),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/expanded_candidate_study/data"),
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    candidate_names = [
        "FGA",
        "FGM",
        "MISS_FG",
        "REB",
        "TOV",
        "VIOL",
        "SUB",
        "SHOT_FOUL_DRAWN",
        "PERS_FOUL_DRAWN",
        "OFF_FOUL_DRAWN",
        "AST",
        "BLK",
        "STL",
        "TIMEOUT",
        "JUMP_BALL",
    ]
    for season, year in SEASONS.items():
        base_path = (
            args.base_dir
            / f"game-quarter_counts_{season}_drawn_missfg_reb.csv"
        )
        base = pd.read_csv(base_path)
        raw = args.dataset_dir / f"nbastats_{year}.csv"
        counts = collect(raw, args.chunksize)
        for name in candidate_names:
            base[name] = [
                counts[(clean_game_id(game), int(period))][name]
                for game, period in zip(base["game_id"], base["period"])
            ]
        base["MISS_FT"] = base["FTA"] - base["FTM"]
        if (base["MISS_FT"] < 0).any():
            raise ValueError(f"{season}: negative MISS_FT")
        output = args.outputs_dir / f"expanded_game_quarter_{season}.csv"
        base.to_csv(output, index=False)
        print(season, len(base), output, flush=True)


if __name__ == "__main__":
    main()
