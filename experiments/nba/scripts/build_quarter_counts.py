import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = Path(
    os.environ.get("PTSEM_NBA_RAW_DIR", REPO_ROOT / "data/nba/raw")
)

SEASONS = {
    "2015-16": 2015,
    "2016-17": 2016,
    "2017-18": 2017,
    "2018-19": 2018,
    "2019-20": 2019,
    "2020-21": 2020,
    "2021-22": 2021,
    "2022-23": 2022,
    "2023-24": 2023,
    "2024-25": 2024,
}

PBP_COLUMNS = [
    "GAME_ID",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "SCORE",
    "PLAYER1_TEAM_ID",
    "PLAYER2_TEAM_ID",
    "PLAYER3_TEAM_ID",
]

FOUL_EVENT = 6
FREE_THROW_EVENT = 3
OFFENSIVE_FOUL_ACTIONS = {4, 26}


def clean_team_id(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def clean_game_id(value):
    return str(int(value)).zfill(8)


def collect_game_teams(path, chunksize):
    teams_by_game = defaultdict(set)
    for chunk in pd.read_csv(path, usecols=PBP_COLUMNS, chunksize=chunksize, low_memory=False):
        for col in ["PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID", "PLAYER3_TEAM_ID"]:
            sub = chunk[["GAME_ID", col]].dropna()
            for game_id, team_id in sub.itertuples(index=False):
                team = clean_team_id(team_id)
                if team and team != "0":
                    teams_by_game[clean_game_id(game_id)].add(team)
    return teams_by_game


def init_counts(teams_by_game):
    counts = {}
    for game_id, teams in teams_by_game.items():
        for team in sorted(teams):
            for period in range(1, 5):
                counts[(game_id, period, team)] = {"FOUL": 0, "FTA": 0, "FTM": 0}
    return counts


def foul_beneficiary(row, mode):
    committed = clean_team_id(row.PLAYER1_TEAM_ID)
    drawn = clean_team_id(row.PLAYER2_TEAM_ID)
    if mode == "committed":
        return committed
    if drawn and drawn != "0" and drawn != committed:
        return drawn
    return None


def aggregate_file(path, season, out_path, foul_mode, chunksize):
    teams_by_game = collect_game_teams(path, chunksize)
    counts = init_counts(teams_by_game)
    stats = {
        "rows": 0,
        "games": len(teams_by_game),
        "foul_rows": 0,
        "excluded_offensive_fouls": 0,
        "counted_fouls": 0,
        "free_throw_rows": 0,
        "made_free_throws": 0,
    }

    dtypes = {
        "EVENTMSGTYPE": "Int64",
        "EVENTMSGACTIONTYPE": "Int64",
        "PERIOD": "Int64",
    }
    for chunk in pd.read_csv(path, usecols=PBP_COLUMNS, chunksize=chunksize, dtype=dtypes, low_memory=False):
        stats["rows"] += len(chunk)
        chunk = chunk[chunk["PERIOD"].isin([1, 2, 3, 4])]
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            game_id = clean_game_id(row.GAME_ID)
            period = int(row.PERIOD)
            event_type = int(row.EVENTMSGTYPE)

            if event_type == FREE_THROW_EVENT:
                team = clean_team_id(row.PLAYER1_TEAM_ID)
                if not team or team == "0":
                    continue
                key = (game_id, period, team)
                if key not in counts:
                    counts[key] = {"FOUL": 0, "FTA": 0, "FTM": 0}
                counts[key]["FTA"] += 1
                stats["free_throw_rows"] += 1
                if isinstance(row.SCORE, str) and row.SCORE.strip():
                    counts[key]["FTM"] += 1
                    stats["made_free_throws"] += 1

            elif event_type == FOUL_EVENT:
                stats["foul_rows"] += 1
                action = int(row.EVENTMSGACTIONTYPE) if not pd.isna(row.EVENTMSGACTIONTYPE) else -1
                if action in OFFENSIVE_FOUL_ACTIONS:
                    stats["excluded_offensive_fouls"] += 1
                    continue
                team = foul_beneficiary(row, foul_mode)
                if not team or team == "0":
                    continue
                key = (game_id, period, team)
                if key not in counts:
                    counts[key] = {"FOUL": 0, "FTA": 0, "FTM": 0}
                counts[key]["FOUL"] += 1
                stats["counted_fouls"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["season", "game_id", "period", "team_id", "FOUL", "FTA", "FTM"])
        for (game_id, period, team), values in sorted(counts.items()):
            writer.writerow([season, game_id, period, team, values["FOUL"], values["FTA"], values["FTM"]])
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--season", choices=list(SEASONS), help="Build only one season.")
    parser.add_argument("--foul-mode", choices=["drawn", "committed"], default="drawn")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    seasons = {args.season: SEASONS[args.season]} if args.season else SEASONS
    summary_rows = []
    for label, start_year in seasons.items():
        src = args.dataset_dir / f"nbastats_{start_year}.csv"
        out = args.out_dir / f"quarter_counts_{label}_{args.foul_mode}.csv"
        if not src.exists():
            raise FileNotFoundError(src)
        print(f"[{label}] aggregating {src.name} -> {out}", flush=True)
        stats = aggregate_file(src, label, out, args.foul_mode, args.chunksize)
        stats["season"] = label
        stats["source"] = str(src)
        stats["output"] = str(out)
        summary_rows.append(stats)
        print(
            f"  games={stats['games']} excluded_offensive_fouls={stats['excluded_offensive_fouls']} "
            f"counted_fouls={stats['counted_fouls']} "
            f"fta={stats['free_throw_rows']} ftm={stats['made_free_throws']}",
            flush=True,
        )

    summary_path = args.out_dir / "quarter_counts_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
