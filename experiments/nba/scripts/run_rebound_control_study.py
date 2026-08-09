import argparse
import ast
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import ptsem_three_node as pt
from build_quarter_counts import DEFAULT_DATASET_DIR, SEASONS, clean_game_id, clean_team_id


VARIABLES = ["FOUL", "FTA", "FTM", "MISS_FG", "REB"]
MISS_FG_EVENT = 2
REB_EVENT = 4


def collect_event_counts(path, event_type, chunksize):
    counts = defaultdict(int)
    columns = ["GAME_ID", "EVENTMSGTYPE", "PERIOD", "PLAYER1_TEAM_ID"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[(chunk["PERIOD"].isin([1, 2, 3, 4])) & (chunk["EVENTMSGTYPE"] == event_type)]
        for row in chunk.itertuples(index=False):
            team = clean_team_id(row.PLAYER1_TEAM_ID)
            if team and team != "0":
                counts[(clean_game_id(row.GAME_ID), int(row.PERIOD), team)] += 1
    return counts


def validate_frame(frame, path):
    values = frame[VARIABLES].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"{path} has nonnumeric values")
    if (values < 0).any().any():
        raise ValueError(f"{path} has negative count values")
    if not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
        raise ValueError(f"{path} has noninteger count values")
    if (frame["FTM"] > frame["FTA"]).any():
        raise ValueError(f"{path} violates FTM <= FTA")


def write_local_scores(path, family, local, fits):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "child", "parent_set", "bic", "fit"])
        for child, table in enumerate(local):
            for parent_mask, bic in sorted(table.items()):
                parents = [VARIABLES[j] for j in range(len(VARIABLES)) if parent_mask & (1 << j)]
                writer.writerow([family, VARIABLES[child], ";".join(parents), f"{bic:.12g}", fits[child][parent_mask]])


def maybe_reuse_local_scores(path):
    if not path.exists():
        return None
    local = [{} for _ in VARIABLES]
    fits = [{} for _ in VARIABLES]
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            child = VARIABLES.index(row["child"])
            parent_mask = 0
            if row["parent_set"]:
                for parent in row["parent_set"].split(";"):
                    parent_mask |= 1 << VARIABLES.index(parent)
            local[child][parent_mask] = float(row["bic"])
            fits[child][parent_mask] = ast.literal_eval(row["fit"]) if row["fit"] else None
    return local, fits


def run_season(season, start_year, args):
    counts_path = args.processed_dir / f"quarter_counts_{season}_{args.foul_mode}.csv"
    frame = pd.read_csv(counts_path)
    raw_path = args.dataset_dir / f"nbastats_{start_year}.csv"
    miss_counts = collect_event_counts(raw_path, MISS_FG_EVENT, args.chunksize)
    rebound_counts = collect_event_counts(raw_path, REB_EVENT, args.chunksize)
    frame["MISS_FG"] = [
        miss_counts[(clean_game_id(game_id), int(period), clean_team_id(team_id))]
        for game_id, period, team_id in zip(frame["game_id"], frame["period"], frame["team_id"])
    ]
    frame["REB"] = [
        rebound_counts[(clean_game_id(game_id), int(period), clean_team_id(team_id))]
        for game_id, period, team_id in zip(frame["game_id"], frame["period"], frame["team_id"])
    ]
    validate_frame(frame, counts_path)

    if args.unit == "game-quarter":
        frame = frame.groupby(["season", "game_id", "period"], as_index=False)[VARIABLES].sum()
        validate_frame(frame, counts_path)

    count_path = args.outputs_dir / args.unit / f"{args.unit}_counts_{season}_{args.foul_mode}_missfg_reb.csv"
    count_path.parent.mkdir(parents=True, exist_ok=True)
    id_columns = ["season", "game_id", "period"] + ([] if args.unit == "game-quarter" else ["team_id"])
    frame[[*id_columns, *VARIABLES]].to_csv(count_path, index=False)

    pt.VARIABLES = VARIABLES
    data = frame[VARIABLES].to_numpy(dtype="int64")
    print(
        f"[{season}] n={len(frame)} means="
        + ", ".join(f"{name}={data[:, i].mean():.3f}" for i, name in enumerate(VARIABLES)),
        flush=True,
    )
    row = {"season": season, "n": len(frame)}
    for family in args.families:
        local_path = args.outputs_dir / f"local_scores_{season}_{family}.csv"
        cached = None if args.refit else maybe_reuse_local_scores(local_path)
        if cached is None:
            print(f"[{season}] fitting {family}", flush=True)
            local, fits = pt.build_local_scores(data, family, args.maxiter, args.workers)
            if args.write_local:
                write_local_scores(local_path, family, local, fits)
        else:
            print(f"[{season}] reusing {family} local scores", flush=True)
            local, fits = cached

        n_acyclic, n_finite, ranked = pt.enumerate_dags(local, args.top_k)
        if n_acyclic != 29281:
            raise RuntimeError(f"expected 29281 acyclic DAGs over five nodes, got {n_acyclic}")
        if n_finite != 29281:
            raise RuntimeError(f"{season} {family} has only {n_finite} finite DAG scores")
        best_bic, best_edges, best_parent_masks = ranked[0]
        row[f"{family}_bic"] = best_bic
        row[f"{family}_edges"] = pt.format_edges(best_edges)
        row[f"{family}_parent_masks"] = "|".join(map(str, best_parent_masks))
        row[f"{family}_n_acyclic"] = n_acyclic
        row[f"{family}_n_finite"] = n_finite
        print(f"  {family}: BIC={best_bic:,.3f} {pt.format_edges(best_edges)}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/rebound_control_study"))
    parser.add_argument("--families", nargs="+", default=["poisson", "nb"], choices=["poisson", "nb"])
    parser.add_argument("--season", choices=list(SEASONS), help="Run only one season.")
    parser.add_argument("--foul-mode", choices=["drawn"], default="drawn")
    parser.add_argument("--unit", choices=["team-quarter", "game-quarter"], default="team-quarter")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--write-local", action="store_true")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    seasons = {args.season: SEASONS[args.season]} if args.season else SEASONS
    rows = [run_season(season, start_year, args) for season, start_year in seasons.items()]
    out_path = args.outputs_dir / args.unit / f"real_foul_fta_ftm_missfg_reb_results_{args.foul_mode}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
