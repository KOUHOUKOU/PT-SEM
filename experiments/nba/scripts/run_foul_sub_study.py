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


VARIABLES = ["FOUL", "FTA", "FTM", "SUB"]
SUB_EVENT = 8

# FOUL->FTA, FTA->FTM, FOUL->SUB with VARIABLES order above.
TARGET_MASKS = (0, 1, 2, 1)

# SUB->FOUL, FOUL->FTA, FTA->FTM.
REVERSE_SUB_MASKS = (8, 1, 2, 0)


def collect_substitution_counts(path, chunksize):
    counts = defaultdict(int)
    columns = ["GAME_ID", "EVENTMSGTYPE", "PERIOD", "PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[(chunk["PERIOD"].isin([1, 2, 3, 4])) & (chunk["EVENTMSGTYPE"] == SUB_EVENT)]
        for row in chunk.itertuples(index=False):
            team1 = clean_team_id(row.PLAYER1_TEAM_ID)
            team2 = clean_team_id(row.PLAYER2_TEAM_ID)
            if not team1 or team1 == "0":
                raise ValueError(f"{path} has a substitution row without PLAYER1_TEAM_ID")
            if team2 and team2 != "0" and team1 != team2:
                raise ValueError(f"{path} has a substitution team mismatch: {team1} vs {team2}")
            counts[(clean_game_id(row.GAME_ID), int(row.PERIOD), team1)] += 1
    return counts


def validate_counts(frame, path):
    required = {"season", "game_id", "period", "team_id", *VARIABLES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    duplicates = frame.duplicated(["game_id", "period", "team_id"])
    if duplicates.any():
        raise ValueError(f"{path} has {int(duplicates.sum())} duplicate game-period-team rows")

    values = frame[VARIABLES].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"{path} has nonnumeric count values")
    if (values < 0).any().any():
        raise ValueError(f"{path} has negative count values")
    if not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
        raise ValueError(f"{path} has noninteger count values")
    if (frame["FTM"] > frame["FTA"]).any():
        raise ValueError(f"{path} violates FTM <= FTA")

    teams_per_quarter = frame.groupby(["game_id", "period"]).size()
    bad = teams_per_quarter[teams_per_quarter != 2]
    if not bad.empty:
        raise ValueError(f"{path} has {len(bad)} game-quarter groups without exactly two teams")


def build_team_quarter_frame(season, start_year, args):
    counts_path = args.processed_dir / f"quarter_counts_{season}_{args.foul_mode}.csv"
    if not counts_path.exists():
        raise FileNotFoundError(counts_path)
    frame = pd.read_csv(counts_path)

    raw_path = args.dataset_dir / f"nbastats_{start_year}.csv"
    sub_counts = collect_substitution_counts(raw_path, args.chunksize)
    frame["SUB"] = [
        sub_counts[(clean_game_id(game_id), int(period), clean_team_id(team_id))]
        for game_id, period, team_id in zip(frame["game_id"], frame["period"], frame["team_id"])
    ]
    validate_counts(frame, counts_path)
    return frame


def to_game_quarter_frame(team_frame):
    columns = ["season", "game_id", "period"]
    frame = team_frame.groupby(columns, as_index=False)[VARIABLES].sum()
    if (frame["FTM"] > frame["FTA"]).any():
        raise ValueError("game-quarter frame violates FTM <= FTA")
    return frame


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


def score_masks(local, masks):
    return float(sum(local[child][mask] for child, mask in enumerate(masks)))


def fit_frame(frame, season, unit, args):
    pt.VARIABLES = VARIABLES
    data = frame[VARIABLES].to_numpy(dtype="int64")
    print(
        f"[{unit} {season}] n={len(frame)} means="
        + ", ".join(f"{name}={data[:, i].mean():.3f}" for i, name in enumerate(VARIABLES)),
        flush=True,
    )

    row = {"unit": unit, "season": season, "n": len(frame)}
    for family in args.families:
        local_path = args.outputs_dir / unit / f"local_scores_{season}_{family}.csv"
        cached = None if args.refit else maybe_reuse_local_scores(local_path)
        if cached is None:
            print(f"[{unit} {season}] fitting {family}", flush=True)
            local, fits = pt.build_local_scores(data, family, args.maxiter, args.workers)
            if args.write_local:
                write_local_scores(local_path, family, local, fits)
        else:
            print(f"[{unit} {season}] reusing {family} local scores", flush=True)
            local, fits = cached

        n_acyclic, n_finite, ranked = pt.enumerate_dags(local, args.top_k)
        if n_acyclic != 543:
            raise RuntimeError(f"expected 543 acyclic DAGs over four nodes, got {n_acyclic}")
        if n_finite != 543:
            raise RuntimeError(f"{unit} {season} {family} has only {n_finite} finite DAG scores")

        best_bic, best_edges, best_parent_masks = ranked[0]
        target_bic = score_masks(local, TARGET_MASKS)
        reverse_bic = score_masks(local, REVERSE_SUB_MASKS)
        row[f"{family}_bic"] = best_bic
        row[f"{family}_edges"] = pt.format_edges(best_edges)
        row[f"{family}_parent_masks"] = "|".join(map(str, best_parent_masks))
        row[f"{family}_target_bic"] = target_bic
        row[f"{family}_target_delta"] = target_bic - best_bic
        row[f"{family}_reverse_sub_bic"] = reverse_bic
        row[f"{family}_reverse_sub_delta"] = reverse_bic - best_bic
        row[f"{family}_n_acyclic"] = n_acyclic
        row[f"{family}_n_finite"] = n_finite
        print(
            f"  {family}: BIC={best_bic:,.3f} {pt.format_edges(best_edges)} "
            f"target_delta={target_bic - best_bic:.3f}",
            flush=True,
        )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/foul_sub_study"))
    parser.add_argument("--families", nargs="+", default=["poisson", "nb"], choices=["poisson", "nb"])
    parser.add_argument("--season", choices=list(SEASONS), help="Run only one season.")
    parser.add_argument("--foul-mode", choices=["drawn"], default="drawn")
    parser.add_argument("--unit", choices=["team-quarter", "game-quarter", "both"], default="both")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--write-local", action="store_true")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    seasons = {args.season: SEASONS[args.season]} if args.season else SEASONS
    rows_by_unit = defaultdict(list)
    for season, start_year in seasons.items():
        team_frame = build_team_quarter_frame(season, start_year, args)
        team_out = args.outputs_dir / "team-quarter" / f"quarter_counts_{season}_{args.foul_mode}_sub.csv"
        team_out.parent.mkdir(parents=True, exist_ok=True)
        team_frame.to_csv(team_out, index=False)

        if args.unit in {"team-quarter", "both"}:
            rows_by_unit["team-quarter"].append(fit_frame(team_frame, season, "team-quarter", args))
        if args.unit in {"game-quarter", "both"}:
            game_frame = to_game_quarter_frame(team_frame)
            game_out = args.outputs_dir / "game-quarter" / f"game_quarter_counts_{season}_{args.foul_mode}_sub.csv"
            game_out.parent.mkdir(parents=True, exist_ok=True)
            game_frame.to_csv(game_out, index=False)
            rows_by_unit["game-quarter"].append(fit_frame(game_frame, season, "game-quarter", args))

    for unit, rows in rows_by_unit.items():
        out_path = args.outputs_dir / unit / f"real_foul_fta_ftm_sub_results_{args.foul_mode}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
