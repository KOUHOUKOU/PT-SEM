import argparse
import ast
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import ptsem_three_node as pt
from build_quarter_counts import DEFAULT_DATASET_DIR, SEASONS, clean_game_id


VARIABLES = ["FOUL", "FTA", "FTM", "TIMEOUT"]
TIMEOUT_EVENT = 9
TIMEOUT_ACTIONS = {
    "team": {1, 2, 7},
    "all": {1, 2, 4, 7},
}

# FOUL->FTA, FTA->FTM, FOUL->TIMEOUT with VARIABLES order above.
TARGET_MASKS = (0, 1, 2, 1)

# TIMEOUT->FOUL, FOUL->FTA, FTA->FTM.
REVERSE_TIMEOUT_MASKS = (8, 1, 2, 0)


def collect_timeout_counts(path, timeout_mode, chunksize):
    counts = defaultdict(int)
    actions = TIMEOUT_ACTIONS[timeout_mode]
    columns = ["GAME_ID", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[
            (chunk["PERIOD"].isin([1, 2, 3, 4]))
            & (chunk["EVENTMSGTYPE"] == TIMEOUT_EVENT)
            & (chunk["EVENTMSGACTIONTYPE"].isin(actions))
        ]
        for row in chunk.itertuples(index=False):
            counts[(clean_game_id(row.GAME_ID), int(row.PERIOD))] += 1
    return counts


def to_game_quarter_frame(team_frame):
    columns = ["season", "game_id", "period"]
    frame = team_frame.groupby(columns, as_index=False)[["FOUL", "FTA", "FTM"]].sum()
    if (frame["FTM"] > frame["FTA"]).any():
        raise ValueError("game-quarter frame violates FTM <= FTA")
    return frame


def validate_frame(frame):
    values = frame[VARIABLES].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("nonnumeric count values")
    if (values < 0).any().any():
        raise ValueError("negative count values")
    if not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
        raise ValueError("noninteger count values")
    if (frame["FTM"] > frame["FTA"]).any():
        raise ValueError("FTM > FTA")


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


def run_season(season, start_year, args):
    team_counts_path = args.processed_dir / f"quarter_counts_{season}_{args.foul_mode}.csv"
    if not team_counts_path.exists():
        raise FileNotFoundError(team_counts_path)
    frame = to_game_quarter_frame(pd.read_csv(team_counts_path))

    raw_path = args.dataset_dir / f"nbastats_{start_year}.csv"
    timeout_counts = collect_timeout_counts(raw_path, args.timeout_mode, args.chunksize)
    frame["TIMEOUT"] = [
        timeout_counts[(clean_game_id(game_id), int(period))]
        for game_id, period in zip(frame["game_id"], frame["period"])
    ]
    validate_frame(frame)

    count_path = args.outputs_dir / f"game_quarter_counts_{season}_{args.foul_mode}_{args.timeout_mode}_timeout.csv"
    count_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(count_path, index=False)

    pt.VARIABLES = VARIABLES
    data = frame[VARIABLES].to_numpy(dtype="int64")
    print(
        f"[{season}] n={len(frame)} means="
        + ", ".join(f"{name}={data[:, i].mean():.3f}" for i, name in enumerate(VARIABLES)),
        flush=True,
    )

    row = {"season": season, "n": len(frame), "timeout_mode": args.timeout_mode}
    for family in args.families:
        local_path = args.outputs_dir / f"local_scores_{season}_{args.timeout_mode}_{family}.csv"
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
        if n_acyclic != 543:
            raise RuntimeError(f"expected 543 acyclic DAGs over four nodes, got {n_acyclic}")
        if n_finite != 543:
            raise RuntimeError(f"{season} {family} has only {n_finite} finite DAG scores")

        best_bic, best_edges, best_parent_masks = ranked[0]
        target_bic = score_masks(local, TARGET_MASKS)
        reverse_bic = score_masks(local, REVERSE_TIMEOUT_MASKS)
        row[f"{family}_bic"] = best_bic
        row[f"{family}_edges"] = pt.format_edges(best_edges)
        row[f"{family}_parent_masks"] = "|".join(map(str, best_parent_masks))
        row[f"{family}_target_bic"] = target_bic
        row[f"{family}_target_delta"] = target_bic - best_bic
        row[f"{family}_reverse_timeout_bic"] = reverse_bic
        row[f"{family}_reverse_timeout_delta"] = reverse_bic - best_bic
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
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/foul_timeout_study"))
    parser.add_argument("--families", nargs="+", default=["poisson", "nb"], choices=["poisson", "nb"])
    parser.add_argument("--season", choices=list(SEASONS), help="Run only one season.")
    parser.add_argument("--foul-mode", choices=["drawn"], default="drawn")
    parser.add_argument("--timeout-mode", choices=sorted(TIMEOUT_ACTIONS), default="team")
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
    out_path = args.outputs_dir / f"real_game_quarter_foul_fta_ftm_{args.timeout_mode}_timeout_results_{args.foul_mode}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
