import argparse
import ast
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from build_quarter_counts import DEFAULT_DATASET_DIR, SEASONS, aggregate_file
from ptsem_three_node import VARIABLES, build_local_scores, enumerate_dags, format_edges


def validate_counts_frame(frame, path):
    required = {"season", "game_id", "period", "team_id", *VARIABLES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    key = ["game_id", "period", "team_id"]
    duplicates = frame.duplicated(key)
    if duplicates.any():
        raise ValueError(f"{path} has {int(duplicates.sum())} duplicate game-period-team rows")

    periods = pd.to_numeric(frame["period"], errors="coerce")
    if periods.isna().any() or not set(periods.astype(int).unique()).issubset({1, 2, 3, 4}):
        raise ValueError(f"{path} contains periods outside regulation quarters 1--4")

    for name in VARIABLES:
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{path} has nonnumeric values in {name}")
        if (values < 0).any():
            raise ValueError(f"{path} has negative values in {name}")
        if not np.allclose(values, np.round(values)):
            raise ValueError(f"{path} has noninteger values in {name}")

    if (frame["FTM"] > frame["FTA"]).any():
        raise ValueError(f"{path} violates FTM <= FTA")

    teams_per_quarter = frame.groupby(["game_id", "period"]).size()
    bad_quarters = teams_per_quarter[teams_per_quarter != 2]
    if not bad_quarters.empty:
        raise ValueError(f"{path} has {len(bad_quarters)} game-quarter groups without exactly two teams")


def load_counts(path, validate=True):
    frame = pd.read_csv(path)
    if validate:
        validate_counts_frame(frame, path)
    return frame[VARIABLES].to_numpy(dtype="int64"), frame


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
    if args.rebuild_counts or not counts_path.exists():
        src = args.dataset_dir / f"nbastats_{start_year}.csv"
        print(f"[{season}] building quarter counts", flush=True)
        stats = aggregate_file(src, season, counts_path, args.foul_mode, args.chunksize)
        print(
            f"  games={stats['games']} excluded_offensive_fouls={stats['excluded_offensive_fouls']} "
            f"counted_fouls={stats['counted_fouls']} "
            f"fta={stats['free_throw_rows']} ftm={stats['made_free_throws']}",
            flush=True,
        )

    data, frame = load_counts(counts_path, validate=not args.no_validate_counts)
    print(f"[{season}] observations={len(frame)} means=" + ", ".join(
        f"{name}={data[:, i].mean():.3f}" for i, name in enumerate(VARIABLES)
    ), flush=True)

    row = {"season": season, "n": len(frame)}
    for family in args.families:
        local_path = args.outputs_dir / f"local_scores_{season}_{args.foul_mode}_{family}.csv"
        cached = None if args.refit else maybe_reuse_local_scores(local_path)
        if cached is None:
            print(f"[{season}] fitting {family}", flush=True)
            local, fits = build_local_scores(data, family, args.maxiter, args.workers)
            if args.write_local:
                write_local_scores(local_path, family, local, fits)
        else:
            print(f"[{season}] reusing {family} local scores", flush=True)
            local, fits = cached
        n_acyclic, n_finite, ranked = enumerate_dags(local, args.top_k)
        if n_acyclic != 25:
            raise RuntimeError(f"expected 25 acyclic DAGs over three nodes, got {n_acyclic}")
        if n_finite != 25:
            raise RuntimeError(f"{season} {family} has only {n_finite} finite DAG scores")
        if not ranked:
            raise RuntimeError(f"{season} {family} produced no ranked DAGs")
        best_bic, best_edges, best_parent_masks = ranked[0]
        row[f"{family}_bic"] = best_bic
        row[f"{family}_edges"] = format_edges(best_edges)
        row[f"{family}_parent_masks"] = "|".join(map(str, best_parent_masks))
        row[f"{family}_n_acyclic"] = n_acyclic
        row[f"{family}_n_finite"] = n_finite
        print(f"  {family}: BIC={best_bic:,.3f} {format_edges(best_edges)}", flush=True)
    return row


def file_info(path):
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def write_metadata(path, args, seasons):
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "families": args.families,
        "foul_mode": args.foul_mode,
        "maxiter": args.maxiter,
        "top_k": args.top_k,
        "dataset_dir": str(args.dataset_dir),
        "processed_dir": str(args.processed_dir),
        "outputs_dir": str(args.outputs_dir),
        "seasons": {
            season: {
                "source": file_info(args.dataset_dir / f"nbastats_{start_year}.csv"),
                "counts": file_info(args.processed_dir / f"quarter_counts_{season}_{args.foul_mode}.csv"),
            }
            for season, start_year in seasons.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--families", nargs="+", default=["poisson", "nb"], choices=["poisson", "nb"])
    parser.add_argument("--season", choices=list(SEASONS), help="Run only one season.")
    parser.add_argument("--foul-mode", choices=["drawn", "committed"], default="drawn")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--write-local", action="store_true")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--rebuild-counts", action="store_true")
    parser.add_argument("--no-validate-counts", action="store_true")
    args = parser.parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    seasons = {args.season: SEASONS[args.season]} if args.season else SEASONS
    rows = [run_season(season, start_year, args) for season, start_year in seasons.items()]
    out_path = args.outputs_dir / f"real_foul_fta_ftm_results_{args.foul_mode}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    write_metadata(out_path.with_suffix(".metadata.json"), args, seasons)
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
