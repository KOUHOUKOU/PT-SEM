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


BASE_VARIABLES = ["FOUL", "FTA", "FTM"]

CANDIDATES = {
    "FGA": {
        "description": "field-goal attempts; EVENTMSGTYPE in {1,2}, team=PLAYER1_TEAM_ID",
        "event_types": {1, 2},
        "team_col": "PLAYER1_TEAM_ID",
    },
    "FGM": {
        "description": "made field goals; EVENTMSGTYPE=1, team=PLAYER1_TEAM_ID",
        "event_types": {1},
        "team_col": "PLAYER1_TEAM_ID",
    },
    "MISS_FG": {
        "description": "missed field goals; EVENTMSGTYPE=2, team=PLAYER1_TEAM_ID",
        "event_types": {2},
        "team_col": "PLAYER1_TEAM_ID",
    },
    "REB": {
        "description": "rebounds; EVENTMSGTYPE=4, team=PLAYER1_TEAM_ID",
        "event_types": {4},
        "team_col": "PLAYER1_TEAM_ID",
    },
    "TOV": {
        "description": "turnovers; EVENTMSGTYPE=5, team=PLAYER1_TEAM_ID",
        "event_types": {5},
        "team_col": "PLAYER1_TEAM_ID",
    },
    "VIOL": {
        "description": "violations; EVENTMSGTYPE=7, team=PLAYER1_TEAM_ID",
        "event_types": {7},
        "team_col": "PLAYER1_TEAM_ID",
    },
    "SUB": {
        "description": "substitutions; EVENTMSGTYPE=8, team=PLAYER1_TEAM_ID",
        "event_types": {8},
        "team_col": "PLAYER1_TEAM_ID",
    },
    "SHOT_FOUL_DRAWN": {
        "description": "drawn shooting fouls; EVENTMSGTYPE=6, action in {2,29}, team=PLAYER2_TEAM_ID",
        "event_types": {6},
        "action_types": {2, 29},
        "team_col": "PLAYER2_TEAM_ID",
    },
    "PERS_FOUL_DRAWN": {
        "description": "drawn non-shooting personal/block/take fouls; EVENTMSGTYPE=6, action in {1,27,28}, team=PLAYER2_TEAM_ID",
        "event_types": {6},
        "action_types": {1, 27, 28},
        "team_col": "PLAYER2_TEAM_ID",
    },
    "OFF_FOUL_DRAWN": {
        "description": "drawn offensive fouls/charges; EVENTMSGTYPE=6, action in {4,26}, team=PLAYER2_TEAM_ID",
        "event_types": {6},
        "action_types": {4, 26},
        "team_col": "PLAYER2_TEAM_ID",
    },
}


def selected_candidates(names):
    if names == ["all"]:
        return list(CANDIDATES)
    unknown = [name for name in names if name not in CANDIDATES]
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}; choices are {sorted(CANDIDATES)}")
    return names


def collect_candidate_counts(path, candidate, chunksize):
    spec = CANDIDATES[candidate]
    counts = defaultdict(int)
    columns = ["GAME_ID", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", spec["team_col"]]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["PERIOD"].isin([1, 2, 3, 4])]
        chunk = chunk[chunk["EVENTMSGTYPE"].isin(spec["event_types"])]
        if "action_types" in spec:
            chunk = chunk[chunk["EVENTMSGACTIONTYPE"].isin(spec["action_types"])]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            team = clean_team_id(getattr(row, spec["team_col"]))
            if not team or team == "0":
                continue
            counts[(clean_game_id(row.GAME_ID), int(row.PERIOD), team)] += 1
    return counts


def validate_frame(frame, candidate, path):
    variables = BASE_VARIABLES + [candidate]
    values = frame[variables].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"{path} has nonnumeric values")
    if (values < 0).any().any():
        raise ValueError(f"{path} has negative values")
    if not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
        raise ValueError(f"{path} has noninteger values")
    if (frame["FTM"] > frame["FTA"]).any():
        raise ValueError(f"{path} violates FTM <= FTA")
    teams_per_quarter = frame.groupby(["game_id", "period"]).size()
    bad = teams_per_quarter[teams_per_quarter != 2]
    if not bad.empty:
        raise ValueError(f"{path} has {len(bad)} game-quarter groups without exactly two teams")


def write_local_scores(path, variables, family, local, fits):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "child", "parent_set", "bic", "fit"])
        for child, table in enumerate(local):
            for parent_mask, bic in sorted(table.items()):
                parents = [variables[j] for j in range(len(variables)) if parent_mask & (1 << j)]
                writer.writerow([family, variables[child], ";".join(parents), f"{bic:.12g}", fits[child][parent_mask]])


def maybe_reuse_local_scores(path, variables):
    if not path.exists():
        return None
    local = [{} for _ in variables]
    fits = [{} for _ in variables]
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            child = variables.index(row["child"])
            parent_mask = 0
            if row["parent_set"]:
                for parent in row["parent_set"].split(";"):
                    parent_mask |= 1 << variables.index(parent)
            local[child][parent_mask] = float(row["bic"])
            fits[child][parent_mask] = ast.literal_eval(row["fit"]) if row["fit"] else None
    return local, fits


def has_edge(edges, source, target):
    return (source, target) in set(map(tuple, edges))


def run_candidate(season, start_year, candidate, args):
    variables = BASE_VARIABLES + [candidate]
    counts_path = args.processed_dir / f"quarter_counts_{season}_{args.foul_mode}.csv"
    frame = pd.read_csv(counts_path)
    raw_path = args.dataset_dir / f"nbastats_{start_year}.csv"
    candidate_counts = collect_candidate_counts(raw_path, candidate, args.chunksize)
    frame[candidate] = [
        candidate_counts[(clean_game_id(game_id), int(period), clean_team_id(team_id))]
        for game_id, period, team_id in zip(frame["game_id"], frame["period"], frame["team_id"])
    ]
    validate_frame(frame, candidate, counts_path)

    count_path = args.outputs_dir / candidate / f"quarter_counts_{season}_{args.foul_mode}_{candidate}.csv"
    count_path.parent.mkdir(parents=True, exist_ok=True)
    frame[["season", "game_id", "period", "team_id", *variables]].to_csv(count_path, index=False)

    pt.VARIABLES = variables
    data = frame[variables].to_numpy(dtype="int64")
    print(
        f"[{candidate} {season}] n={len(frame)} means="
        + ", ".join(f"{name}={data[:, i].mean():.3f}" for i, name in enumerate(variables)),
        flush=True,
    )

    row = {
        "candidate": candidate,
        "season": season,
        "n": len(frame),
        "candidate_mean": float(frame[candidate].mean()),
        "candidate_nonzero_rate": float((frame[candidate] > 0).mean()),
    }
    for family in args.families:
        local_path = args.outputs_dir / candidate / f"local_scores_{season}_{family}.csv"
        cached = None if args.refit else maybe_reuse_local_scores(local_path, variables)
        if cached is None:
            local, fits = pt.build_local_scores(data, family, args.maxiter, args.workers)
            if args.write_local:
                write_local_scores(local_path, variables, family, local, fits)
        else:
            local, fits = cached

        n_acyclic, n_finite, ranked = pt.enumerate_dags(local, args.top_k)
        if n_acyclic != 543:
            raise RuntimeError(f"expected 543 acyclic DAGs over four nodes, got {n_acyclic}")
        if n_finite != 543:
            raise RuntimeError(f"{candidate} {season} {family} has only {n_finite} finite DAG scores")

        best_bic, best_edges, best_parent_masks = ranked[0]
        chain_ok = has_edge(best_edges, "FOUL", "FTA") and has_edge(best_edges, "FTA", "FTM")
        row[f"{family}_bic"] = best_bic
        row[f"{family}_edges"] = pt.format_edges(best_edges)
        row[f"{family}_parent_masks"] = "|".join(map(str, best_parent_masks))
        row[f"{family}_chain_ok"] = chain_ok
        row[f"{family}_candidate_parents"] = ";".join(
            variables[j] for j in range(len(variables)) if best_parent_masks[variables.index(candidate)] & (1 << j)
        )
        row[f"{family}_candidate_children"] = ";".join(
            child for child, mask in zip(variables, best_parent_masks) if mask & (1 << variables.index(candidate))
        )
        row[f"{family}_n_acyclic"] = n_acyclic
        row[f"{family}_n_finite"] = n_finite
        print(f"  {family}: {pt.format_edges(best_edges)}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/four_var_candidate_scan"))
    parser.add_argument("--families", nargs="+", default=["poisson", "nb"], choices=["poisson", "nb"])
    parser.add_argument("--season", choices=list(SEASONS), help="Run only one season.")
    parser.add_argument("--foul-mode", choices=["drawn"], default="drawn")
    parser.add_argument("--candidates", nargs="+", default=["all"])
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--write-local", action="store_true")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    candidates = selected_candidates(args.candidates)
    seasons = {args.season: SEASONS[args.season]} if args.season else SEASONS

    rows = []
    for candidate in candidates:
        print(f"=== {candidate}: {CANDIDATES[candidate]['description']} ===", flush=True)
        for season, start_year in seasons.items():
            rows.append(run_candidate(season, start_year, candidate, args))

    out_path = args.outputs_dir / f"four_var_candidate_scan_results_{args.foul_mode}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
