"""Fast six-family LibraryDP screen of interpretable four/five-node hypotheses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


BASE = ("FOUL", "FTA", "FTM")
BASE_EDGES = {("FOUL", "FTA"), ("FTA", "FTM")}
SEASONS = (
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
)

HYPOTHESES = {
    "shot_foul_4": {
        "variables": (*BASE, "SHOT_FOUL_DRAWN"),
        "edges": {
            *BASE_EDGES,
            ("FOUL", "SHOT_FOUL_DRAWN"),
            ("SHOT_FOUL_DRAWN", "FTA"),
        },
    },
    "personal_foul_4": {
        "variables": (*BASE, "PERS_FOUL_DRAWN"),
        "edges": {*BASE_EDGES, ("FOUL", "PERS_FOUL_DRAWN")},
    },
    "miss_ft_4": {
        "variables": (*BASE, "MISS_FT"),
        "edges": {*BASE_EDGES, ("FTA", "MISS_FT")},
    },
    "reb_4": {
        "variables": (*BASE, "REB"),
        "edges": {*BASE_EDGES, ("FTA", "REB")},
    },
    "miss_fg_control_4": {
        "variables": (*BASE, "MISS_FG"),
        "edges": set(BASE_EDGES),
    },
    "fga_fgm_5": {
        "variables": (*BASE, "FGA", "FGM"),
        "edges": {*BASE_EDGES, ("FGA", "FGM")},
    },
    "fgm_ast_5": {
        "variables": (*BASE, "FGM", "AST"),
        "edges": {*BASE_EDGES, ("FGM", "AST")},
    },
    "miss_blk_5": {
        "variables": (*BASE, "MISS_FG", "BLK"),
        "edges": {*BASE_EDGES, ("MISS_FG", "BLK")},
    },
    "tov_stl_5": {
        "variables": (*BASE, "TOV", "STL"),
        "edges": {*BASE_EDGES, ("TOV", "STL")},
    },
    "miss_reb_5": {
        "variables": (*BASE, "MISS_FG", "REB"),
        "edges": {
            *BASE_EDGES,
            ("FTA", "REB"),
            ("MISS_FG", "REB"),
        },
    },
    "foul_components_5": {
        "variables": (*BASE, "SHOT_FOUL_DRAWN", "PERS_FOUL_DRAWN"),
        "edges": {
            *BASE_EDGES,
            ("FOUL", "SHOT_FOUL_DRAWN"),
            ("FOUL", "PERS_FOUL_DRAWN"),
            ("SHOT_FOUL_DRAWN", "FTA"),
        },
    },
    "shot_missft_5": {
        "variables": (*BASE, "SHOT_FOUL_DRAWN", "MISS_FT"),
        "edges": {
            *BASE_EDGES,
            ("FOUL", "SHOT_FOUL_DRAWN"),
            ("SHOT_FOUL_DRAWN", "FTA"),
            ("FTA", "MISS_FT"),
        },
    },
    "shot_foul_4_team": {
        "variables": (*BASE, "SHOT_FOUL_DRAWN"),
        "edges": {
            *BASE_EDGES,
            ("FOUL", "SHOT_FOUL_DRAWN"),
            ("SHOT_FOUL_DRAWN", "FTA"),
        },
    },
    "personal_foul_4_team": {
        "variables": (*BASE, "PERS_FOUL_DRAWN"),
        "edges": {*BASE_EDGES, ("FOUL", "PERS_FOUL_DRAWN")},
    },
    "foul_components_5_team": {
        "variables": (*BASE, "SHOT_FOUL_DRAWN", "PERS_FOUL_DRAWN"),
        "edges": {
            *BASE_EDGES,
            ("FOUL", "SHOT_FOUL_DRAWN"),
            ("FOUL", "PERS_FOUL_DRAWN"),
            ("SHOT_FOUL_DRAWN", "FTA"),
        },
    },
    "loose_foul_4_team": {
        "variables": (*BASE, "LOOSE_BALL_FOUL_DRAWN"),
        "edges": {*BASE_EDGES, ("FOUL", "LOOSE_BALL_FOUL_DRAWN")},
    },
    "foul_leaves_5_team": {
        "variables": (*BASE, "PERS_FOUL_DRAWN", "LOOSE_BALL_FOUL_DRAWN"),
        "edges": {
            *BASE_EDGES,
            ("FOUL", "PERS_FOUL_DRAWN"),
            ("FOUL", "LOOSE_BALL_FOUL_DRAWN"),
        },
    },
}


def load_core(path: Path):
    os.environ["PTSEM_FAMILY_LIBRARY"] = (
        "Poisson,NB,ZIP,Geom,Binomial,Bernoulli"
    )
    spec = importlib.util.spec_from_file_location("expanded_scan_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def edges_from_masks(masks: dict[int, int], variables: tuple[str, ...]) -> set:
    return {
        (variables[parent], variables[child])
        for child, mask in masks.items()
        for parent in range(len(variables))
        if (mask >> parent) & 1
    }


def fmt(edges: set) -> str:
    return ", ".join(f"{a}->{b}" for a, b in sorted(edges)) or "(none)"


def metrics(pred: set, truth: set) -> tuple[float, float, float]:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/expanded_candidate_study/data"),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/expanded_candidate_study/moment_screen"),
    )
    parser.add_argument(
        "--core", type=Path, default=REPO_ROOT / "experiments/simulation/legacy/d.py"
    )
    args = parser.parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    core = load_core(args.core.resolve())
    rows = []
    for name, spec in HYPOTHESES.items():
        variables = tuple(spec["variables"])
        truth = set(spec["edges"])
        truth_skeleton = {frozenset(edge) for edge in truth}
        for season in SEASONS:
            frame = pd.read_csv(
                args.data_dir / f"expanded_game_quarter_{season}.csv"
            )
            X = frame[list(variables)].to_numpy(dtype=np.int64)
            scores, fits = core.precompute_local_scores(
                X, "LibraryDP", max_parents=len(variables) - 1
            )
            masks, score = core.exact_order_dp(scores)
            pred = edges_from_masks(masks, variables)
            pred_skeleton = {frozenset(edge) for edge in pred}
            sp, sr, sf = metrics(pred_skeleton, truth_skeleton)
            dp, dr, df = metrics(pred, truth)
            selected = [
                fits[node][masks[node]].family
                for node in range(len(variables))
            ]
            rows.append(
                {
                    "hypothesis": name,
                    "season": season,
                    "variables": ";".join(variables),
                    "reference_edges": fmt(truth),
                    "estimated_edges": fmt(pred),
                    "selected_families": ";".join(selected),
                    "score": score,
                    "skeleton_precision": sp,
                    "skeleton_recall": sr,
                    "skeleton_f1": sf,
                    "directed_precision": dp,
                    "directed_recall": dr,
                    "directed_f1": df,
                    "exact": pred == truth,
                }
            )
            print(name, season, fmt(pred), flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(args.outputs_dir / "screen_details.csv", index=False)
    summary = (
        detail.groupby("hypothesis", sort=False)
        .agg(
            n_seasons=("season", "nunique"),
            mean_skeleton_precision=("skeleton_precision", "mean"),
            mean_skeleton_recall=("skeleton_recall", "mean"),
            mean_skeleton_f1=("skeleton_f1", "mean"),
            mean_directed_precision=("directed_precision", "mean"),
            mean_directed_recall=("directed_recall", "mean"),
            mean_directed_f1=("directed_f1", "mean"),
            exact_count=("exact", "sum"),
        )
        .reset_index()
        .sort_values(
            ["exact_count", "mean_directed_f1", "mean_skeleton_f1"],
            ascending=False,
        )
    )
    summary.to_csv(args.outputs_dir / "screen_summary.csv", index=False)
    (args.outputs_dir / "hypotheses.json").write_text(
        json.dumps(
            {
                key: {
                    "variables": value["variables"],
                    "edges": sorted(f"{a}->{b}" for a, b in value["edges"]),
                }
                for key, value in HYPOTHESES.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
