"""Derive the four-node FOUL/FTA/FTM/REB LibraryDP fit from validated caches.

Local likelihoods depend only on a child and its selected parent columns.
Therefore the matching local fits from the completed five-node optimization
are exactly the local fits required after MISS_FG is removed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


VARIABLES = ("FOUL", "FTA", "FTM", "REB")
REFERENCE = {("FOUL", "FTA"), ("FTA", "FTM"), ("FTA", "REB")}
SEASONS = ("2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_mask(text: object) -> int:
    if pd.isna(text) or not str(text).strip():
        return 0
    names = str(text).split(";")
    return sum(1 << VARIABLES.index(name) for name in names)


def edge_set(masks: tuple[int, ...]) -> set[tuple[str, str]]:
    return {
        (VARIABLES[parent], VARIABLES[child])
        for child, mask in enumerate(masks)
        for parent in range(len(VARIABLES))
        if (mask >> parent) & 1
    }


def fmt(edges: set[tuple[str, str]]) -> str:
    return ", ".join(f"{a}->{b}" for a, b in sorted(edges)) or "(none)"


def score(predicted: set, truth: set) -> tuple[float, float, float]:
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def evaluate(edges: set[tuple[str, str]]) -> dict[str, object]:
    skeleton = {frozenset(edge) for edge in edges}
    truth_skeleton = {frozenset(edge) for edge in REFERENCE}
    sp, sr, sf = score(skeleton, truth_skeleton)
    dp, dr, df = score(edges, REFERENCE)
    return {
        "skeleton_precision": sp,
        "skeleton_recall": sr,
        "skeleton_f1": sf,
        "directed_precision": dp,
        "directed_recall": dr,
        "directed_f1": df,
        "exact_reference_graph": edges == REFERENCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/adaptive_ptsem_real_study/five_game"),
    )
    parser.add_argument(
        "--core", type=Path, default=REPO_ROOT / "experiments/simulation/legacy/d.py"
    )
    parser.add_argument(
        "--runner", type=Path, default=Path("src/run_adaptive_ptsem_real_study.py")
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/adaptive_ptsem_four_reb_study"),
    )
    args = parser.parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    core = load_module(args.core.resolve(), "four_reb_latest_core")
    runner = load_module(args.runner.resolve(), "four_reb_real_runner")

    rows = []
    selected_rows = []
    provenance = {}
    for season in SEASONS:
        source = (
            args.cache_root
            / season
            / "optimization"
            / "local_family_fits.csv"
        )
        frame = pd.read_csv(source)
        frame = frame[
            frame["child"].isin(VARIABLES)
            & frame["parent_set"].fillna("").map(
                lambda text: all(
                    name in VARIABLES
                    for name in str(text).split(";")
                    if name
                )
            )
        ].copy()
        frame["new_parent_mask"] = frame["parent_set"].map(parent_mask)
        d = len(VARIABLES)
        scores = [[math.inf] * (1 << d) for _ in range(d)]
        chosen = {}
        for child, child_name in enumerate(VARIABLES):
            subset = frame[frame["child"].eq(child_name)]
            for mask, group in subset.groupby("new_parent_mask"):
                finite = group[np.isfinite(group["bic"])]
                if finite.empty:
                    continue
                best = finite.loc[finite["bic"].idxmin()]
                scores[child][int(mask)] = float(best["bic"])
                chosen[(child, int(mask))] = best

        masks_dict, best_score = core.exact_order_dp(scores)
        masks = tuple(int(masks_dict[node]) for node in range(d))
        dag_count, ranked = runner.top_dags(scores, max_parents=d - 1, top_k=10)
        if dag_count != 543:
            raise RuntimeError(f"{season}: expected 543 DAGs, got {dag_count}")
        if ranked[0][1] != masks:
            masks = tuple(ranked[0][1])
            best_score = float(ranked[0][0])
        edges = edge_set(masks)
        second = float(ranked[1][0])
        selected_families = []
        for child, mask in enumerate(masks):
            fit = chosen[(child, mask)]
            selected_families.append(str(fit["family"]))
            selected_rows.append(
                {
                    "season": season,
                    "node": VARIABLES[child],
                    "parent_set": fit["parent_set"],
                    "family": fit["family"],
                    "bic": fit["bic"],
                    "alpha": fit["alpha"],
                    "params": fit["params"],
                }
            )
        row = {
            "dataset": "four_reb_game",
            "unit": "game-quarter",
            "season": season,
            "n": int(frame["n"].iloc[0]),
            "estimator": "optimization",
            "edges": fmt(edges),
            "selected_families": ";".join(selected_families),
            "score": float(best_score),
            "bic_delta_second": second - float(best_score),
            "dag_count": dag_count,
            **evaluate(edges),
        }
        rows.append(row)
        provenance[season] = {
            "source": str(source.resolve()),
            "source_dataset": "five_game",
            "excluded_variable": "MISS_FG",
            "identity_reason": (
                "A local likelihood uses only its child and selected parents; "
                "all retained child-parent column sets are unchanged."
            ),
        }
        print(season, fmt(edges), flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(args.outputs_dir / "graph_results_detailed.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(
        args.outputs_dir / "selected_local_fits.csv", index=False
    )
    summary = {
        "method": "PT-SEM (LibraryDP)",
        "n_seasons": 6,
        **{
            f"mean_{metric}": float(detail[metric].mean())
            for metric in (
                "skeleton_precision",
                "skeleton_recall",
                "skeleton_f1",
                "directed_precision",
                "directed_recall",
                "directed_f1",
            )
        },
        "exact_recovery_count": int(detail["exact_reference_graph"].sum()),
    }
    pd.DataFrame([summary]).to_csv(
        args.outputs_dir / "ptsem_summary.csv", index=False
    )
    (args.outputs_dir / "cache_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
