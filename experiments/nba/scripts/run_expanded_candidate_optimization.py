"""Run the final optimized six-family PT-SEM for one candidate hypothesis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

import run_adaptive_ptsem_real_study as runner
from scan_expanded_candidates_moment import HYPOTHESES, SEASONS

REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(edges: set[tuple[str, str]]) -> str:
    return ", ".join(f"{a}->{b}" for a, b in sorted(edges)) or "(none)"


def parse_edges(text: str) -> set[tuple[str, str]]:
    if text == "(none)":
        return set()
    return {
        tuple(part.strip().split("->", 1))
        for part in text.split(",")
    }


def metric(pred: set, truth: set) -> tuple[float, float, float]:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hypothesis", choices=sorted(HYPOTHESES))
    parser.add_argument(
        "--workspace", type=Path, default=Path.cwd()
    )
    parser.add_argument(
        "--core", type=Path, default=REPO_ROOT / "experiments/simulation/legacy/d.py"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/expanded_candidate_study/optimization"),
    )
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seasons", nargs="+", choices=SEASONS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    outputs = args.outputs_dir.resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    spec = HYPOTHESES[args.hypothesis]
    is_team = args.hypothesis.endswith("_team")
    is_loose = args.hypothesis in {
        "loose_foul_4_team",
        "foul_leaves_5_team",
    }
    runner.DATASET_SPECS[args.hypothesis] = {
        "variables": tuple(spec["variables"]),
        "pattern": (
            (
            "outputs/expanded_candidate_study/data_team_loose/"
            if is_loose
            else
            "outputs/expanded_candidate_study/data_team/"
            )
            + "expanded_team_quarter_{season}.csv"
            if is_team
            else
            "outputs/expanded_candidate_study/data/"
            "expanded_game_quarter_{season}.csv"
        ),
        "unit": "team-quarter" if is_team else "game-quarter",
    }
    core = runner.load_paper_core(args.core.resolve())
    core_hash = sha256(args.core.resolve())
    truth = set(spec["edges"])
    truth_skeleton = {frozenset(edge) for edge in truth}
    rows = []
    selected_seasons = tuple(args.seasons) if args.seasons else SEASONS
    for season in selected_seasons:
        result = runner.run_one(
            core,
            workspace,
            outputs,
            args.hypothesis,
            season,
            "optimization",
            args.starts,
            args.maxiter,
            args.workers,
            args.force,
            core_hash,
        )
        pred = parse_edges(result["edges"])
        pred_skeleton = {frozenset(edge) for edge in pred}
        sp, sr, sf = metric(pred_skeleton, truth_skeleton)
        dp, dr, df = metric(pred, truth)
        rows.append(
            {
                "hypothesis": args.hypothesis,
                "season": season,
                "variables": ";".join(spec["variables"]),
                "reference_edges": fmt(truth),
                "estimated_edges": result["edges"],
                "selected_families": ";".join(result["selected_families"]),
                "skeleton_precision": sp,
                "skeleton_recall": sr,
                "skeleton_f1": sf,
                "directed_precision": dp,
                "directed_recall": dr,
                "directed_f1": df,
                "exact": pred == truth,
                "runtime_sec": result["runtime_sec"],
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(outputs / args.hypothesis / "evaluation.csv", index=False)
    summary = {
        "hypothesis": args.hypothesis,
        "n_seasons": len(selected_seasons),
        **{
            f"mean_{name}": float(detail[name].mean())
            for name in (
                "skeleton_precision",
                "skeleton_recall",
                "skeleton_f1",
                "directed_precision",
                "directed_recall",
                "directed_f1",
            )
        },
        "exact_count": int(detail["exact"].sum()),
        "total_runtime_sec": float(detail["runtime_sec"].sum()),
    }
    pd.DataFrame([summary]).to_csv(
        outputs / args.hypothesis / "summary.csv", index=False
    )
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
