"""Run the four-variable NBA baselines with the pinned author implementations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

# KDEpy versions used by the frozen PB-SCM author code still call this
# NumPy 1.x alias.  Preserve the author algorithm under NumPy 2.x.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda value, dtype=float: np.asarray(value, dtype=dtype)


VARIABLES = ["FOUL", "FTA", "FTM", "MISS_FG"]
REFERENCE_EDGES = {("FOUL", "FTA"), ("FTA", "FTM")}
SEASONS = (
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def canonical(edge: tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(edge))


def score_sets(predicted: set, truth: set) -> tuple[float, float, float, int, int, int]:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


def format_edges(edges: set[tuple[str, str]], separator: str = "->") -> str:
    if not edges:
        return "(none)"
    return ", ".join(f"{a}{separator}{b}" for a, b in sorted(edges))


def decode_adjacency(adjacency: np.ndarray) -> tuple[set, set]:
    directed: set[tuple[str, str]] = set()
    undirected: set[tuple[str, str]] = set()
    for i in range(len(VARIABLES)):
        for j in range(i + 1, len(VARIABLES)):
            i_to_j = bool(adjacency[i, j])
            j_to_i = bool(adjacency[j, i])
            if i_to_j and not j_to_i:
                directed.add((VARIABLES[i], VARIABLES[j]))
            elif j_to_i and not i_to_j:
                directed.add((VARIABLES[j], VARIABLES[i]))
            elif i_to_j or j_to_i:
                undirected.add(canonical((VARIABLES[i], VARIABLES[j])))
    return directed, undirected


def decode_graph_estimate(estimate) -> tuple[set, set]:
    directed = {
        (VARIABLES[parent], VARIABLES[child])
        for parent, child in estimate.directed_edges
    }
    undirected = {
        canonical((VARIABLES[first], VARIABLES[second]))
        for first, second in estimate.undirected_edges
    }
    return directed, undirected


def evaluate(
    directed: set[tuple[str, str]],
    undirected: set[tuple[str, str]] | None = None,
) -> dict[str, object]:
    undirected = undirected or set()
    predicted_skeleton = {canonical(edge) for edge in directed} | set(undirected)
    true_skeleton = {canonical(edge) for edge in REFERENCE_EDGES}
    sp, sr, sf, stp, sfp, sfn = score_sets(predicted_skeleton, true_skeleton)
    dp, dr, df, dtp, dfp, dfn = score_sets(directed, REFERENCE_EDGES)
    return {
        "pred_directed_edges": format_edges(directed),
        "pred_undirected_edges": format_edges(undirected, separator="--"),
        "n_pred_directed_edges": len(directed),
        "n_pred_undirected_edges": len(undirected),
        "skeleton_precision": sp,
        "skeleton_recall": sr,
        "skeleton_f1": sf,
        "skeleton_tp": stp,
        "skeleton_fp": sfp,
        "skeleton_fn": sfn,
        "directed_precision": dp,
        "directed_recall": dr,
        "directed_f1": df,
        "directed_tp": dtp,
        "directed_fp": dfp,
        "directed_fn": dfn,
        "exact_recovery": not undirected and directed == REFERENCE_EDGES,
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, subset in detail.groupby("method", sort=False):
        row = {"method": method, "n_seasons": int(subset["season"].nunique())}
        for metric in (
            "skeleton_precision",
            "skeleton_recall",
            "skeleton_f1",
            "directed_precision",
            "directed_recall",
            "directed_f1",
        ):
            row[f"mean_{metric}"] = float(subset[metric].mean())
        row["exact_recovery_count"] = int(subset["exact_recovery"].sum())
        row["exact_recovery_rate"] = float(subset["exact_recovery"].mean())
        row["model_applicable_all_seasons"] = bool(
            subset["model_applicable"].all()
        )
        reasons = sorted(
            {
                str(value)
                for value in subset["applicability_reason"].dropna()
                if str(value)
            }
        )
        row["applicability_reason"] = " | ".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def run_cumulant(core, data: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list]:
    pb_module, util_module = core._load_pbscm_author_modules()
    model = pb_module.PB_SCM(data.T, seed=seed)
    skeleton = np.asarray(model.Hill_Climb_search())
    adjacency = np.asarray(
        util_module.learning_causal_direction(data.T, skeleton)
    )

    # Record why each possible first edge was accepted or rejected. This makes
    # an empty output scientifically auditable instead of silently treating it
    # as an adapter failure.
    empty = np.zeros((len(VARIABLES), len(VARIABLES)))
    empty_score, _ = model.get_total_likelihood(empty)
    candidates = []
    for parent in range(len(VARIABLES)):
        for child in range(len(VARIABLES)):
            if parent == child:
                continue
            graph = empty.copy()
            graph[parent, child] = 1
            score, coefficients = model.get_total_likelihood(graph)
            candidates.append(
                {
                    "edge": f"{VARIABLES[parent]}->{VARIABLES[child]}",
                    "coefficient": float(coefficients[child, parent]),
                    "noise_mean": float(coefficients[child, -1]),
                    "score": float(score),
                    "delta_from_empty": float(score - empty_score),
                    "admissible": bool(np.isfinite(score)),
                }
            )
    return skeleton, adjacency, candidates


def run_pgf(core, data: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pb_module, _ = core._load_pbscm_pgf_author_modules()
    np.random.seed(seed)
    model = pb_module.PBSCM_PGF(data.T)
    model.learn()
    return np.asarray(model.skeleton), np.asarray(model.dag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/rebound_control_study/game-quarter"),
    )
    parser.add_argument(
        "--pattern", default="game-quarter_counts_{season}_drawn_missfg_reb.csv"
    )
    parser.add_argument(
        "--core",
        type=Path,
        default=REPO_ROOT / "src/nba_core.py",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/adaptive_ptsem_four_var_study/baselines"),
    )
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    core = load_module(args.core.resolve(), "ptsem_baseline_core")

    checkpoint_path = args.outputs_dir / "checkpoint.json"
    checkpoint_signature = {
        "runner_sha256": sha256(Path(__file__).resolve()),
        "core_sha256": sha256(args.core.resolve()),
        "variables": VARIABLES,
        "reference_edges": sorted(f"{a}->{b}" for a, b in REFERENCE_EDGES),
        "seasons": list(SEASONS),
        "seed": args.seed,
        "input_dir": str(args.input_dir.resolve()),
        "pattern": args.pattern,
    }
    partial_rows = args.outputs_dir / "partial_per_season_graph_recovery.csv"
    partial_inputs = args.outputs_dir / "partial_season_input_summary.csv"
    partial_diagnostics = args.outputs_dir / "partial_diagnostics.json"
    rows = (
        pd.read_csv(partial_rows).to_dict("records")
        if partial_rows.is_file()
        else []
    )
    input_rows = (
        pd.read_csv(partial_inputs).to_dict("records")
        if partial_inputs.is_file()
        else []
    )
    diagnostics = (
        json.loads(partial_diagnostics.read_text(encoding="utf-8"))
        if partial_diagnostics.is_file()
        else {}
    )
    if checkpoint_path.is_file():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("signature") != checkpoint_signature:
            raise RuntimeError(
                f"Refusing incompatible NBA baseline checkpoint {checkpoint_path}"
            )
    elif rows or input_rows or diagnostics:
        raise RuntimeError("Partial NBA baseline files exist without a checkpoint")
    completed_seasons = {
        season
        for season in SEASONS
        if sum(str(row["season"]) == season for row in rows) == 4
        and season in diagnostics
    }
    input_by_season = {str(row["season"]): row for row in input_rows}
    for season in completed_seasons:
        path = args.input_dir / args.pattern.format(season=season)
        recorded = str(input_by_season.get(season, {}).get("input_sha256", ""))
        if not path.is_file() or recorded != sha256(path):
            raise RuntimeError(
                f"NBA baseline checkpoint input changed or lacks a hash: {path}"
            )
    for season in SEASONS:
        if season in completed_seasons:
            print(f"[{season}] resume: complete", flush=True)
            continue
        path = args.input_dir / args.pattern.format(season=season)
        values = pd.read_csv(path)[VARIABLES].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any() or (values < 0).any().any():
            raise ValueError(f"invalid counts in {path}")
        data = values.to_numpy(dtype=np.int64)
        input_rows.append(
            {
                "season": season,
                "n": len(data),
                "input": str(path.resolve()),
                "input_sha256": sha256(path),
                **{f"mean_{name}": float(values[name].mean()) for name in VARIABLES},
            }
        )
        print(f"[{season}] n={len(data)}", flush=True)

        start = time.perf_counter()
        ods_estimate = core.run_poisson_dag_ods_baseline(data, seed=args.seed)
        ods_directed, ods_undirected = decode_graph_estimate(ods_estimate)
        ods_diagnostics = ods_estimate.diagnostics or {}
        rows.append(
            {
                "season": season,
                "method": "ODS",
                "n": len(data),
                "runtime_sec": time.perf_counter() - start,
                "model_applicable": True,
                "applicability_reason": "",
                **evaluate(ods_directed, ods_undirected),
            }
        )

        start = time.perf_counter()
        pc_estimate = core.run_pc_rcit_baseline(data, seed=args.seed)
        pc_directed, pc_undirected = decode_graph_estimate(pc_estimate)
        rows.append(
            {
                "season": season,
                "method": "PC (RCIT)",
                "n": len(data),
                "runtime_sec": time.perf_counter() - start,
                "model_applicable": True,
                "applicability_reason": "",
                **evaluate(pc_directed, pc_undirected),
            }
        )

        start = time.perf_counter()
        c_skeleton, c_adjacency, candidates = run_cumulant(
            core, data, args.seed
        )
        c_directed, c_undirected = decode_adjacency(c_adjacency)
        rows.append(
            {
                "season": season,
                "method": "PB-SCM (Cumulant)",
                "n": len(data),
                "runtime_sec": time.perf_counter() - start,
                "model_applicable": False,
                "applicability_reason": (
                    "Raw NBA counts violate PB-SCM binomial-thinning "
                    "constraints: the positive reference-edge coefficients "
                    "exceed 1."
                ),
                **evaluate(c_directed, c_undirected),
            }
        )

        start = time.perf_counter()
        p_skeleton, p_adjacency = run_pgf(core, data, args.seed)
        p_directed, p_undirected = decode_adjacency(p_adjacency)
        rows.append(
            {
                "season": season,
                "method": "PB-SCM (PGF)",
                "n": len(data),
                "runtime_sec": time.perf_counter() - start,
                "model_applicable": False,
                "applicability_reason": (
                    "The PGF procedure assumes the same PB-SCM "
                    "binomial-thinning model; the positive reference-edge "
                    "coefficients exceed 1."
                ),
                **evaluate(p_directed, p_undirected),
            }
        )

        diagnostics[season] = {
            "ods": ods_diagnostics,
            "cumulant_raw_skeleton": c_skeleton.tolist(),
            "cumulant_raw_adjacency": c_adjacency.tolist(),
            "cumulant_one_edge_candidates": candidates,
            "pgf_raw_skeleton": p_skeleton.tolist(),
            "pgf_raw_adjacency": p_adjacency.tolist(),
        }
        atomic_csv(pd.DataFrame(rows), partial_rows)
        atomic_csv(pd.DataFrame(input_rows), partial_inputs)
        atomic_json(diagnostics, partial_diagnostics)
        completed_now = sorted(
            {
                str(row["season"])
                for row in rows
                if sum(
                    str(candidate["season"]) == str(row["season"])
                    for candidate in rows
                ) == 4
                and str(row["season"]) in diagnostics
            },
            key=SEASONS.index,
        )
        atomic_json(
            {
                "status": "running",
                "signature": checkpoint_signature,
                "completed_seasons": completed_now,
            },
            checkpoint_path,
        )
        print(
            "  PC (RCIT):",
            format_edges(pc_directed),
            format_edges(pc_undirected, "--"),
            flush=True,
        )
        print(
            "  Cumulant:",
            format_edges(c_directed),
            format_edges(c_undirected, "--"),
            flush=True,
        )
        print(
            "  PGF:",
            format_edges(p_directed),
            format_edges(p_undirected, "--"),
            flush=True,
        )

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    atomic_csv(detail, args.outputs_dir / "per_season_graph_recovery.csv")
    atomic_csv(summary, args.outputs_dir / "method_summary.csv")
    atomic_csv(pd.DataFrame(input_rows), args.outputs_dir / "season_input_summary.csv")
    atomic_json(diagnostics, args.outputs_dir / "pbscm_raw_diagnostics.json")
    atomic_json(
        {
            "variables": VARIABLES,
            "reference_edges": sorted(
                f"{a}->{b}" for a, b in REFERENCE_EDGES
            ),
            "seed": args.seed,
            "pc_rcit_configuration": {
                "alpha": 0.05,
                "stable": True,
                "approx": "lpd4",
                "num_f": 100,
                "num_f2": 5,
                "rcit": True,
            },
            "cumulant_source": str(
                (args.core.resolve().parent / "external/PBSCM").resolve()
            ),
            "pgf_source": str(
                (
                    args.core.resolve().parent / "external/PBSCM_PGF"
                ).resolve()
            ),
            "undirected_edge_policy": (
                "counts toward skeleton recovery but not directed recovery"
            ),
            "historical_results_consumed": False,
        },
        args.outputs_dir / "metadata.json",
    )
    atomic_json(
        {
            "status": "complete",
            "signature": checkpoint_signature,
            "completed_seasons": list(SEASONS),
        },
        checkpoint_path,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
