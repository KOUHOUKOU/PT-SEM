"""Run the paper-final adaptive-family PT-SEM on the NBA real-world data.

This driver deliberately imports the validated likelihood and order-DP core
from BEST_PTSEM/d.py without modifying it.  It supports two local estimators:

* moment: the exact plug-in moment/BIC procedure used by the final simulation;
* optimization: joint constrained maximum likelihood, initialized by moments.

Every node/parent-set is scored against the paper's six-family library:
Poisson, NB, ZIP, Geom, Binomial, and Bernoulli.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, xlogy

REPO_ROOT = Path(__file__).resolve().parents[3]


FAMILIES = ("Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli")
SEASONS = ("2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21")
DATASET_SPECS = {
    "three_team": {
        "variables": ("FOUL", "FTA", "FTM"),
        "pattern": "data/processed/quarter_counts_{season}_drawn.csv",
        "unit": "team-quarter",
    },
    "five_team": {
        "variables": ("FOUL", "FTA", "FTM", "MISS_FG", "REB"),
        "pattern": (
            "outputs/rebound_control_study/"
            "quarter_counts_{season}_drawn_missfg_reb.csv"
        ),
        "unit": "team-quarter",
    },
    "four_team": {
        "variables": ("FOUL", "FTA", "FTM", "MISS_FG"),
        "pattern": (
            "outputs/rebound_control_study/"
            "quarter_counts_{season}_drawn_missfg_reb.csv"
        ),
        "unit": "team-quarter",
    },
    "five_game": {
        "variables": ("FOUL", "FTA", "FTM", "MISS_FG", "REB"),
        "pattern": (
            "outputs/rebound_control_study/game-quarter/"
            "game-quarter_counts_{season}_drawn_missfg_reb.csv"
        ),
        "unit": "game-quarter",
    },
    "four_game": {
        "variables": ("FOUL", "FTA", "FTM", "MISS_FG"),
        "pattern": (
            "outputs/rebound_control_study/game-quarter/"
            "game-quarter_counts_{season}_drawn_missfg_reb.csv"
        ),
        "unit": "game-quarter",
    },
}
EXPECTED_DAG_COUNTS = {3: 25, 4: 543, 5: 29281}
EPS = 1e-10


@dataclass
class FamilyFit:
    family: str
    bic: float
    loglik: float
    alpha: list[float]
    params: dict[str, float | int]
    parameter_count: int
    success: bool
    status: str
    nit: int
    nfev: int
    starts: int
    moment_loglik: float
    loglik_improvement: float
    runtime_sec: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_paper_core(core_path: Path):
    os.environ["PTSEM_FAMILY_LIBRARY"] = ",".join(FAMILIES)
    os.environ["PTSEM_FAMILY_ASSIGNMENT"] = "iid_uniform"
    module_name = "best_ptsem_paper_core"
    spec = importlib.util.spec_from_file_location(module_name, core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import PT-SEM core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if tuple(module.FAMLIB) != FAMILIES:
        raise RuntimeError(f"paper family library mismatch: {module.FAMLIB}")
    return module


def validate_frame(frame: pd.DataFrame, variables: Sequence[str], path: Path) -> None:
    missing = [name for name in variables if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing variables: {missing}")
    values = frame[list(variables)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"{path} contains nonnumeric values")
    if (values < 0).any().any():
        raise ValueError(f"{path} contains negative counts")
    if not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
        raise ValueError(f"{path} contains noninteger counts")
    if "FTA" in variables and "FTM" in variables and (values["FTM"] > values["FTA"]).any():
        raise ValueError(f"{path} violates FTM <= FTA")
    keys = [name for name in ("game_id", "period", "team_id") if name in frame.columns]
    if keys and frame.duplicated(keys).any():
        raise ValueError(f"{path} contains duplicate rows for {keys}")


def load_dataset(
    root: Path,
    dataset: str,
    season: str,
) -> tuple[np.ndarray, pd.DataFrame, tuple[str, ...], Path]:
    spec = DATASET_SPECS[dataset]
    variables = tuple(spec["variables"])
    path = root / str(spec["pattern"]).format(season=season)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    validate_frame(frame, variables, path)
    data = frame[list(variables)].to_numpy(dtype=np.int64)
    return data, frame, variables, path


def parents_from_mask(mask: int, d: int) -> list[int]:
    return [node for node in range(d) if (mask >> node) & 1]


def family_dimension(family: str) -> int:
    return 2 if family in ("NB", "ZIP", "Binomial") else 1


def exact_convolution_loglik(
    core,
    observations: np.ndarray,
    thinning_mean: np.ndarray,
    family: str,
    params: dict[str, float | int],
) -> float:
    """Use the paper likelihood with a stable vectorized Geometric path.

    The paper core's Geometric closed form is fast in the interior but falls
    back observation-by-observation near p=1.  The grouped convolution below
    is mathematically identical and avoids that optimization-boundary cost.
    """
    if family != "Geom":
        return float(
            core.convolution_loglik(
                observations, thinning_mean, family, params
            )
        )
    y = np.asarray(observations, dtype=int)
    theta = np.asarray(thinning_mean, dtype=float)
    probability = float(np.clip(params["p"], 1e-12, 1.0 - 1e-12))
    total = 0.0
    for observed in np.unique(y):
        indices = np.flatnonzero(y == observed)
        epsilon = np.arange(observed + 1, dtype=int)
        poisson_count = observed - epsilon
        local_theta = theta[indices]
        log_poisson = (
            xlogy(poisson_count[:, None], local_theta[None, :])
            - local_theta[None, :]
            - gammaln(poisson_count[:, None] + 1.0)
        )
        log_epsilon = (
            math.log(probability)
            + epsilon * math.log1p(-probability)
        )
        total += float(
            logsumexp(log_poisson + log_epsilon[:, None], axis=0).sum()
        )
    return total


def residual_moments_for_alpha(
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parents: Sequence[int],
    alpha: np.ndarray,
) -> tuple[float, float]:
    if not parents:
        return (
            max(float(mean[child]), EPS),
            max(float(covariance[child, child]), EPS),
        )
    parent_mean = mean[list(parents)]
    parent_cov = covariance[np.ix_(parents, parents)]
    residual_mean = float(mean[child] - alpha @ parent_mean)
    residual_variance = float(
        covariance[child, child]
        - alpha @ parent_mean
        - alpha @ parent_cov @ alpha
    )
    return max(residual_mean, EPS), max(residual_variance, EPS)


def moment_family_fit(
    core,
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parent_mask: int,
    family: str,
) -> FamilyFit:
    started = time.perf_counter()
    parents = parents_from_mask(parent_mask, data.shape[1])
    alpha, thinning_mean, residual_mean, residual_variance = (
        core.estimate_alpha_and_residual_moments(
            data, mean, covariance, child, parent_mask
        )
    )
    params, dimension = core.invert_moments(
        family, residual_mean, residual_variance
    )
    candidates = [params]
    if family == "Binomial":
        center = int(params["n"])
        trials = sorted(
            {
                center,
                min(core.BINOMIAL_N_MAX, center + 1),
            }
        )
        candidates = [
            {
                "n": int(n_trials),
                "p": float(
                    np.clip(
                        residual_mean / n_trials,
                        EPS,
                        1.0 - EPS,
                    )
                ),
            }
            for n_trials in trials
        ]
    best_loglik = -math.inf
    best_params: dict[str, float | int] = {}
    for candidate in candidates:
        loglik = exact_convolution_loglik(
            core,
            data[:, child], thinning_mean, family, candidate
        )
        if np.isfinite(loglik) and loglik > best_loglik:
            best_loglik = float(loglik)
            best_params = {
                key: int(value) if key == "n" else float(value)
                for key, value in candidate.items()
            }
    parameter_count = len(parents) + dimension
    bic = (
        -2.0 * best_loglik + parameter_count * math.log(len(data))
        if np.isfinite(best_loglik)
        else math.inf
    )
    return FamilyFit(
        family=family,
        bic=float(bic),
        loglik=float(best_loglik),
        alpha=np.asarray(alpha, dtype=float).tolist(),
        params=best_params,
        parameter_count=parameter_count,
        success=bool(np.isfinite(best_loglik)),
        status="moment",
        nit=0,
        nfev=1,
        starts=1,
        moment_loglik=float(best_loglik),
        loglik_improvement=0.0,
        runtime_sec=time.perf_counter() - started,
    )


def logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-9, 1.0 - 1e-9))
    return math.log(clipped / (1.0 - clipped))


def expit(value: float) -> float:
    if value >= 0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def transformed_family_values(
    family: str,
    params: dict[str, float | int],
) -> tuple[list[float], list[tuple[float | None, float | None]]]:
    if family == "Poisson":
        return [math.log(max(float(params["lam"]), 1e-12))], [(-27.0, 20.0)]
    if family == "NB":
        return [
            math.log(max(float(params["r"]), 1e-8)),
            logit(float(params["p"])),
        ], [(-18.0, math.log(1e8)), (-16.0, 16.0)]
    if family == "ZIP":
        return [
            math.log(max(float(params["lam"]), 1e-12)),
            logit(float(params["rho"])),
        ], [(-27.0, 20.0), (-20.0, 16.0)]
    if family in ("Geom", "Bernoulli", "Binomial"):
        return [logit(float(params["p"]))], [(-16.0, 16.0)]
    raise ValueError(family)


def unpack_family_values(
    family: str,
    values: Sequence[float],
    fixed_trials: int | None,
) -> dict[str, float | int]:
    if family == "Poisson":
        return {"lam": math.exp(float(values[0]))}
    if family == "NB":
        return {
            "r": math.exp(float(values[0])),
            "p": expit(float(values[1])),
        }
    if family == "ZIP":
        return {
            "lam": math.exp(float(values[0])),
            "rho": expit(float(values[1])),
        }
    if family == "Geom":
        return {"p": expit(float(values[0]))}
    if family == "Bernoulli":
        return {"p": expit(float(values[0]))}
    if family == "Binomial":
        if fixed_trials is None:
            raise ValueError("Binomial optimization requires fixed trials")
        return {"n": int(fixed_trials), "p": expit(float(values[0]))}
    raise ValueError(family)


def optimization_starts(
    core,
    family: str,
    alpha_moment: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parents: Sequence[int],
    fixed_trials: int | None,
    count: int,
    alpha_upper: float,
) -> list[tuple[np.ndarray, list[tuple[float | None, float | None]]]]:
    factors = (1.0, 0.65, 1.35, 0.25, 1.75)
    starts = []
    for factor in factors[: max(1, count)]:
        alpha = np.clip(alpha_moment * factor, 0.0, alpha_upper)
        residual_mean, residual_variance = residual_moments_for_alpha(
            mean, covariance, child, parents, alpha
        )
        params, _ = core.invert_moments(
            family, residual_mean, residual_variance
        )
        if family == "Binomial":
            if fixed_trials is None:
                raise ValueError("missing Binomial trial count")
            params = {
                "n": int(fixed_trials),
                "p": float(
                    np.clip(
                        residual_mean / fixed_trials,
                        1e-9,
                        1.0 - 1e-9,
                    )
                ),
            }
        family_values, family_bounds = transformed_family_values(family, params)
        vector = np.r_[alpha, family_values].astype(float)
        bounds = [(0.0, alpha_upper)] * len(parents) + family_bounds
        starts.append((vector, bounds))
    return starts


def optimize_family_for_trials(
    core,
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parent_mask: int,
    family: str,
    fixed_trials: int | None,
    starts_count: int,
    maxiter: int,
) -> dict[str, Any]:
    parents = parents_from_mask(parent_mask, data.shape[1])
    alpha_moment, _, _, _ = core.estimate_alpha_and_residual_moments(
        data, mean, covariance, child, parent_mask
    )
    parent_values = (
        data[:, parents].astype(float)
        if parents
        else np.zeros((len(data), 0), dtype=float)
    )
    positive_parent_means = mean[list(parents)][mean[list(parents)] > EPS] if parents else []
    ratio_bound = (
        (float(data[:, child].max()) + 10.0) / float(np.min(positive_parent_means))
        if len(positive_parent_means)
        else 10.0
    )
    alpha_upper = float(np.clip(max(10.0, 3.0 * ratio_bound), 10.0, 100.0))
    starts = optimization_starts(
        core,
        family,
        np.asarray(alpha_moment, dtype=float),
        mean,
        covariance,
        child,
        parents,
        fixed_trials,
        starts_count,
        alpha_upper,
    )

    def evaluate(vector: np.ndarray) -> float:
        alpha = vector[: len(parents)]
        params = unpack_family_values(
            family, vector[len(parents) :], fixed_trials
        )
        thinning_mean = parent_values @ alpha if parents else np.zeros(len(data))
        try:
            loglik = exact_convolution_loglik(
                core, data[:, child], thinning_mean, family, params
            )
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        return -float(loglik) if np.isfinite(loglik) else 1e100

    evaluated_starts = [(float(evaluate(vector)), vector, bounds) for vector, bounds in starts]
    baseline_value, baseline_vector, _ = min(evaluated_starts, key=lambda item: item[0])
    best_value = baseline_value
    best_vector = baseline_vector.copy()
    best_result = None
    total_nfev = len(evaluated_starts)
    total_nit = 0
    for _, vector, bounds in evaluated_starts:
        result = minimize(
            evaluate,
            vector,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "ftol": 1e-10,
                "gtol": 1e-6,
                "maxls": 30,
            },
        )
        total_nfev += int(getattr(result, "nfev", 0))
        total_nit += int(getattr(result, "nit", 0))
        value = float(result.fun)
        if np.isfinite(value) and value < best_value:
            best_value = value
            best_vector = np.asarray(result.x, dtype=float)
            best_result = result
    return {
        "loglik": -float(best_value) if best_value < 1e99 else -math.inf,
        "moment_loglik": (
            -float(baseline_value) if baseline_value < 1e99 else -math.inf
        ),
        "alpha": best_vector[: len(parents)].tolist(),
        "params": unpack_family_values(
            family, best_vector[len(parents) :], fixed_trials
        ),
        "success": bool(best_result is not None and best_result.success),
        "status": (
            str(best_result.message)
            if best_result is not None
            else "moment_start_retained"
        ),
        "nit": total_nit,
        "nfev": total_nfev,
        "starts": len(starts),
    }


def optimization_family_fit(
    core,
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parent_mask: int,
    family: str,
    starts_count: int,
    maxiter: int,
    optimization_workers: int,
) -> FamilyFit:
    started = time.perf_counter()
    parents = parents_from_mask(parent_mask, data.shape[1])
    if family == "Bernoulli" and not parents and int(data[:, child].max()) > 1:
        return FamilyFit(
            family, math.inf, -math.inf, [], {}, len(parents) + 1,
            False, "infeasible_root_support", 0, 0, 0, -math.inf, 0.0,
            time.perf_counter() - started,
        )
    trial_values: Iterable[int | None]
    if family == "Binomial":
        if not parents:
            minimum = max(core.BINOMIAL_N_MIN, int(data[:, child].max()))
            trial_values = range(minimum, core.BINOMIAL_N_MAX + 1)
        else:
            trial_values = range(core.BINOMIAL_N_MIN, core.BINOMIAL_N_MAX + 1)
    else:
        trial_values = (None,)
    trial_values = list(trial_values)

    def run_trials(trials: int | None) -> dict[str, Any]:
        return optimize_family_for_trials(
                core,
                data,
                mean,
                covariance,
                child,
                parent_mask,
                family,
                trials,
                starts_count,
                maxiter,
            )

    if family == "Binomial" and len(trial_values) > 1 and optimization_workers > 1:
        with ThreadPoolExecutor(
            max_workers=min(optimization_workers, len(trial_values))
        ) as executor:
            candidates = list(executor.map(run_trials, trial_values))
    else:
        candidates = [run_trials(trials) for trials in trial_values]
    if not candidates:
        return FamilyFit(
            family, math.inf, -math.inf, [], {}, len(parents) + family_dimension(family),
            False, "infeasible_root_support", 0, 0, 0, -math.inf, 0.0,
            time.perf_counter() - started,
        )
    best = max(candidates, key=lambda item: item["loglik"])
    best_moment_loglik = max(item["moment_loglik"] for item in candidates)
    dimension = family_dimension(family)
    parameter_count = len(parents) + dimension
    loglik = float(best["loglik"])
    bic = (
        -2.0 * loglik + parameter_count * math.log(len(data))
        if np.isfinite(loglik)
        else math.inf
    )
    improvement = (
        loglik - best_moment_loglik
        if np.isfinite(loglik) and np.isfinite(best_moment_loglik)
        else 0.0
    )
    if improvement < -1e-7:
        raise RuntimeError(
            f"optimization degraded likelihood by {improvement} for "
            f"child={child}, parents={parent_mask}, family={family}"
        )
    return FamilyFit(
        family=family,
        bic=float(bic),
        loglik=loglik,
        alpha=[float(value) for value in best["alpha"]],
        params={
            key: int(value) if key == "n" else float(value)
            for key, value in best["params"].items()
        },
        parameter_count=parameter_count,
        success=bool(best["success"]),
        status=str(best["status"]),
        nit=int(sum(item["nit"] for item in candidates)),
        nfev=int(sum(item["nfev"] for item in candidates)),
        starts=int(sum(item["starts"] for item in candidates)),
        moment_loglik=float(best_moment_loglik),
        loglik_improvement=float(max(0.0, improvement)),
        runtime_sec=time.perf_counter() - started,
    )


def top_dags(
    scores: list[list[float]],
    max_parents: int,
    top_k: int = 10,
) -> tuple[int, list[tuple[float, tuple[int, ...]]]]:
    d = len(scores)
    unique: dict[tuple[int, ...], float] = {}
    nodes = tuple(range(d))
    for order in itertools.permutations(nodes):
        options = []
        predecessors: list[int] = []
        for child in order:
            masks = []
            for size in range(min(max_parents, len(predecessors)) + 1):
                for subset in itertools.combinations(predecessors, size):
                    mask = sum(1 << parent for parent in subset)
                    if np.isfinite(scores[child][mask]):
                        masks.append(mask)
            options.append(masks)
            predecessors.append(child)
        for ordered_masks in itertools.product(*options):
            masks_by_node = [0] * d
            for child, mask in zip(order, ordered_masks):
                masks_by_node[child] = mask
            key = tuple(masks_by_node)
            if key not in unique:
                unique[key] = float(
                    sum(scores[child][key[child]] for child in nodes)
                )
    ranked = sorted((score, masks) for masks, score in unique.items())[:top_k]
    return len(unique), ranked


def format_edges(parent_masks: Sequence[int], variables: Sequence[str]) -> str:
    edges = []
    for child, mask in enumerate(parent_masks):
        for parent in parents_from_mask(mask, len(variables)):
            edges.append(f"{variables[parent]}->{variables[child]}")
    return ", ".join(sorted(edges)) if edges else "(none)"


def run_one(
    core,
    workspace: Path,
    outputs: Path,
    dataset: str,
    season: str,
    estimator: str,
    starts: int,
    maxiter: int,
    optimization_workers: int,
    force: bool,
    core_hash: str,
) -> dict[str, Any]:
    cell_dir = outputs / dataset / season / estimator
    graph_path = cell_dir / "graph_result.json"
    if graph_path.exists() and not force:
        cached = json.loads(graph_path.read_text(encoding="utf-8"))
        if (
            cached.get("core_sha256") == core_hash
            and cached.get("optimization_starts") == starts
            and cached.get("optimization_maxiter") == maxiter
        ):
            print(f"[reuse] {dataset} {season} {estimator}", flush=True)
            return cached

    data, frame, variables, input_path = load_dataset(workspace, dataset, season)
    mean, covariance = core.empirical_moments(data)
    d = data.shape[1]
    max_parents = d - 1
    mask_count = 1 << d
    scores = [[math.inf] * mask_count for _ in range(d)]
    selected: list[list[FamilyFit | None]] = [
        [None] * mask_count for _ in range(d)
    ]
    rows = []
    started = time.perf_counter()
    print(
        f"[fit] {dataset} {season} {estimator} n={len(data)} d={d}",
        flush=True,
    )
    for child in range(d):
        for parent_mask in range(mask_count):
            if (parent_mask >> child) & 1:
                continue
            family_fits = []
            for family in FAMILIES:
                if estimator == "moment":
                    fit = moment_family_fit(
                        core, data, mean, covariance, child, parent_mask, family
                    )
                elif estimator == "optimization":
                    fit = optimization_family_fit(
                        core,
                        data,
                        mean,
                        covariance,
                        child,
                        parent_mask,
                        family,
                        starts,
                        maxiter,
                        optimization_workers,
                    )
                else:
                    raise ValueError(estimator)
                family_fits.append(fit)
                row = {
                    "dataset": dataset,
                    "unit": DATASET_SPECS[dataset]["unit"],
                    "season": season,
                    "n": len(data),
                    "estimator": estimator,
                    "child": variables[child],
                    "child_index": child,
                    "parent_set": ";".join(
                        variables[node]
                        for node in parents_from_mask(parent_mask, d)
                    ),
                    "parent_mask": parent_mask,
                    **asdict(fit),
                }
                row["alpha"] = json.dumps(row["alpha"])
                row["params"] = json.dumps(row["params"], sort_keys=True)
                rows.append(row)
            best = min(family_fits, key=lambda item: item.bic)
            if not np.isfinite(best.bic):
                raise RuntimeError(
                    f"no finite family score for {dataset} {season} "
                    f"{variables[child]} parents={parent_mask}"
                )
            scores[child][parent_mask] = best.bic
            selected[child][parent_mask] = best
        print(
            f"  child {variables[child]} complete "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    parent_masks_dict, dp_score = core.exact_order_dp(scores)
    parent_masks = tuple(parent_masks_dict[node] for node in range(d))
    dag_count, ranked = top_dags(scores, max_parents, top_k=10)
    expected = EXPECTED_DAG_COUNTS[d]
    if dag_count != expected:
        raise RuntimeError(f"expected {expected} DAGs, obtained {dag_count}")
    if abs(ranked[0][0] - dp_score) > 1e-7:
        raise RuntimeError(
            f"DP/enumeration score mismatch: {dp_score} vs {ranked[0][0]}"
        )
    if ranked[0][1] != parent_masks:
        parent_masks = ranked[0][1]

    node_fits = [selected[node][parent_masks[node]] for node in range(d)]
    if any(fit is None for fit in node_fits):
        raise RuntimeError("selected graph has a missing local fit")
    alpha_matrix = np.zeros((d, d), dtype=float)
    node_rows = []
    for child, fit in enumerate(node_fits):
        assert fit is not None
        parents = parents_from_mask(parent_masks[child], d)
        for position, parent in enumerate(parents):
            alpha_matrix[child, parent] = fit.alpha[position]
        node_rows.append(
            {
                "node": variables[child],
                "parent_set": [variables[parent] for parent in parents],
                "family": fit.family,
                "params": fit.params,
                "alpha": {
                    variables[parent]: fit.alpha[position]
                    for position, parent in enumerate(parents)
                },
                "local_bic": fit.bic,
                "local_loglik": fit.loglik,
                "optimizer_success": fit.success,
                "optimizer_status": fit.status,
            }
        )

    runtime = time.perf_counter() - started
    result = {
        "created_utc": utc_now(),
        "dataset": dataset,
        "unit": DATASET_SPECS[dataset]["unit"],
        "season": season,
        "n": len(data),
        "variables": list(variables),
        "means": {
            variables[index]: float(mean[index]) for index in range(d)
        },
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "estimator": estimator,
        "family_library": list(FAMILIES),
        "max_parents": max_parents,
        "score": float(ranked[0][0]),
        "second_score": float(ranked[1][0]),
        "bic_delta_second": float(ranked[1][0] - ranked[0][0]),
        "edges": format_edges(parent_masks, variables),
        "parent_masks": list(parent_masks),
        "selected_families": [fit.family for fit in node_fits if fit is not None],
        "node_fits": node_rows,
        "alpha_matrix": alpha_matrix.tolist(),
        "dag_count": dag_count,
        "top_dags": [
            {
                "rank": rank,
                "score": float(score),
                "delta": float(score - ranked[0][0]),
                "edges": format_edges(masks, variables),
                "parent_masks": list(masks),
            }
            for rank, (score, masks) in enumerate(ranked, start=1)
        ],
        "runtime_sec": runtime,
        "core_path": str(Path(core.__file__).resolve()),
        "core_sha256": core_hash,
        "optimization_starts": starts,
        "optimization_maxiter": maxiter,
        "optimization_workers": optimization_workers,
    }
    atomic_write_csv(pd.DataFrame(rows), cell_dir / "local_family_fits.csv")
    atomic_write_json(graph_path, result)
    print(
        f"[done] {dataset} {season} {estimator} "
        f"{result['edges']} ({runtime:.1f}s)",
        flush=True,
    )
    return result


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: result[key]
        for key in (
            "dataset",
            "unit",
            "season",
            "n",
            "estimator",
            "score",
            "second_score",
            "bic_delta_second",
            "edges",
            "dag_count",
            "runtime_sec",
            "input",
            "input_sha256",
            "core_sha256",
        )
    }
    row["variables"] = ";".join(result["variables"])
    row["selected_families"] = ";".join(result["selected_families"])
    row["parent_masks"] = "|".join(map(str, result["parent_masks"]))
    for node_fit in result["node_fits"]:
        node = node_fit["node"]
        row[f"family_{node}"] = node_fit["family"]
        row[f"parents_{node}"] = ";".join(node_fit["parent_set"])
        row[f"alpha_{node}"] = json.dumps(node_fit["alpha"], sort_keys=True)
        row[f"params_{node}"] = json.dumps(node_fit["params"], sort_keys=True)
    return row


def collect_saved_results(outputs_dir: Path) -> list[dict[str, Any]]:
    saved = []
    for path in sorted(outputs_dir.glob("*/*/*/graph_result.json")):
        try:
            saved.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return saved


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    core_path: Path,
    core_hash: str,
    status: str,
    error: str | None = None,
) -> None:
    payload = {
        "status": status,
        "updated_utc": utc_now(),
        "command": sys.argv,
        "workspace": str(Path(args.workspace).resolve()),
        "outputs_dir": str(Path(args.outputs_dir).resolve()),
        "core_path": str(core_path),
        "core_sha256": core_hash,
        "datasets": args.datasets,
        "seasons": args.seasons,
        "estimators": args.estimators,
        "optimization_starts": args.optimization_starts,
        "optimization_maxiter": args.optimization_maxiter,
        "optimization_workers": args.optimization_workers,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    if error:
        payload["error"] = error
    atomic_write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--core-path",
        type=Path,
        default=REPO_ROOT / "experiments/simulation/legacy/d.py",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/adaptive_ptsem_real_study"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        choices=SEASONS,
        default=list(SEASONS),
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=("moment", "optimization"),
        default=["moment", "optimization"],
    )
    parser.add_argument("--optimization-starts", type=int, default=2)
    parser.add_argument("--optimization-maxiter", type=int, default=250)
    parser.add_argument("--optimization-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.workspace = args.workspace.resolve()
    if not args.outputs_dir.is_absolute():
        args.outputs_dir = args.workspace / args.outputs_dir
    args.outputs_dir = args.outputs_dir.resolve()
    core_path = args.core_path.resolve()
    if not core_path.exists():
        raise FileNotFoundError(core_path)
    core_hash = file_sha256(core_path)
    core = load_paper_core(core_path)
    manifest_path = args.outputs_dir / "run_metadata.json"
    write_run_metadata(
        manifest_path, args, core_path, core_hash, status="running"
    )
    results = []
    try:
        for dataset in args.datasets:
            for season in args.seasons:
                for estimator in args.estimators:
                    results.append(
                        run_one(
                            core,
                            args.workspace,
                            args.outputs_dir,
                            dataset,
                            season,
                            estimator,
                            args.optimization_starts,
                            args.optimization_maxiter,
                            args.optimization_workers,
                            args.force,
                            core_hash,
                        )
                    )
                    atomic_write_csv(
                        pd.DataFrame([flatten_result(item) for item in results]),
                        args.outputs_dir / "partial_results.csv",
                    )
        all_saved = collect_saved_results(args.outputs_dir)
        summary = pd.DataFrame(
            [flatten_result(item) for item in all_saved]
        ).sort_values(["dataset", "season", "estimator"])
        atomic_write_csv(summary, args.outputs_dir / "adaptive_ptsem_results.csv")
        write_run_metadata(
            manifest_path, args, core_path, core_hash, status="complete"
        )
        print(
            f"All adaptive PT-SEM real-data results written to "
            f"{args.outputs_dir}",
            flush=True,
        )
    except BaseException as exc:
        write_run_metadata(
            manifest_path,
            args,
            core_path,
            core_hash,
            status="failed",
            error=repr(exc),
        )
        raise


if __name__ == "__main__":
    main()
