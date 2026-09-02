"""Run the paper-final adaptive-family PT-SEM on the NBA real-world data.

This driver deliberately imports the validated likelihood and order-DP core
from ``src/nba_core.py`` without copying it.  It supports two local estimators:

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


BLAS_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _blas_variable in BLAS_VARIABLES:
    # Set before NumPy/SciPy import so Windows-spawned profile workers inherit
    # a single BLAS thread and cannot oversubscribe the process pool.
    os.environ[_blas_variable] = "1"

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy

REPO_ROOT = Path(__file__).resolve().parents[3]


FAMILIES = ("Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli")
SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2025))
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
MLE_SOLVER_SPEC = "certified_mle_v1"
NBA_EXECUTION_SPEC = "profile_process_pool_v1"
NBA_BINOMIAL_N_MIN = 2
NBA_BINOMIAL_N_MAX = 100


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
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class LocalProfileTask:
    child: int
    parent_mask: int
    family: str


@dataclass
class LocalProfileExecution:
    task: LocalProfileTask
    fit: FamilyFit
    process_id: int
    started_ns: int
    completed_ns: int
    blas_thread_values: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_contract(core) -> dict[str, Any]:
    """Return the truth-independent estimator contract used by NBA."""
    contract = {
        "candidate_families": list(FAMILIES),
        "estimator_binomial_n_min": int(core.ESTIMATOR_BINOMIAL_N_MIN),
        "estimator_binomial_n_max": int(core.ESTIMATOR_BINOMIAL_N_MAX),
        "nba_binomial_n_min": NBA_BINOMIAL_N_MIN,
        "nba_binomial_n_max": NBA_BINOMIAL_N_MAX,
        "q_poisson": int(core.family_dimension("Poisson")),
        "q_binomial": int(core.family_dimension("Binomial")),
        "binomial_discrete_rule": "single H_Binomial index; no likelihood profiling",
    }
    expected = {
        "candidate_families": list(FAMILIES),
        "estimator_binomial_n_min": 2,
        "estimator_binomial_n_max": 50,
        "nba_binomial_n_min": 2,
        "nba_binomial_n_max": 100,
        "q_poisson": 1,
        "q_binomial": 1,
        "binomial_discrete_rule": "single H_Binomial index; no likelihood profiling",
    }
    if contract != expected:
        raise RuntimeError(f"NBA method contract mismatch: {contract}")
    return contract


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
    if module.NUMERICAL_CORE_VERSION != "ptsem_submission_corrected_v3":
        raise RuntimeError(
            f"NBA requires the recorded numerical core version, got "
            f"{module.NUMERICAL_CORE_VERSION}"
        )
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


def exact_convolution_loglik(
    core,
    observations: np.ndarray,
    thinning_mean: np.ndarray,
    family: str,
    params: dict[str, float | int],
) -> float:
    """Delegate every NBA probability calculation to the canonical core."""
    return float(
        core.convolution_loglik(
            observations,
            thinning_mean,
            family,
            params,
            binomial_n_max=NBA_BINOMIAL_N_MAX,
        )
    )


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
    try:
        params, dimension = core.invert_moments(
            family, residual_mean, residual_variance
        )
    except core.InvalidMomentInversion as exc:
        return FamilyFit(
            family, math.inf, -math.inf, [], {},
            len(parents) + core.family_dimension(family), False,
            f"invalid_moment:{exc}", 0, 0, 0, -math.inf, 0.0,
            time.perf_counter() - started,
        )
    best_loglik = exact_convolution_loglik(
        core, data[:, child], thinning_mean, family, params
    )
    best_params = {
        key: int(value) if key == "n" else float(value)
        for key, value in params.items()
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
        return [math.log(max(float(params["lam"]), 1e-12))], [(None, None)]
    if family == "NB":
        return [
            math.log(max(float(params["r"]), 1e-8)),
            logit(float(params["p"])),
        ], [(None, None), (None, None)]
    if family == "ZIP":
        return [
            math.log(max(float(params["lam"]), 1e-12)),
            logit(float(params["rho"])),
        ], [(None, None), (None, None)]
    if family in ("Geom", "Bernoulli", "Binomial"):
        return [logit(float(params["p"]))], [(None, None)]
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
) -> list[tuple[np.ndarray, list[tuple[float | None, float | None]]]]:
    factors = (1.0, 0.65, 1.35, 0.25, 1.75)
    starts = []
    for factor in factors[: max(1, count)]:
        alpha = np.maximum(alpha_moment * factor, 0.0)
        residual_mean, residual_variance = residual_moments_for_alpha(
            mean, covariance, child, parents, alpha
        )
        try:
            params, _ = core.invert_moments(
                family,
                residual_mean,
                residual_variance,
                binomial_n_max=NBA_BINOMIAL_N_MAX,
            )
        except core.InvalidMomentInversion:
            # Initialization only: the optimized parameter space remains
            # unchanged, and no DGP bound enters these deterministic starts.
            local_mean = max(float(residual_mean), EPS)
            local_variance = max(float(residual_variance), EPS)
            if family == "Poisson":
                params = {"lam": local_mean}
            elif family == "Geom":
                params = {"p": 1.0 / (1.0 + local_mean)}
            elif family == "NB":
                size = max(local_mean, 1.0)
                params = {"r": size, "p": size / (size + local_mean)}
            elif family == "ZIP":
                rho = (
                    max(0.0, (local_variance - local_mean))
                    / max(
                        local_variance - local_mean + local_mean * local_mean,
                        EPS,
                    )
                )
                rho = min(rho, 1.0 - 1e-9)
                params = {
                    "lam": local_mean / (1.0 - rho),
                    "rho": rho,
                }
            elif family == "Bernoulli":
                params = {"p": float(np.clip(local_mean, 1e-9, 1.0 - 1e-9))}
            elif family == "Binomial":
                if fixed_trials is None:
                    raise ValueError("missing Binomial trial count")
                params = {
                    "n": int(fixed_trials),
                    "p": float(
                        np.clip(local_mean / fixed_trials, 1e-9, 1.0 - 1e-9)
                    ),
                }
            else:
                raise ValueError(family)
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
        bounds = [(0.0, None)] * len(parents) + family_bounds
        starts.append((vector, bounds))
    return starts


def objective_certification_tolerance(reference: float) -> float:
    """Tight scale-aware tolerance used only for solver certification."""
    return 1e-8 + 1e-10 * max(1.0, abs(float(reference)))


def optimizer_result_diagnostic(method: str, result: Any) -> dict[str, Any]:
    diagnostic = {
        "method": method,
        "success": bool(getattr(result, "success", False)),
        "status": int(getattr(result, "status", -1)),
        "message": str(getattr(result, "message", "")),
        "fun": float(getattr(result, "fun", math.inf)),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
    }
    if hasattr(result, "x"):
        diagnostic["x"] = np.asarray(result.x, dtype=float).tolist()
    if hasattr(result, "jac"):
        diagnostic["jac"] = np.asarray(result.jac, dtype=float).tolist()
    return diagnostic


def analytic_root_candidate(
    core,
    observations: np.ndarray,
    family: str,
    fixed_trials: int | None,
) -> dict[str, Any]:
    """Return a certified closed-form root MLE for supported families."""
    observations = np.asarray(observations, dtype=int)
    sample_mean = float(np.mean(observations))
    if family == "Poisson":
        params: dict[str, float | int] = {"lam": sample_mean}
    elif family == "Geom":
        params = {"p": 1.0 / (1.0 + sample_mean)}
    elif family == "Bernoulli":
        if observations.size and int(observations.max()) > 1:
            return {
                "loglik": -math.inf,
                "moment_loglik": -math.inf,
                "alpha": [],
                "params": {},
                "success": False,
                "status": "infeasible_root_support",
                "nit": 0,
                "nfev": 0,
                "starts": 0,
                "fixed_trials": None,
                "diagnostics": {"solver_path": "analytic_root"},
            }
        params = {"p": sample_mean}
    elif family == "Binomial":
        if fixed_trials is None:
            raise ValueError("Binomial analytic root requires fixed trials")
        if observations.size and int(observations.max()) > fixed_trials:
            return {
                "loglik": -math.inf,
                "moment_loglik": -math.inf,
                "alpha": [],
                "params": {"n": int(fixed_trials)},
                "success": False,
                "status": "infeasible_root_support",
                "nit": 0,
                "nfev": 0,
                "starts": 0,
                "fixed_trials": int(fixed_trials),
                "diagnostics": {"solver_path": "analytic_root"},
            }
        params = {
            "n": int(fixed_trials),
            "p": sample_mean / float(fixed_trials),
        }
    else:
        raise ValueError(f"no analytic root MLE registered for {family}")

    try:
        loglik = exact_convolution_loglik(
            core,
            observations,
            np.zeros(len(observations), dtype=float),
            family,
            params,
        )
    except (FloatingPointError, OverflowError, ValueError):
        # The sample can put an MLE on a boundary excluded by the current
        # declared parameter space (for example p=1 for Geometric).  Such a
        # candidate is support/parameter-infeasible; it is not optimizer failure.
        return {
            "loglik": -math.inf,
            "moment_loglik": -math.inf,
            "alpha": [],
            "params": params,
            "success": False,
            "status": "infeasible_root_support",
            "nit": 0,
            "nfev": 0,
            "starts": 0,
            "fixed_trials": fixed_trials,
            "diagnostics": {"solver_path": "analytic_root"},
        }
    return {
        "loglik": float(loglik),
        "moment_loglik": float(loglik),
        "alpha": [],
        "params": params,
        "success": bool(np.isfinite(loglik)),
        "status": "analytic_mle" if np.isfinite(loglik) else "infeasible_root_support",
        "nit": 0,
        "nfev": 1,
        "starts": 0,
        "fixed_trials": fixed_trials,
        "diagnostics": {"solver_path": "analytic_root"},
    }


def poisson_negative_loglik_and_gradient(
    vector: np.ndarray,
    observations: np.ndarray,
    design: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Convex Poisson-regression objective for identity-link nonnegative rates."""
    vector = np.asarray(vector, dtype=float)
    observations = np.asarray(observations, dtype=float)
    rates = np.asarray(design, dtype=float) @ vector
    if (
        np.any(~np.isfinite(vector))
        or np.any(vector < 0.0)
        or np.any(~np.isfinite(rates))
        or np.any(rates < 0.0)
        or np.any((rates == 0.0) & (observations > 0.0))
    ):
        return 1e100, np.zeros_like(vector)
    objective = float(
        np.sum(rates - xlogy(observations, rates) + gammaln(observations + 1.0))
    )
    ratio = np.zeros_like(rates)
    positive = rates > 0.0
    ratio[positive] = observations[positive] / rates[positive]
    gradient = np.asarray(design, dtype=float).T @ (1.0 - ratio)
    if not np.isfinite(objective) or np.any(~np.isfinite(gradient)):
        return 1e100, np.zeros_like(vector)
    return objective, np.asarray(gradient, dtype=float)


def optimize_poisson_with_parents(
    core,
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parent_mask: int,
    starts_count: int,
    maxiter: int,
) -> dict[str, Any]:
    """Certified convex MLE for Poisson local models with parents."""
    parents = parents_from_mask(parent_mask, data.shape[1])
    if not parents:
        raise ValueError("parented Poisson solver requires at least one parent")
    observations = data[:, child].astype(float)
    parent_values = data[:, parents].astype(float)
    design = np.column_stack([np.ones(len(data), dtype=float), parent_values])
    alpha_moment, _, _, _ = core.estimate_alpha_and_residual_moments(
        data, mean, covariance, child, parent_mask
    )
    factors = (1.0, 0.65, 1.35, 0.25, 1.75)
    starts: list[np.ndarray] = []
    for factor in factors[: max(1, starts_count)]:
        alpha = np.maximum(np.asarray(alpha_moment, dtype=float) * factor, 0.0)
        lam = max(float(mean[child] - alpha @ mean[parents]), 0.0)
        vector = np.r_[lam, alpha]
        rates = design @ vector
        if np.any((rates == 0.0) & (observations > 0.0)):
            vector[0] = EPS
        starts.append(vector)

    evaluated_starts = [
        (poisson_negative_loglik_and_gradient(vector, observations, design)[0], vector)
        for vector in starts
    ]
    baseline_value, baseline_vector = min(evaluated_starts, key=lambda item: item[0])
    tolerance = objective_certification_tolerance(baseline_value)
    raw_diagnostics: list[dict[str, Any]] = []
    certified: list[tuple[float, np.ndarray, float]] = []
    total_nfev = len(evaluated_starts)
    total_nit = 0

    def objective(vector: np.ndarray) -> float:
        return poisson_negative_loglik_and_gradient(vector, observations, design)[0]

    def gradient(vector: np.ndarray) -> np.ndarray:
        return poisson_negative_loglik_and_gradient(vector, observations, design)[1]

    for vector in starts:
        result = minimize(
            objective,
            vector,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(0.0, None)] * len(vector),
            options={
                "maxiter": maxiter,
                "ftol": 1e-10,
                "gtol": 1e-6,
                "maxls": 30,
            },
        )
        raw_diagnostics.append(optimizer_result_diagnostic("L-BFGS-B", result))
        total_nfev += int(getattr(result, "nfev", 0))
        total_nit += int(getattr(result, "nit", 0))
        candidate_vector = np.asarray(result.x, dtype=float)
        specialized_value = objective(candidate_vector)
        if candidate_vector.size != len(parents) + 1:
            continue
        lam = float(candidate_vector[0])
        alpha = candidate_vector[1:]
        try:
            canonical_loglik = exact_convolution_loglik(
                core,
                data[:, child],
                parent_values @ alpha,
                "Poisson",
                {"lam": lam},
            )
        except (FloatingPointError, OverflowError, ValueError):
            continue
        agreement = abs(float(canonical_loglik) + specialized_value)
        agreement_tolerance = objective_certification_tolerance(canonical_loglik)
        if (
            bool(result.success)
            and np.all(np.isfinite(candidate_vector))
            and np.all(candidate_vector >= 0.0)
            and np.isfinite(specialized_value)
            and np.isfinite(canonical_loglik)
            and agreement <= agreement_tolerance
            and specialized_value <= baseline_value + tolerance
        ):
            certified.append(
                (float(specialized_value), candidate_vector, float(agreement))
            )

    diagnostics = {
        "solver_path": "poisson_convex",
        "optimizer_results": raw_diagnostics,
        "optimizer_primary_successful_starts": len(certified),
        "optimizer_primary_failed_starts": len(starts) - len(certified),
        "poisson_canonical_agreement_abs": None,
    }
    if certified:
        best_value, best_vector, agreement = min(certified, key=lambda item: item[0])
        if baseline_value < best_value:
            best_value = float(baseline_value)
            best_vector = baseline_vector.copy()
            alpha = best_vector[1:]
            canonical_loglik = exact_convolution_loglik(
                core,
                data[:, child],
                parent_values @ alpha,
                "Poisson",
                {"lam": float(best_vector[0])},
            )
            agreement = abs(float(canonical_loglik) + best_value)
        diagnostics["poisson_canonical_agreement_abs"] = float(agreement)
        return {
            "loglik": -float(best_value),
            "moment_loglik": -float(baseline_value),
            "alpha": best_vector[1:].tolist(),
            "params": {"lam": float(best_vector[0])},
            "success": True,
            "status": "poisson_convex_mle",
            "nit": total_nit,
            "nfev": total_nfev,
            "starts": len(starts),
            "fixed_trials": None,
            "diagnostics": diagnostics,
        }

    return {
        "loglik": -float(baseline_value) if baseline_value < 1e99 else -math.inf,
        "moment_loglik": -float(baseline_value) if baseline_value < 1e99 else -math.inf,
        "alpha": baseline_vector[1:].tolist(),
        "params": {"lam": float(baseline_vector[0])},
        "success": False,
        "status": "uncertified_optimizer_failure",
        "nit": total_nit,
        "nfev": total_nfev,
        "starts": len(starts),
        "fixed_trials": None,
        "diagnostics": diagnostics,
    }


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
    """Certified generic convolution MLE with one derivative-free rescue."""
    parents = parents_from_mask(parent_mask, data.shape[1])
    alpha_moment, _, _, _ = core.estimate_alpha_and_residual_moments(
        data, mean, covariance, child, parent_mask
    )
    parent_values = (
        data[:, parents].astype(float)
        if parents
        else np.zeros((len(data), 0), dtype=float)
    )
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
    )

    def evaluate(vector: np.ndarray) -> float:
        try:
            alpha = vector[: len(parents)]
            params = unpack_family_values(
                family, vector[len(parents) :], fixed_trials
            )
            thinning_mean = (
                parent_values @ alpha if parents else np.zeros(len(data))
            )
            loglik = exact_convolution_loglik(
                core, data[:, child], thinning_mean, family, params
            )
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100
        return -float(loglik) if np.isfinite(loglik) else 1e100

    evaluated_starts = [
        (float(evaluate(vector)), vector, bounds) for vector, bounds in starts
    ]
    baseline_value, baseline_vector, baseline_bounds = min(
        evaluated_starts, key=lambda item: item[0]
    )
    tolerance = objective_certification_tolerance(baseline_value)
    total_nfev = len(evaluated_starts)
    total_nit = 0
    raw_diagnostics: list[dict[str, Any]] = []
    successful: list[tuple[float, np.ndarray]] = []
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
        raw_diagnostics.append(optimizer_result_diagnostic("L-BFGS-B", result))
        total_nfev += int(getattr(result, "nfev", 0))
        total_nit += int(getattr(result, "nit", 0))
        candidate_vector = np.asarray(result.x, dtype=float)
        candidate_value = evaluate(candidate_vector)
        if (
            bool(result.success)
            and np.all(np.isfinite(candidate_vector))
            and np.isfinite(candidate_value)
            and candidate_value < 1e99
            and candidate_value <= baseline_value + tolerance
        ):
            successful.append((float(candidate_value), candidate_vector))

    status = "optimizer_converged_lbfgsb"
    powell_attempted = False
    powell_success = False
    if not successful:
        powell_attempted = True
        result = minimize(
            evaluate,
            baseline_vector,
            method="Powell",
            bounds=baseline_bounds,
            options={
                "maxiter": maxiter,
                "xtol": 1e-8,
                "ftol": 1e-10,
            },
        )
        raw_diagnostics.append(optimizer_result_diagnostic("Powell", result))
        total_nfev += int(getattr(result, "nfev", 0))
        total_nit += int(getattr(result, "nit", 0))
        candidate_vector = np.asarray(result.x, dtype=float)
        candidate_value = evaluate(candidate_vector)
        if (
            bool(result.success)
            and np.all(np.isfinite(candidate_vector))
            and np.isfinite(candidate_value)
            and candidate_value < 1e99
            and candidate_value <= baseline_value + tolerance
        ):
            successful.append((float(candidate_value), candidate_vector))
            status = "optimizer_converged_powell"
            powell_success = True

    diagnostics = {
        "solver_path": "general_convolution",
        "optimizer_results": raw_diagnostics,
        "optimizer_primary_successful_starts": sum(
            item["method"] == "L-BFGS-B" and item["success"]
            for item in raw_diagnostics
        ),
        "optimizer_primary_failed_starts": sum(
            item["method"] == "L-BFGS-B" and not item["success"]
            for item in raw_diagnostics
        ),
        "optimizer_powell_attempted": powell_attempted,
        "optimizer_powell_success": powell_success,
    }
    if successful:
        best_value, best_vector = min(successful, key=lambda item: item[0])
        if baseline_value < best_value:
            best_value = float(baseline_value)
            best_vector = baseline_vector.copy()
        return {
            "loglik": -float(best_value),
            "moment_loglik": -float(baseline_value),
            "alpha": best_vector[: len(parents)].tolist(),
            "params": unpack_family_values(
                family, best_vector[len(parents) :], fixed_trials
            ),
            "success": True,
            "status": status,
            "nit": total_nit,
            "nfev": total_nfev,
            "starts": len(starts),
            "fixed_trials": fixed_trials,
            "diagnostics": diagnostics,
        }

    return {
        "loglik": -float(baseline_value) if baseline_value < 1e99 else -math.inf,
        "moment_loglik": -float(baseline_value) if baseline_value < 1e99 else -math.inf,
        "alpha": baseline_vector[: len(parents)].tolist(),
        "params": unpack_family_values(
            family, baseline_vector[len(parents) :], fixed_trials
        ),
        "success": False,
        "status": "uncertified_optimizer_failure",
        "nit": total_nit,
        "nfev": total_nfev,
        "starts": len(starts),
        "fixed_trials": fixed_trials,
        "diagnostics": diagnostics,
    }


def nba_binomial_trial_index(
    core,
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parent_mask: int,
) -> tuple[int, bool, float, float]:
    """Select discrete n by the paper's fixed Binomial moment map.

    NBA remains a full likelihood optimization over every continuous local
    parameter.  Only the discrete component is fixed by H_Binomial at the
    paper's projected-alpha residual moments, matching the manuscript's
    locally constant treatment of discrete parameters.  There is no growing
    sieve, observed-count bound, or simulation parameter range.
    """
    _, _, residual_mean, residual_variance = (
        core.estimate_alpha_and_residual_moments(
            data, mean, covariance, child, parent_mask
        )
    )
    params, used_extension = core.binomial_moment_map(
        residual_mean,
        residual_variance,
        binomial_n_max=NBA_BINOMIAL_N_MAX,
    )
    return (
        int(params["n"]),
        bool(used_extension),
        float(residual_mean),
        float(residual_variance),
    )


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
    discrete_extension = False
    discrete_residual_mean = None
    discrete_residual_variance = None
    if family == "Binomial":
        fixed_trials, discrete_extension, discrete_residual_mean, discrete_residual_variance = (
            nba_binomial_trial_index(
                core, data, mean, covariance, child, parent_mask
            )
        )
        trial_values: list[int | None] = [fixed_trials]
        minimum = maximum = fixed_trials
    else:
        minimum = None
        maximum = None
        trial_values = [None]

    def run_trials(trials: int | None) -> dict[str, Any]:
        if parents and family in ("Bernoulli", "Binomial"):
            support_maximum = 1 if family == "Bernoulli" else int(trials)
            parent_total = data[:, parents].sum(axis=1)
            impossible = (data[:, child] > support_maximum) & (parent_total == 0)
            if np.any(impossible):
                return {
                    "loglik": -math.inf,
                    "moment_loglik": -math.inf,
                    "alpha": [],
                    "params": (
                        {"n": support_maximum} if family == "Binomial" else {}
                    ),
                    "success": False,
                    "status": "infeasible_conditional_support",
                    "nit": 0,
                    "nfev": 0,
                    "starts": 0,
                    "fixed_trials": trials,
                    "diagnostics": {
                        "solver_path": "conditional_support_check",
                        "support_violating_rows": int(np.sum(impossible)),
                    },
                }
        if not parents and family in ("Poisson", "Geom", "Bernoulli", "Binomial"):
            return analytic_root_candidate(
                core, data[:, child], family, trials
            )
        if family == "Poisson":
            return optimize_poisson_with_parents(
                core,
                data,
                mean,
                covariance,
                child,
                parent_mask,
                starts_count,
                maxiter,
            )
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

    candidates = [run_trials(trials) for trials in trial_values]
    if not candidates:
        return FamilyFit(
            family, math.inf, -math.inf, [], {},
            len(parents) + core.family_dimension(family),
            False, "infeasible_root_support", 0, 0, 0, -math.inf, 0.0,
            time.perf_counter() - started,
        )
    # There is one paper-defined discrete index.  All continuous parameters
    # follow the certified analytic/specialized/general full-MLE paths above.
    best = candidates[0]
    best_moment_loglik = max(item["moment_loglik"] for item in candidates)
    dimension = core.family_dimension(family)
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
        diagnostics={
            **best.get("diagnostics", {}),
            "binomial_n_min": minimum,
            "binomial_n_max": maximum,
            "binomial_n_candidates": (
                len(trial_values) if family == "Binomial" else 0
            ),
            "binomial_n_tie_break": (
                "not_applicable_single_mom_index"
                if family == "Binomial" else None
            ),
            "binomial_n_policy": (
                "paper_H_binomial_discrete_index_continuous_full_mle"
                if family == "Binomial" else None
            ),
            "binomial_h_extension_used": (
                discrete_extension if family == "Binomial" else None
            ),
            "binomial_discrete_residual_mean": (
                discrete_residual_mean if family == "Binomial" else None
            ),
            "binomial_discrete_residual_variance": (
                discrete_residual_variance if family == "Binomial" else None
            ),
            "certified_profiles": int(
                sum(bool(item["success"]) for item in candidates)
            ),
            "uncertified_profiles": int(
                sum(
                    np.isfinite(float(item["loglik"]))
                    and not bool(item["success"])
                    for item in candidates
                )
            ),
        },
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


def uncertified_finite_families(family_fits: Sequence[FamilyFit]) -> list[str]:
    """Identify local candidates that require fail-closed termination."""
    return [
        fit.family
        for fit in family_fits
        if (
            fit.status == "uncertified_optimizer_failure"
            or int(fit.diagnostics.get("uncertified_profiles", 0)) > 0
            or (
                np.isfinite(fit.bic)
                and not fit.success
                and fit.status
                not in ("infeasible_root_support", "infeasible_conditional_support")
            )
        )
    ]


def format_edges(parent_masks: Sequence[int], variables: Sequence[str]) -> str:
    edges = []
    for child, mask in enumerate(parent_masks):
        for parent in parents_from_mask(mask, len(variables)):
            edges.append(f"{variables[parent]}->{variables[child]}")
    return ", ".join(sorted(edges)) if edges else "(none)"


_PROFILE_WORKER_CORE: Any | None = None
_PROFILE_WORKER_DATA: np.ndarray | None = None
_PROFILE_WORKER_MEAN: np.ndarray | None = None
_PROFILE_WORKER_COVARIANCE: np.ndarray | None = None


def local_profile_tasks(
    d: int,
    *,
    children: Sequence[int] | None = None,
    families: Sequence[str] = FAMILIES,
) -> tuple[LocalProfileTask, ...]:
    """Return the deterministic independent local-profile task grid."""
    selected_children = tuple(range(d)) if children is None else tuple(children)
    tasks = []
    for child in selected_children:
        if not 0 <= child < d:
            raise ValueError(f"invalid child index {child} for d={d}")
        for parent_mask in range(1 << d):
            if (parent_mask >> child) & 1:
                continue
            for family in families:
                if family not in FAMILIES:
                    raise ValueError(f"unsupported local family {family}")
                tasks.append(LocalProfileTask(child, parent_mask, family))
    return tuple(tasks)


def _fit_local_profile(
    core,
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    task: LocalProfileTask,
    estimator: str,
    starts: int,
    maxiter: int,
) -> FamilyFit:
    if estimator == "moment":
        return moment_family_fit(
            core,
            data,
            mean,
            covariance,
            task.child,
            task.parent_mask,
            task.family,
        )
    if estimator == "optimization":
        # There is exactly one process pool, around independent local profiles.
        # The compatibility parameter is passed as one so no inner layer can interpret
        # it as permission to create a nested pool.
        return optimization_family_fit(
            core,
            data,
            mean,
            covariance,
            task.child,
            task.parent_mask,
            task.family,
            starts,
            maxiter,
            1,
        )
    raise ValueError(estimator)


def _initialize_profile_worker(
    core_path: str,
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> None:
    """Windows-spawn-safe initializer; workers receive read-only computations."""
    global _PROFILE_WORKER_CORE
    global _PROFILE_WORKER_DATA
    global _PROFILE_WORKER_MEAN
    global _PROFILE_WORKER_COVARIANCE
    for variable in BLAS_VARIABLES:
        os.environ[variable] = "1"
    _PROFILE_WORKER_CORE = load_paper_core(Path(core_path))
    _PROFILE_WORKER_DATA = np.asarray(data)
    _PROFILE_WORKER_MEAN = np.asarray(mean)
    _PROFILE_WORKER_COVARIANCE = np.asarray(covariance)


def _profile_worker_task(
    task: LocalProfileTask,
    estimator: str,
    starts: int,
    maxiter: int,
) -> LocalProfileExecution:
    if (
        _PROFILE_WORKER_CORE is None
        or _PROFILE_WORKER_DATA is None
        or _PROFILE_WORKER_MEAN is None
        or _PROFILE_WORKER_COVARIANCE is None
    ):
        raise RuntimeError("NBA local-profile worker was not initialized")
    started_ns = time.perf_counter_ns()
    fit = _fit_local_profile(
        _PROFILE_WORKER_CORE,
        _PROFILE_WORKER_DATA,
        _PROFILE_WORKER_MEAN,
        _PROFILE_WORKER_COVARIANCE,
        task,
        estimator,
        starts,
        maxiter,
    )
    completed_ns = time.perf_counter_ns()
    return LocalProfileExecution(
        task=task,
        fit=fit,
        process_id=os.getpid(),
        started_ns=started_ns,
        completed_ns=completed_ns,
        blas_thread_values=tuple(os.environ.get(name, "") for name in BLAS_VARIABLES),
    )


def submit_local_profile_tasks(
    executor: Any,
    tasks: Sequence[LocalProfileTask],
    estimator: str,
    starts: int,
    maxiter: int,
) -> dict[Any, LocalProfileTask]:
    """Submit the complete batch before collection (instrumentable contract)."""
    return {
        executor.submit(
            _profile_worker_task,
            task,
            estimator,
            starts,
            maxiter,
        ): task
        for task in tasks
    }


def peak_overlapping_profile_tasks(
    executions: Sequence[LocalProfileExecution],
) -> int:
    events: list[tuple[int, int]] = []
    for execution in executions:
        events.append((execution.started_ns, 1))
        events.append((execution.completed_ns, -1))
    active = 0
    peak = 0
    # End events sort before start events at an identical timestamp.
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def compute_local_profiles(
    core,
    core_path: Path,
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    tasks: Sequence[LocalProfileTask],
    estimator: str,
    starts: int,
    maxiter: int,
    workers: int,
    *,
    executor_factory: Callable[..., Any] | None = None,
) -> tuple[dict[LocalProfileTask, FamilyFit], dict[str, Any]]:
    """Compute pure local profiles serially or in one process pool."""
    if workers < 1:
        raise ValueError("optimization workers must be positive")
    ordered_tasks = tuple(tasks)
    if len(set(ordered_tasks)) != len(ordered_tasks):
        raise ValueError("duplicate NBA local-profile task")
    executions: list[LocalProfileExecution] = []
    maximum_workers = min(workers, len(ordered_tasks)) if ordered_tasks else 0
    maximum_outstanding = 0
    if workers == 1 or len(ordered_tasks) <= 1:
        for task in ordered_tasks:
            started_ns = time.perf_counter_ns()
            fit = _fit_local_profile(
                core, data, mean, covariance, task, estimator, starts, maxiter
            )
            executions.append(
                LocalProfileExecution(
                    task=task,
                    fit=fit,
                    process_id=os.getpid(),
                    started_ns=started_ns,
                    completed_ns=time.perf_counter_ns(),
                    blas_thread_values=tuple(
                        os.environ.get(name, "") for name in BLAS_VARIABLES
                    ),
                )
            )
        execution_mode = "serial"
        maximum_outstanding = 1 if ordered_tasks else 0
    else:
        factory = executor_factory or ProcessPoolExecutor
        with factory(
            max_workers=maximum_workers,
            initializer=_initialize_profile_worker,
            initargs=(
                str(Path(core_path).resolve()),
                np.asarray(data),
                np.asarray(mean),
                np.asarray(covariance),
            ),
        ) as executor:
            futures = submit_local_profile_tasks(
                executor, ordered_tasks, estimator, starts, maxiter
            )
            maximum_outstanding = len(futures)
            for future in as_completed(futures):
                expected_task = futures[future]
                execution = future.result()
                if execution.task != expected_task:
                    raise RuntimeError("NBA local-profile worker returned the wrong task")
                executions.append(execution)
        execution_mode = "process_pool"

    by_task: dict[LocalProfileTask, FamilyFit] = {}
    for execution in sorted(executions, key=lambda item: item.task):
        if execution.task in by_task:
            raise RuntimeError(f"duplicate NBA local-profile result: {execution.task}")
        by_task[execution.task] = execution.fit
    if set(by_task) != set(ordered_tasks):
        raise RuntimeError("NBA local-profile result grid is incomplete")
    worker_process_ids = sorted({item.process_id for item in executions})
    blas_values = sorted(
        {value for item in executions for value in item.blas_thread_values}
    )
    telemetry = {
        "execution_spec": NBA_EXECUTION_SPEC,
        "task_unit": "(child,parent_mask,family)",
        "execution_mode": execution_mode,
        "requested_workers": workers,
        "maximum_workers": maximum_workers,
        "submitted_tasks": len(ordered_tasks),
        "completed_tasks": len(executions),
        "maximum_outstanding_futures": maximum_outstanding,
        "unique_worker_processes": len(worker_process_ids),
        "worker_process_ids": worker_process_ids,
        "peak_overlapping_tasks": peak_overlapping_profile_tasks(executions),
        "worker_blas_thread_values": blas_values,
        "nested_process_pools": False,
        "worker_writes": False,
        "parent_only_canonical_writes": True,
    }
    return by_task, telemetry


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
    data, frame, variables, input_path = load_dataset(workspace, dataset, season)
    del frame
    input_hash = file_sha256(input_path)
    runner_hash = file_sha256(Path(__file__).resolve())
    estimator_contract = method_contract(core)
    if graph_path.exists() and not force:
        cached = json.loads(graph_path.read_text(encoding="utf-8"))
        if (
            cached.get("core_sha256") == core_hash
            and cached.get("runner_sha256") == runner_hash
            and cached.get("mle_solver_spec") == MLE_SOLVER_SPEC
            and cached.get("input_sha256") == input_hash
            and cached.get("estimator") == estimator
            and cached.get("optimization_starts") == starts
            and cached.get("optimization_maxiter") == maxiter
            and cached.get("optimization_workers") == optimization_workers
            and cached.get("parallel_execution", {}).get("execution_spec")
            == NBA_EXECUTION_SPEC
            and cached.get("method_contract") == estimator_contract
        ):
            print(f"[resume] {dataset} {season} {estimator}", flush=True)
            return cached
        raise RuntimeError(
            f"Refusing incompatible NBA checkpoint {graph_path}; use a new "
            "run-id or remove only this run's cell after inspection"
        )

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
    tasks = local_profile_tasks(d)
    profile_results, parallel_telemetry = compute_local_profiles(
        core,
        Path(core.__file__).resolve(),
        data,
        mean,
        covariance,
        tasks,
        estimator,
        starts,
        maxiter,
        optimization_workers,
    )
    for child in range(d):
        for parent_mask in range(mask_count):
            if (parent_mask >> child) & 1:
                continue
            family_fits = []
            for family in FAMILIES:
                fit = profile_results[
                    LocalProfileTask(child, parent_mask, family)
                ]
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
                row["diagnostics"] = json.dumps(
                    row["diagnostics"], sort_keys=True
                )
                rows.append(row)
            if estimator == "optimization":
                unconverged = uncertified_finite_families(family_fits)
                if unconverged:
                    raise RuntimeError(
                        "Fail-closed NBA MLE profiling: at least one feasible "
                        f"profile did not converge for {dataset} {season} "
                        f"child={variables[child]} parents={parent_mask}: "
                        f"{unconverged}"
                    )
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
                "diagnostics": fit.diagnostics,
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
        "input_sha256": input_hash,
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
        "runner_sha256": runner_hash,
        "mle_solver_spec": MLE_SOLVER_SPEC,
        "method_contract": estimator_contract,
        "optimization_starts": starts,
        "optimization_maxiter": maxiter,
        "optimization_workers": optimization_workers,
        "parallel_execution": parallel_telemetry,
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
    core,
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
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "mle_solver_spec": MLE_SOLVER_SPEC,
        "parallel_execution_spec": NBA_EXECUTION_SPEC,
        "parallel_task_unit": "(child,parent_mask,family)",
        "method_contract": method_contract(core),
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
        default=REPO_ROOT / "src/nba_core.py",
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
        manifest_path, args, core_path, core_hash, core, status="running"
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
            manifest_path, args, core_path, core_hash, core, status="complete"
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
            core,
            status="failed",
            error=repr(exc),
        )
        raise


if __name__ == "__main__":
    main()
