"""Focused regression tests for the adaptive-family NBA driver."""

from __future__ import annotations

import json
import inspect
import math
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_adaptive_ptsem_real_study as real


REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_PATH = REPO_ROOT / "src/nba_core.py"


class AdaptiveRealStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = real.load_paper_core(CORE_PATH)

    def test_family_library_and_dag_enumeration(self) -> None:
        self.assertEqual(tuple(self.core.FAMLIB), real.FAMILIES)
        rng = np.random.default_rng(17)
        for d, expected in ((3, 25), (4, 543), (5, 29281)):
            scores = [[float("inf")] * (1 << d) for _ in range(d)]
            for child in range(d):
                for mask in range(1 << d):
                    if not ((mask >> child) & 1):
                        scores[child][mask] = float(rng.normal())
            masks, dp_score = self.core.exact_order_dp(scores)
            count, ranked = real.top_dags(scores, d - 1, top_k=2)
            self.assertEqual(count, expected)
            self.assertAlmostEqual(dp_score, ranked[0][0], places=10)
            self.assertEqual(
                tuple(masks[node] for node in range(d)), ranked[0][1]
            )

    def test_vectorized_geometric_matches_paper_core(self) -> None:
        rng = np.random.default_rng(44)
        observations = rng.integers(0, 31, 500)
        thinning_mean = rng.uniform(0.0, 20.0, 500)
        for probability in (0.02, 0.2, 0.8, 0.99, 0.999999):
            expected = self.core.convolution_loglik(
                observations,
                thinning_mean,
                "Geom",
                {"p": probability},
            )
            obtained = real.exact_convolution_loglik(
                self.core,
                observations,
                thinning_mean,
                "Geom",
                {"p": probability},
            )
            self.assertAlmostEqual(expected, obtained, places=8)

    def test_moment_local_score_matches_paper_core(self) -> None:
        data, _, _ = self.core.simulate_ptsem(
            d=3,
            n=500,
            seed=812,
            avg_indegree=1.0,
            max_parents=2,
            dag_generation_mode="exact_edges",
        )
        mean, covariance = self.core.empirical_moments(data)
        for child in range(3):
            for mask in range(1 << 3):
                if (mask >> child) & 1:
                    continue
                expected = self.core.local_score(
                    data,
                    mean,
                    covariance,
                    child,
                    mask,
                    real.FAMILIES,
                )
                fits = [
                    real.moment_family_fit(
                        self.core,
                        data,
                        mean,
                        covariance,
                        child,
                        mask,
                        family,
                    )
                    for family in real.FAMILIES
                ]
                obtained = min(fits, key=lambda fit: fit.bic)
                if expected.family is None:
                    self.assertTrue(np.isinf(expected.bic))
                    self.assertTrue(all(np.isinf(fit.bic) for fit in fits))
                    continue
                self.assertEqual(expected.family, obtained.family)
                self.assertAlmostEqual(expected.bic, obtained.bic, places=7)
                np.testing.assert_allclose(
                    expected.alpha, obtained.alpha, rtol=0.0, atol=1e-10
                )

    def test_optimization_does_not_degrade_start(self) -> None:
        rng = np.random.default_rng(919)
        # Keep the parent strictly positive so every bounded-support family
        # remains a feasible convolutional candidate in this unit test.
        parent = rng.poisson(4.0, 240) + 1
        child = rng.poisson(0.8 * parent) + rng.negative_binomial(
            4.0, 0.55, 240
        )
        data = np.column_stack([parent, child]).astype(int)
        mean, covariance = self.core.empirical_moments(data)
        for family in real.FAMILIES:
            fit = real.optimization_family_fit(
                self.core,
                data,
                mean,
                covariance,
                child=1,
                parent_mask=1,
                family=family,
                starts_count=1,
                maxiter=40,
                optimization_workers=2,
            )
            self.assertTrue(np.isfinite(fit.bic), family)
            self.assertGreaterEqual(fit.loglik_improvement, -1e-8)

    def test_forensic_root_poisson_regressions_are_analytic(self) -> None:
        variables = [
            "FOUL",
            "FTA",
            "FTM",
            "PERS_FOUL_DRAWN",
            "LOOSE_BALL_FOUL_DRAWN",
        ]
        expected = {
            "2016-17": (
                0.31290650406504067,
                -6972.9777584722515,
                13955.14972793455,
            ),
            "2015-16": (
                0.29715447154471547,
                -6790.897491791102,
                13590.98919457225,
            ),
        }
        with mock.patch.object(
            real, "minimize", side_effect=AssertionError("analytic root called scipy")
        ):
            for season, (lam, loglik, bic) in expected.items():
                path = (
                    REPO_ROOT
                    / "data/nba/processed_team_quarter"
                    / f"expanded_team_quarter_{season}.csv"
                )
                data = pd.read_csv(path)[variables].to_numpy(dtype=np.int64)
                mean, covariance = self.core.empirical_moments(data)
                fit = real.optimization_family_fit(
                    self.core,
                    data,
                    mean,
                    covariance,
                    child=4,
                    parent_mask=0,
                    family="Poisson",
                    starts_count=2,
                    maxiter=250,
                    optimization_workers=1,
                )
                self.assertTrue(fit.success)
                self.assertEqual(fit.status, "analytic_mle")
                self.assertEqual(fit.nit, 0)
                self.assertAlmostEqual(float(fit.params["lam"]), lam, places=15)
                self.assertAlmostEqual(fit.loglik, loglik, places=10)
                self.assertAlmostEqual(fit.bic, bic, places=10)
                self.assertEqual(fit.diagnostics["uncertified_profiles"], 0)

    def test_synthetic_analytic_root_families(self) -> None:
        cases = {
            "Poisson": np.array([0, 1, 2, 3, 1, 4], dtype=int),
            "Geom": np.array([0, 0, 1, 2, 4, 5], dtype=int),
            "Bernoulli": np.array([0, 1, 1, 0, 1, 0], dtype=int),
            # Exact mean=2 and variance=1 imply H_Binomial n=4.
            "Binomial": np.repeat(np.arange(5), [1, 4, 6, 4, 1]).astype(int),
        }
        with mock.patch.object(
            real, "minimize", side_effect=AssertionError("analytic root called scipy")
        ):
            for family, observations in cases.items():
                with self.subTest(family=family):
                    data = observations[:, None]
                    mean, covariance = self.core.empirical_moments(data)
                    fit = real.optimization_family_fit(
                        self.core, data, mean, covariance, 0, 0, family, 2, 100, 1
                    )
                    self.assertTrue(fit.success)
                    self.assertEqual(fit.status, "analytic_mle")
                    self.assertEqual(fit.nit, 0)
                    expected_loglik = self.core.convolution_loglik(
                        observations,
                        np.zeros(len(observations)),
                        family,
                        fit.params,
                    )
                    self.assertAlmostEqual(fit.loglik, expected_loglik, places=12)
                    if family == "Poisson":
                        self.assertAlmostEqual(fit.params["lam"], observations.mean())
                    elif family == "Geom":
                        self.assertAlmostEqual(
                            fit.params["p"], 1.0 / (1.0 + observations.mean())
                        )
                    elif family == "Bernoulli":
                        self.assertAlmostEqual(fit.params["p"], observations.mean())
                    else:
                        self.assertEqual(fit.params["n"], 4)
                        self.assertAlmostEqual(fit.params["p"], 0.5)

    def test_nba_fixed_binomial_cap_is_100_and_single_index(self) -> None:
        contract = real.method_contract(self.core)
        self.assertEqual(contract["estimator_binomial_n_max"], 50)
        self.assertEqual(contract["nba_binomial_n_max"], 100)
        for raw_n, probability, expected_n in (
            (51, 0.2, 51),
            (100, 0.2, 100),
            (101, 0.2, 100),
            (500, 0.05, 100),
        ):
            params, used_extension = self.core.binomial_moment_map(
                raw_n * probability,
                raw_n * probability * (1.0 - probability),
                binomial_n_max=real.NBA_BINOMIAL_N_MAX,
            )
            self.assertFalse(used_extension)
            self.assertEqual(params["n"], expected_n)
        source = inspect.getsource(real.optimization_family_fit)
        self.assertIn("trial_values: list[int | None] = [fixed_trials]", source)
        self.assertNotIn("fixed_trials + 1", source)

    def test_analytic_root_support_failures(self) -> None:
        bernoulli = real.analytic_root_candidate(
            self.core, np.array([0, 1, 2]), "Bernoulli", None
        )
        binomial = real.analytic_root_candidate(
            self.core, np.array([0, 2, 5]), "Binomial", 4
        )
        for candidate in (bernoulli, binomial):
            self.assertFalse(candidate["success"])
            self.assertEqual(candidate["status"], "infeasible_root_support")
            self.assertEqual(candidate["loglik"], -math.inf)

    def test_conditional_support_infeasibility_is_not_optimizer_failure(self) -> None:
        # When the parent count is zero, thinning is identically zero for every
        # alpha; child=2 therefore cannot be generated by Bernoulli exogenous noise.
        data = np.array([[0, 2], [1, 1], [2, 2], [3, 1]], dtype=int)
        mean, covariance = self.core.empirical_moments(data)
        with mock.patch.object(
            real, "minimize", side_effect=AssertionError("infeasible support optimized")
        ):
            fit = real.optimization_family_fit(
                self.core, data, mean, covariance, 1, 1, "Bernoulli", 2, 100, 1
            )
        self.assertFalse(fit.success)
        self.assertTrue(np.isinf(fit.bic))
        self.assertEqual(fit.status, "infeasible_conditional_support")
        self.assertEqual(real.uncertified_finite_families([fit]), [])

    def test_poisson_parented_objective_gradient_and_solver(self) -> None:
        rng = np.random.default_rng(20260831)
        parent = rng.poisson(3.0, 800)
        child = rng.poisson(1.1 + 0.72 * parent)
        data = np.column_stack([parent, child]).astype(int)
        design = np.column_stack([np.ones(len(data)), parent])
        for vector in (
            np.array([0.4, 0.2]),
            np.array([1.1, 0.72]),
            np.array([2.0, 1.1]),
        ):
            objective, gradient = real.poisson_negative_loglik_and_gradient(
                vector, child, design
            )
            canonical = self.core.convolution_loglik(
                child, parent * vector[1], "Poisson", {"lam": vector[0]}
            )
            self.assertAlmostEqual(objective, -canonical, places=10)
            step = 1e-6
            finite_difference = np.empty_like(vector)
            for index in range(len(vector)):
                delta = np.zeros_like(vector)
                delta[index] = step
                plus = real.poisson_negative_loglik_and_gradient(
                    vector + delta, child, design
                )[0]
                minus = real.poisson_negative_loglik_and_gradient(
                    vector - delta, child, design
                )[0]
                finite_difference[index] = (plus - minus) / (2.0 * step)
            np.testing.assert_allclose(gradient, finite_difference, rtol=2e-6, atol=2e-5)

        mean, covariance = self.core.empirical_moments(data)
        fit = real.optimization_family_fit(
            self.core, data, mean, covariance, 1, 1, "Poisson", 2, 250, 1
        )
        self.assertTrue(fit.success)
        self.assertEqual(fit.status, "poisson_convex_mle")
        self.assertEqual(fit.diagnostics["optimizer_primary_successful_starts"], 2)
        objectives = [
            item["fun"] for item in fit.diagnostics["optimizer_results"]
        ]
        self.assertAlmostEqual(objectives[0], objectives[1], places=7)
        self.assertLessEqual(
            fit.diagnostics["poisson_canonical_agreement_abs"], 1e-8
        )

    def test_general_solver_certification_and_powell_rescue(self) -> None:
        rng = np.random.default_rng(77)
        observations = rng.negative_binomial(4, 0.55, 1200)
        data = observations[:, None]
        mean, covariance = self.core.empirical_moments(data)
        fit = real.optimization_family_fit(
            self.core, data, mean, covariance, 0, 0, "NB", 2, 200, 1
        )
        self.assertTrue(fit.success)
        self.assertEqual(fit.status, "optimizer_converged_lbfgsb")

        def fake_powell_success(fun, x0, method, bounds, options, **kwargs):
            value = float(fun(np.asarray(x0, dtype=float)))
            return SimpleNamespace(
                success=method == "Powell",
                status=0 if method == "Powell" else 2,
                message="mock",
                fun=value,
                x=np.asarray(x0, dtype=float),
                nit=1,
                nfev=1,
            )

        with mock.patch.object(real, "minimize", side_effect=fake_powell_success):
            rescued = real.optimization_family_fit(
                self.core, data, mean, covariance, 0, 0, "NB", 2, 200, 1
            )
        self.assertTrue(rescued.success)
        self.assertEqual(rescued.status, "optimizer_converged_powell")

        def fake_all_fail(fun, x0, method, bounds, options, **kwargs):
            return SimpleNamespace(
                success=False,
                status=2,
                message="mock failure",
                fun=float(fun(np.asarray(x0, dtype=float))),
                x=np.asarray(x0, dtype=float),
                nit=0,
                nfev=1,
            )

        with mock.patch.object(real, "minimize", side_effect=fake_all_fail):
            failed = real.optimization_family_fit(
                self.core, data, mean, covariance, 0, 0, "NB", 2, 200, 1
            )
        self.assertFalse(failed.success)
        self.assertEqual(failed.status, "uncertified_optimizer_failure")
        self.assertEqual(failed.diagnostics["uncertified_profiles"], 1)
        self.assertEqual(real.uncertified_finite_families([failed]), ["NB"])

    def test_cache_requires_current_runner_and_solver_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            outputs = root / "outputs"
            workspace.mkdir()
            input_path = workspace / "input.csv"
            pd.DataFrame({"Y": [0, 1, 2]}).to_csv(input_path, index=False)
            dataset = "cache_provenance_test"
            real.DATASET_SPECS[dataset] = {
                "variables": ("Y",),
                "pattern": "input.csv",
                "unit": "test",
            }
            try:
                graph_path = outputs / dataset / "2016-17" / "optimization" / "graph_result.json"
                graph_path.parent.mkdir(parents=True)
                base = {
                    "marker": "cached",
                    "core_sha256": "core-hash",
                    "runner_sha256": real.file_sha256(Path(real.__file__).resolve()),
                    "mle_solver_spec": real.MLE_SOLVER_SPEC,
                    "input_sha256": real.file_sha256(input_path),
                    "estimator": "optimization",
                    "optimization_starts": 2,
                    "optimization_maxiter": 250,
                    "optimization_workers": 1,
                    "parallel_execution": {
                        "execution_spec": real.NBA_EXECUTION_SPEC
                    },
                    "method_contract": real.method_contract(self.core),
                }
                for missing in ("runner_sha256", "mle_solver_spec", "method_contract"):
                    payload = dict(base)
                    payload.pop(missing)
                    graph_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        real.run_one(
                            self.core, workspace, outputs, dataset, "2016-17",
                            "optimization", 2, 250, 1, False, "core-hash"
                        )
                graph_path.write_text(json.dumps(base), encoding="utf-8")
                reused = real.run_one(
                    self.core, workspace, outputs, dataset, "2016-17",
                    "optimization", 2, 250, 1, False, "core-hash"
                )
                self.assertEqual(reused["marker"], "cached")
            finally:
                real.DATASET_SPECS.pop(dataset, None)

    def test_workers_four_and_twelve_preserve_all_nonruntime_profile_results(self) -> None:
        data, _, _ = self.core.simulate_ptsem(
            d=3,
            n=120,
            seed=99017,
            avg_indegree=1.0,
            max_parents=2,
            dag_generation_mode="exact_edges",
        )
        mean, covariance = self.core.empirical_moments(data)
        tasks = real.local_profile_tasks(3, children=(1,))
        serial, serial_telemetry = real.compute_local_profiles(
            self.core,
            CORE_PATH,
            data,
            mean,
            covariance,
            tasks,
            "optimization",
            2,
            250,
            1,
        )
        self.assertEqual(serial_telemetry["peak_overlapping_tasks"], 1)
        for workers in (4, 12):
            with self.subTest(workers=workers):
                parallel, parallel_telemetry = real.compute_local_profiles(
                    self.core,
                    CORE_PATH,
                    data,
                    mean,
                    covariance,
                    tasks,
                    "optimization",
                    2,
                    250,
                    workers,
                )
                self.assertEqual(set(serial), set(parallel))
                for task in tasks:
                    left = asdict(serial[task])
                    right = asdict(parallel[task])
                    left.pop("runtime_sec")
                    right.pop("runtime_sec")
                    self.assertEqual(left, right, task)
                self.assertEqual(
                    parallel_telemetry["execution_mode"], "process_pool"
                )
                self.assertGreaterEqual(
                    parallel_telemetry["unique_worker_processes"], 2
                )
                self.assertGreaterEqual(
                    parallel_telemetry["peak_overlapping_tasks"], 2
                )
                self.assertEqual(
                    parallel_telemetry["worker_blas_thread_values"], ["1"]
                )

    def test_workers_twelve_submit_at_least_twelve_independent_tasks(self) -> None:
        class InstrumentedExecutor:
            def __init__(self) -> None:
                self.calls: list[tuple[object, tuple[object, ...]]] = []

            def submit(self, function, *args):
                token = object()
                self.calls.append((token, (function, *args)))
                return token

        tasks = real.local_profile_tasks(
            3,
            children=(0,),
            families=("Poisson", "NB", "ZIP"),
        )
        self.assertEqual(len(tasks), 12)
        executor = InstrumentedExecutor()
        futures = real.submit_local_profile_tasks(
            executor, tasks, "optimization", 2, 250
        )
        self.assertEqual(len(futures), 12)
        self.assertEqual(len(executor.calls), 12)
        self.assertEqual(set(futures.values()), set(tasks))
        # All 12 futures exist before collection begins: structural occupancy
        # is not limited by starts=2 or children=5.
        self.assertGreaterEqual(len(futures), 8)

    def test_profile_workers_never_create_nested_pools_or_write(self) -> None:
        worker_source = inspect.getsource(real._profile_worker_task)
        fit_source = inspect.getsource(real._fit_local_profile)
        self.assertNotIn("ProcessPoolExecutor", worker_source + fit_source)
        self.assertNotIn("atomic_write", worker_source + fit_source)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    unittest.main()
