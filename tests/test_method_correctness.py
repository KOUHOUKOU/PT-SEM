"""Statistical and numerical correctness tests for PT-SEM."""

from __future__ import annotations

import importlib.util
import inspect
import itertools
import math
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["PTSEM_FAMILY_ASSIGNMENT"] = "iid_uniform"

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, xlogy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/nba/scripts"))
import nba_core as core


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NBA = load_module(
    "ptsem_test_nba_runner",
    ROOT / "experiments/nba/scripts/run_adaptive_ptsem_real_study.py",
)
NBA_SUMMARY = load_module(
    "ptsem_test_nba_summary_helper",
    ROOT / "scripts/summarize_nba.py",
)


def direct_eps_logpmf(family: str, k: int, params: dict[str, float]) -> float:
    if k < 0:
        return -math.inf
    if family == "Poisson":
        lam = float(params["lam"])
        return float(xlogy(k, lam) - lam - gammaln(k + 1))
    if family == "NB":
        r, p = float(params["r"]), float(params["p"])
        return float(
            gammaln(k + r) - gammaln(r) - gammaln(k + 1)
            + r * math.log(p) + k * math.log1p(-p)
        )
    if family == "ZIP":
        lam, rho = float(params["lam"]), float(params["rho"])
        poisson = float(xlogy(k, lam) - lam - gammaln(k + 1))
        if k == 0:
            return float(np.logaddexp(math.log(rho), math.log1p(-rho) + poisson))
        return math.log1p(-rho) + poisson
    if family == "Geom":
        p = float(params["p"])
        return math.log(p) + k * math.log1p(-p)
    if family in ("Binomial", "Bernoulli"):
        n = 1 if family == "Bernoulli" else int(params["n"])
        if k > n:
            return -math.inf
        p = float(params["p"])
        return float(
            gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p)
        )
    raise ValueError(family)


def brute_convolution_loglik(
    observations: np.ndarray,
    thinning_mean: np.ndarray,
    family: str,
    params: dict[str, float],
) -> float:
    total = 0.0
    for observed, mean in zip(observations, thinning_mean):
        terms = []
        for thin_count in range(int(observed) + 1):
            log_poisson = float(
                xlogy(thin_count, mean) - mean - gammaln(thin_count + 1)
            )
            terms.append(
                log_poisson
                + direct_eps_logpmf(family, int(observed) - thin_count, params)
            )
        total += float(logsumexp(terms))
    return total


def all_dag_optimum(scores: list[list[float]]) -> float:
    d = len(scores)
    directed_pairs = [(i, j) for i in range(d) for j in range(d) if i != j]
    best = math.inf
    for edge_bits in range(1 << len(directed_pairs)):
        masks = {node: 0 for node in range(d)}
        for index, (parent, child) in enumerate(directed_pairs):
            if (edge_bits >> index) & 1:
                masks[child] |= 1 << parent
        if core.is_acyclic(masks, d):
            best = min(best, core.graph_score(masks, scores))
    return best


class MomentInversionTests(unittest.TestCase):
    def test_six_family_round_trips(self) -> None:
        cases = {
            "Poisson": (3.2, 3.2, {"lam": 3.2}),
            "NB": (4.0, 6.0, {"r": 8.0, "p": 2.0 / 3.0}),
            "ZIP": (3.0, 5.25, {"lam": 3.75, "rho": 0.2}),
            "Geom": (4.0, 20.0, {"p": 0.2}),
            "Binomial": (6.0, 4.2, {"n": 20, "p": 0.3}),
            "Bernoulli": (0.3, 0.21, {"p": 0.3}),
        }
        for family, (mean, variance, expected) in cases.items():
            with self.subTest(family=family):
                params, used_extension = core._moment_map_with_extension(
                    family, mean, variance
                )
                self.assertFalse(used_extension)
                inverted, dimension = core.invert_moments(
                    family, mean, variance
                )
                self.assertEqual(inverted, params)
                self.assertEqual(dimension, core.family_dimension(family))
                self.assertEqual(set(params), set(expected))
                for name, value in expected.items():
                    self.assertAlmostEqual(float(params[name]), float(value), places=11)

    def test_all_six_total_maps_project_finite_off_domain_moments(self) -> None:
        cases = (
            ("Poisson", -1.0, 1.0),
            ("Geom", -1.0, 1.0),
            ("Bernoulli", -0.2, 1.0),
            ("Bernoulli", 1.2, 1.0),
            ("NB", 2.0, 2.0),
            ("NB", 2.0, 1.0),
            ("ZIP", 2.0, 1.0),
            ("Binomial", 2.0, 2.0),
            ("Binomial", 1.1, 1.4),
            ("Binomial", -1.0, 1.0),
        )
        for family, mean, variance in cases:
            with self.subTest(family=family):
                params, used_extension = core._moment_map_with_extension(
                    family, mean, variance
                )
                self.assertTrue(used_extension)
                self.assertEqual(
                    core._validated_family_params(family, params), params
                )
                inverted, dimension = core.invert_moments(
                    family, mean, variance
                )
                self.assertEqual(inverted, params)
                self.assertEqual(dimension, core.family_dimension(family))

    def test_nonfinite_moments_remain_numerical_failures(self) -> None:
        for family in core.FAMLIB:
            with self.subTest(family=family):
                with self.assertRaises(core.InvalidMomentInversion):
                    core._moment_map_with_extension(family, math.nan, 1.0)

    def test_extreme_finite_moments_still_map_to_legal_parameters(self) -> None:
        maximum = float(np.finfo(float).max)
        for family in core.FAMLIB:
            with self.subTest(family=family):
                params, _ = core._moment_map_with_extension(
                    family, maximum, maximum
                )
                self.assertEqual(
                    core._validated_family_params(family, params), params
                )

    def test_large_valid_inversions_avoid_intermediate_square_overflow(self) -> None:
        nb, nb_extension = core._moment_map_with_extension(
            "NB", 1e200, 2e200
        )
        self.assertFalse(nb_extension)
        self.assertAlmostEqual(nb["r"] / 1e200, 1.0)
        self.assertAlmostEqual(nb["p"], 0.5)
        binomial, binomial_extension = core._moment_map_with_extension(
            "Binomial", 1e150, 0.5e150
        )
        self.assertTrue(binomial_extension)
        self.assertEqual(binomial["n"], 50)
        self.assertEqual(
            core._validated_family_params("Binomial", binomial), binomial
        )
        self.assertLess(binomial["p"], 1.0)

    def test_binomial_fixed_estimator_space_boundaries(self) -> None:
        cases = (
            (31, 0.4, 31, 0.4),
            (40, 0.4, 40, 0.4),
            (50, 0.4, 50, 0.4),
            (51, 0.4, 50, 0.408),
            (60, 0.4, 50, 0.48),
            (100, 0.2, 50, 0.4),
            (500, 0.05, 50, 0.5),
        )
        for generating_n, probability, expected_n, expected_p in cases:
            with self.subTest(n_raw=generating_n):
                mean = generating_n * probability
                variance = generating_n * probability * (1.0 - probability)
                params, used_extension = core.binomial_moment_map(mean, variance)
                self.assertFalse(used_extension)
                self.assertEqual(params["n"], expected_n)
                self.assertAlmostEqual(params["p"], expected_p)

    def test_binomial_estimator_bound_is_independent_of_dgp_bound(self) -> None:
        moments = (31 * 0.4, 31 * 0.4 * 0.6)
        expected = core.binomial_moment_map(*moments)
        with mock.patch.object(core, "DGP_BINOMIAL_N_MIN", 7), mock.patch.object(
            core, "DGP_BINOMIAL_N_MAX", 9
        ):
            self.assertEqual(core.binomial_moment_map(*moments), expected)
        self.assertEqual(core.ESTIMATOR_BINOMIAL_N_MIN, 2)
        self.assertEqual(core.ESTIMATOR_BINOMIAL_N_MAX, 50)

    def test_binomial_local_score_evaluates_one_discrete_index(self) -> None:
        data = np.asarray([[0], [1], [2], [1], [3], [2], [1], [2]], dtype=int)
        mean, covariance = core.empirical_moments(data)
        original = core.binomial_moment_map
        calls: list[tuple[float, float]] = []

        def tracked(
            moment_mean: float,
            moment_variance: float,
            binomial_n_max: int = core.ESTIMATOR_BINOMIAL_N_MAX,
        ):
            calls.append((moment_mean, moment_variance))
            return original(
                moment_mean, moment_variance, binomial_n_max=binomial_n_max
            )

        with mock.patch.object(core, "binomial_moment_map", side_effect=tracked):
            core.local_score(data, mean, covariance, 0, 0, ("Binomial",))
        self.assertEqual(len(calls), 1)
        source = inspect.getsource(core.local_score) + inspect.getsource(
            core.binomial_moment_map
        )
        self.assertNotIn("n_hat + 1", source)
        self.assertNotIn("trials_raw + 1", source)

    def test_binomial_and_bernoulli_spaces_remain_separate(self) -> None:
        self.assertIn("Binomial", core.FAMLIB)
        self.assertIn("Bernoulli", core.FAMLIB)
        params, _ = core.binomial_moment_map(0.3, 0.21)
        self.assertGreaterEqual(params["n"], 2)
        with self.assertRaises(ValueError):
            core._validated_family_params("Binomial", {"n": 1, "p": 0.3})
        self.assertEqual(
            core._validated_family_params("Bernoulli", {"p": 0.3}),
            {"p": 0.3},
        )

    def test_every_binomial_total_map_output_is_in_fixed_finite_space(self) -> None:
        finite_pairs = [
            (-1.0, 2.0), (0.0, 0.0), (0.1, 0.0), (2.0, 3.0),
            (12.4, 7.44), (40.4, 24.24), (50.0, 45.0),
            (1e150, 0.5e150), (float(np.finfo(float).max),) * 2,
        ]
        for mean, variance in finite_pairs:
            with self.subTest(mean=mean, variance=variance):
                params, _ = core.binomial_moment_map(mean, variance)
                self.assertGreaterEqual(params["n"], 2)
                self.assertLessEqual(params["n"], 50)
                self.assertEqual(
                    core._validated_family_params("Binomial", params), params
                )
        with self.assertRaises(ValueError):
            core._validated_family_params("Binomial", {"n": 51, "p": 0.1})
        with self.assertRaises(ValueError):
            core._validated_family_params("Binomial", {"n": 50.5, "p": 0.1})

    def test_poisson_boundary_samples_never_produce_binomial_n_above_50(self) -> None:
        rng = np.random.default_rng(20260831)
        maximum = 0
        for rate in (0.5, 2.0, 5.0, 10.0, 30.0):
            for size in (20, 100, 1000):
                sample = rng.poisson(rate, size=size).astype(float)
                params, _ = core.binomial_moment_map(
                    float(sample.mean()), float(sample.var(ddof=0))
                )
                maximum = max(maximum, int(params["n"]))
        self.assertLessEqual(maximum, 50)

    def test_moment_maps_and_local_score_do_not_read_DGP_information(self) -> None:
        source = "\n".join(
            inspect.getsource(item)
            for item in (
                core._moment_map_with_extension,
                core.binomial_moment_map,
                core.local_score,
            )
        )
        for forbidden in (
            "DGP_BINOMIAL_N_MAX",
            "DGP_BINOMIAL_N_MIN",
            "true_params",
            "true_alpha",
            "true_graph",
            "regime",
            "n_support",
        ):
            self.assertNotIn(forbidden, source)

    def test_bic_dimensions_count_continuous_parameters_only(self) -> None:
        self.assertEqual(core.family_dimension("Poisson"), 1)
        self.assertEqual(core.family_dimension("Binomial"), 1)
        self.assertEqual(
            {family: core.family_dimension(family) for family in core.FAMLIB},
            {
                "Poisson": 1,
                "NB": 2,
                "ZIP": 2,
                "Geom": 1,
                "Binomial": 1,
                "Bernoulli": 1,
            },
        )


class LikelihoodTests(unittest.TestCase):
    def test_all_six_convolutions_match_direct_finite_sum(self) -> None:
        observations = np.array([0, 1, 2, 5, 9, 14])
        means = np.array([0.0, 0.2, 1.1, 3.0, 8.0, 17.0])
        cases = {
            "Poisson": {"lam": 2.3},
            "NB": {"r": 1.7, "p": 0.42},
            "ZIP": {"lam": 2.8, "rho": 0.35},
            "Geom": {"p": 0.27},
            "Binomial": {"n": 7, "p": 0.41},
            "Bernoulli": {"p": 0.63},
        }
        for family, params in cases.items():
            with self.subTest(family=family):
                expected = brute_convolution_loglik(
                    observations, means, family, params
                )
                obtained = core.convolution_loglik(
                    observations, means, family, params
                )
                self.assertAlmostEqual(obtained, expected, places=9)

    def test_nb_difficult_and_large_count_cases_are_finite(self) -> None:
        observations = np.array([0, 3, 25, 80, 160])
        means = np.array([0.0, 0.0, 0.2, 30.0, 180.0])
        for params in ({"r": 1e8, "p": 0.99999997}, {"r": 0.08, "p": 0.015}):
            value = core.convolution_loglik(observations, means, "NB", params)
            self.assertTrue(np.isfinite(value))
            expected = brute_convolution_loglik(observations, means, "NB", params)
            self.assertAlmostEqual(value, expected, places=6)

    def test_alpha_zero_reduces_to_exogenous_likelihood(self) -> None:
        observations = np.array([0, 1, 4, 7])
        means = np.zeros(len(observations))
        params = {"n": 9, "p": 0.35}
        obtained = core.convolution_loglik(observations, means, "Binomial", params)
        expected = sum(
            direct_eps_logpmf("Binomial", int(value), params)
            for value in observations
        )
        self.assertAlmostEqual(obtained, expected, places=11)

    def test_maximum_legal_n_binomial_convolution_is_exact_without_n_sized_array(self) -> None:
        observations = np.array([0, 1, 2, 4, 7, 11])
        means = np.array([0.0, 0.1, 0.5, 1.3, 3.0, 8.0])
        params = {"n": 50, "p": 0.08}
        expected = brute_convolution_loglik(
            observations, means, "Binomial", params
        )
        obtained = core.convolution_loglik(
            observations, means, "Binomial", params
        )
        self.assertTrue(np.isfinite(obtained))
        self.assertAlmostEqual(obtained, expected, places=8)
        source = inspect.getsource(core.convolution_loglik)
        self.assertIn("maximum_epsilon", source)
        self.assertNotIn("np.arange(trials + 1", source)

    def test_nba_delegates_every_family_to_canonical_core(self) -> None:
        source = inspect.getsource(NBA.exact_convolution_loglik)
        self.assertIn("core.convolution_loglik", source)
        for family, params in (
            ("Poisson", {"lam": 1.2}),
            ("NB", {"r": 2.0, "p": 0.6}),
            ("ZIP", {"lam": 1.5, "rho": 0.2}),
            ("Geom", {"p": 0.4}),
            ("Binomial", {"n": 4, "p": 0.3}),
            ("Bernoulli", {"p": 0.3}),
        ):
            y = np.array([0, 1, 3])
            mu = np.array([0.0, 0.5, 2.0])
            self.assertEqual(
                NBA.exact_convolution_loglik(core, y, mu, family, params),
                core.convolution_loglik(y, mu, family, params),
            )


class AlphaAndScoreTests(unittest.TestCase):
    def test_alpha_orientation_projection_and_residual_formula(self) -> None:
        data = np.array([[1, 3], [2, 5], [3, 8]], dtype=float)
        mean = np.array([4.0, 10.0])
        covariance = np.array([[2.0, 3.0], [3.0, 15.0]])
        alpha, thinning, residual_mean, residual_variance = (
            core.estimate_alpha_and_residual_moments(
                data, mean, covariance, child=1, parent_mask=1
            )
        )
        np.testing.assert_allclose(alpha, [1.5])
        np.testing.assert_allclose(thinning, 1.5 * data[:, 0])
        self.assertAlmostEqual(residual_mean, 4.0)
        self.assertAlmostEqual(residual_variance, 4.5)
        self.assertGreater(alpha[0], 1.0)

    def test_singular_covariance_fallback_is_recorded(self) -> None:
        data = np.column_stack([np.arange(8), np.arange(8), np.arange(8) * 2])
        mean, covariance = core.empirical_moments(data)
        audit: Counter[str] = Counter()
        alpha, _, _, _ = core.estimate_alpha_and_residual_moments(
            data, mean, covariance, child=2, parent_mask=0b011, audit=audit
        )
        self.assertTrue(np.all(alpha >= 0.0))
        self.assertEqual(
            audit["alpha_singular_or_ill_conditioned_lstsq_fallback"], 1
        )

    def test_local_bic_hand_calculation(self) -> None:
        data = np.array([[0], [1], [2], [1], [3], [1]], dtype=int)
        mean, covariance = core.empirical_moments(data)
        fit = core.local_score(data, mean, covariance, 0, 0, ("Poisson",))
        loglik = core.convolution_loglik(
            data[:, 0], np.zeros(len(data)), "Poisson", {"lam": mean[0]}
        )
        expected = -2.0 * loglik + math.log(len(data))
        self.assertAlmostEqual(fit.bic, expected, places=11)


class SearchAndIsolationTests(unittest.TestCase):
    def _assert_formal_oracle_regression(
        self,
        replicate: int,
        alpha_low: float,
        alpha_high: float,
        expected_seed: int,
    ) -> None:
        task = core.ReplicateTask(
            d=4,
            n=3200,
            replicate=replicate,
            methods=("OracleDP",),
            max_parents=5,
            avg_indegree=1.5,
            dag_generation_mode="exact_edges",
            alpha_low=alpha_low,
            alpha_high=alpha_high,
            base_seed=20260622,
            blas_threads=1,
            parallel_workers=1,
        )
        output = core.run_replicate_task(task)
        self.assertEqual(len(output.rows), 1)
        row = output.rows[0]
        self.assertEqual(int(row["seed"]), expected_seed)
        self.assertTrue(np.isfinite(float(row["score"])))
        masks = {node: 0 for node in range(4)}
        serialized = str(row["estimated_directed_edges"])
        if serialized:
            for edge in serialized.split(";"):
                parent, child = (int(value) for value in edge.split("->"))
                masks[child] |= 1 << parent
        self.assertTrue(core.is_acyclic(masks, 4))

    def test_formal_restricted_rep20_oracle_regression(self) -> None:
        self._assert_formal_oracle_regression(
            replicate=20,
            alpha_low=0.15,
            alpha_high=0.85,
            expected_seed=14658602722261584593,
        )

    def test_formal_extended_rep92_oracle_regression(self) -> None:
        self._assert_formal_oracle_regression(
            replicate=92,
            alpha_low=0.2,
            alpha_high=2.0,
            expected_seed=16608038829367702311,
        )

    def test_full_parent_set_larger_than_five_is_scored(self) -> None:
        rng = np.random.default_rng(1001)
        data = rng.poisson(2.0, size=(120, 7))
        spec = core.EstimatorSpec(("Poisson",))
        scores, _ = core.precompute_local_scores(data, estimator_spec=spec)
        mask = sum(1 << parent for parent in range(6))
        self.assertTrue(np.isfinite(scores[6][mask]))

    def test_exact_dp_equals_brute_force_for_d3_and_d4(self) -> None:
        rng = np.random.default_rng(733)
        for d in (3, 4):
            scores = [[math.inf] * (1 << d) for _ in range(d)]
            for child in range(d):
                for mask in range(1 << d):
                    if not ((mask >> child) & 1):
                        scores[child][mask] = float(rng.normal())
            masks, dp_score = core.exact_order_dp(scores)
            self.assertTrue(core.is_acyclic(masks, d))
            self.assertAlmostEqual(dp_score, all_dag_optimum(scores), places=11)

    def test_proposed_is_permutation_equivariant(self) -> None:
        data, _, _ = core.simulate_ptsem(
            d=3, n=350, seed=9191, avg_indegree=1.0, max_parents=2,
            dag_generation_mode="exact_edges",
        )
        spec = core.EstimatorSpec(("Poisson",))
        original = core.fit_proposed(data, spec)
        permutation = np.array([2, 0, 1])
        permuted = core.fit_proposed(data[:, permutation], spec)
        inverse = {old: new for new, old in enumerate(permutation)}
        expected = {(inverse[a], inverse[b]) for a, b in original.directed_edges}
        self.assertEqual(permuted.directed_edges, expected)
        self.assertAlmostEqual(permuted.score, original.score, places=8)

    def test_public_proposed_and_greedy_apis_cannot_receive_truth(self) -> None:
        self.assertEqual(list(inspect.signature(core.fit_proposed).parameters), [
            "X", "estimator_spec"
        ])
        self.assertEqual(list(inspect.signature(core.fit_greedy).parameters), [
            "X", "estimator_spec"
        ])
        self.assertEqual(list(inspect.signature(core.fit_oracle).parameters), [
            "X", "true_families", "estimator_spec"
        ])
        proposed_source = inspect.getsource(core._fit_proposed_with_diagnostics)
        self.assertNotIn("true_", proposed_source)
        self.assertNotIn("max_parents", proposed_source)

    def test_dgp_same_seed_is_exactly_reproducible(self) -> None:
        first = core.simulate_ptsem(
            5, 120, 331, 1.5, 3, 0.15, 1.35, "exact_edges"
        )
        second = core.simulate_ptsem(
            5, 120, 331, 1.5, 3, 0.15, 1.35, "exact_edges"
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_one_vs_two_workers_preserves_every_nonruntime_result(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ptsem_workers_") as temporary:
            root = Path(temporary)
            kwargs = dict(
                d_values=(3,), n=90, reps=2, methods=("LibraryDP",),
                max_parents=2, avg_indegree=1.0, seed=8103,
                require_all_methods=True, alpha_low=0.15, alpha_high=0.85,
                resume=False, dag_generation_mode="exact_edges",
            )
            single, _ = core.run_experiment(
                outdir=str(root / "single"), workers=1, **kwargs
            )
            parallel, _ = core.run_experiment(
                outdir=str(root / "parallel"), workers=2, **kwargs
            )
        ignored = {
            "runtime_sec", "local_score_runtime_sec", "search_runtime_sec",
            "parallel_workers",
        }
        columns = [column for column in single.columns if column not in ignored]
        left = single[columns].sort_values(["d", "rep", "method"]).reset_index(drop=True)
        right = parallel[columns].sort_values(["d", "rep", "method"]).reset_index(drop=True)
        for column in columns:
            if pd.api.types.is_numeric_dtype(left[column]):
                np.testing.assert_allclose(
                    left[column].to_numpy(float), right[column].to_numpy(float),
                    rtol=0.0, atol=1e-12, equal_nan=True,
                )
            else:
                self.assertEqual(
                    left[column].fillna("").astype(str).tolist(),
                    right[column].fillna("").astype(str).tolist(),
                    column,
                )


class NBAAndBaselineTests(unittest.TestCase):
    def test_nba_binomial_uses_single_paper_moment_index(self) -> None:
        data = np.array([[0], [1], [2], [3], [2], [1]], dtype=int)
        mean, covariance = core.empirical_moments(data)
        expected, expected_extension = core.binomial_moment_map(
            float(mean[0]),
            float(covariance[0, 0]),
            binomial_n_max=NBA.NBA_BINOMIAL_N_MAX,
        )
        trials, used_extension, residual_mean, residual_variance = (
            NBA.nba_binomial_trial_index(
                core, data, mean, covariance, child=0, parent_mask=0
            )
        )
        self.assertEqual(trials, expected["n"])
        self.assertEqual(used_extension, expected_extension)
        self.assertAlmostEqual(residual_mean, float(mean[0]))
        self.assertAlmostEqual(residual_variance, float(covariance[0, 0]))
        source = inspect.getsource(NBA.nba_binomial_trial_index)
        self.assertNotIn("DGP_", source)
        self.assertNotIn("20", source)
        self.assertNotIn("sqrt", source)
        self.assertNotIn("max(y)", source)

    def test_nba_binomial_fixed_cap100_boundaries(self) -> None:
        cases = (
            (51, 0.4, 51, 0.4),
            (100, 0.4, 100, 0.4),
            (101, 0.4, 100, 0.404),
            (500, 0.05, 100, 0.25),
        )
        self.assertEqual(NBA.NBA_BINOMIAL_N_MIN, 2)
        self.assertEqual(NBA.NBA_BINOMIAL_N_MAX, 100)
        for raw_n, probability, expected_n, expected_p in cases:
            with self.subTest(raw_n=raw_n):
                mean = raw_n * probability
                variance = raw_n * probability * (1.0 - probability)
                with mock.patch.object(
                    core,
                    "estimate_alpha_and_residual_moments",
                    return_value=(np.zeros(0), [], mean, variance),
                ):
                    trials, _, returned_mean, returned_variance = (
                        NBA.nba_binomial_trial_index(
                            core,
                            np.zeros((2, 1), dtype=int),
                            np.zeros(1),
                            np.zeros((1, 1)),
                            child=0,
                            parent_mask=0,
                        )
                    )
                self.assertEqual(trials, expected_n)
                self.assertAlmostEqual(returned_mean, mean)
                self.assertAlmostEqual(returned_variance, variance)
                params, used_extension = core.binomial_moment_map(
                    mean,
                    variance,
                    binomial_n_max=NBA.NBA_BINOMIAL_N_MAX,
                )
                self.assertFalse(used_extension)
                self.assertEqual(params["n"], expected_n)
                self.assertAlmostEqual(params["p"], expected_p)
                self.assertEqual(
                    core._validated_family_params(
                        "Binomial",
                        params,
                        binomial_n_max=NBA.NBA_BINOMIAL_N_MAX,
                    ),
                    params,
                )
        with self.assertRaises(ValueError):
            core._validated_family_params(
                "Binomial", {"n": 101, "p": 0.2}, binomial_n_max=100
            )

    def test_nba_keeps_full_mle_and_q_binomial_one(self) -> None:
        self.assertIn("minimize", inspect.getsource(NBA.optimize_family_for_trials))
        self.assertEqual(core.family_dimension("Binomial"), 1)
        data = np.array([[0], [1], [2], [3], [2], [1]], dtype=int)
        mean, covariance = core.empirical_moments(data)
        fit = NBA.optimization_family_fit(
            core, data, mean, covariance, 0, 0, "Binomial", 1, 15, 1
        )
        self.assertEqual(fit.parameter_count, 1)
        self.assertEqual(fit.diagnostics["binomial_n_candidates"], 1)
        self.assertEqual(
            fit.diagnostics["binomial_n_policy"],
            "paper_H_binomial_discrete_index_continuous_full_mle",
        )
        self.assertEqual(
            fit.diagnostics["binomial_n_tie_break"],
            "not_applicable_single_mom_index",
        )

    def test_genuine_ods_three_stage_smoke(self) -> None:
        rng = np.random.default_rng(2025)
        x0 = rng.poisson(2.0, 180)
        x1 = rng.poisson(0.6 * x0) + rng.poisson(1.2, 180)
        x2 = rng.poisson(0.4 * x1) + rng.poisson(0.8, 180)
        estimate = core.run_poisson_dag_ods_baseline(
            np.column_stack([x0, x1, x2])
        )
        self.assertTrue(core.is_acyclic(
            {
                node: sum(1 << parent for parent, child in estimate.directed_edges
                          if child == node)
                for node in range(3)
            },
            3,
        ))
        self.assertEqual(
            estimate.diagnostics["algorithm"],
            "Park-Raskutti ODS Algorithm 1",
        )
        self.assertEqual(len(estimate.diagnostics["stages"]), 3)

    def test_zip_helper_uses_selected_parameterization(self) -> None:
        self.assertAlmostEqual(
            NBA_SUMMARY.exogenous_mean(
                {"family": "ZIP", "params": {"lam": 4.0, "rho": 0.25}}
            ),
            3.0,
        )

    def test_execution_paths_use_the_public_nba_core(self) -> None:
        formal_paths = (
            ROOT / "reproduce.py",
            ROOT / "scripts/summarize_nba.py",
            ROOT / "experiments/nba/scripts/run_adaptive_ptsem_real_study.py",
            ROOT / "experiments/nba/scripts/run_expanded_candidate_optimization.py",
            ROOT / "experiments/nba/scripts/run_expanded_candidate_baselines.py",
            ROOT / "experiments/nba/scripts/run_four_var_baseline_comparison.py",
        )
        for path in formal_paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("hyp1f1", text, path)
            self.assertNotIn("experiments/nba/core/d.py", text.replace("\\", "/"), path)


if __name__ == "__main__":
    unittest.main()
