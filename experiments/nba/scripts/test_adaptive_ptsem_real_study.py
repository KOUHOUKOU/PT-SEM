"""Focused regression tests for the adaptive-family NBA driver."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_adaptive_ptsem_real_study as real


REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_PATH = REPO_ROOT / "experiments/simulation/legacy/d.py"


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


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    unittest.main()
