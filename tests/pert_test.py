"""Tests for project_risk.pert."""

import numpy as np
import pytest

from project_risk.models import PertEstimate
from project_risk.pert import pert_alpha_beta, sample_pert, sample_pert_batch


class TestPertAlphaBeta:
    def test_symmetric(self, symmetric_estimate: PertEstimate) -> None:
        alpha, beta = pert_alpha_beta(estimate=symmetric_estimate)
        assert alpha == pytest.approx(3.0)
        assert beta == pytest.approx(3.0)

    def test_skewed(self, skewed_estimate: PertEstimate) -> None:
        alpha, beta = pert_alpha_beta(estimate=skewed_estimate)
        # alpha = 1 + 4*(4-2)/(12-2) = 1 + 0.8 = 1.8
        assert alpha == pytest.approx(1.8)
        # beta = 1 + 4*(12-4)/(12-2) = 1 + 3.2 = 4.2
        assert beta == pytest.approx(4.2)

    def test_point_estimate(self, point_estimate: PertEstimate) -> None:
        alpha, beta = pert_alpha_beta(estimate=point_estimate)
        assert alpha == 1.0
        assert beta == 1.0

    def test_custom_lambda(self, symmetric_estimate: PertEstimate) -> None:
        alpha, beta = pert_alpha_beta(estimate=symmetric_estimate, pert_lambda=6.0)
        # alpha = 1 + 6*(10-5)/(15-5) = 1 + 3 = 4.0
        assert alpha == pytest.approx(4.0)
        assert beta == pytest.approx(4.0)


class TestSamplePert:
    def test_samples_within_range(
        self, symmetric_estimate: PertEstimate, rng: np.random.Generator
    ) -> None:
        samples = sample_pert(estimate=symmetric_estimate, size=10_000, rng=rng)
        assert samples.min() >= symmetric_estimate.optimistic
        assert samples.max() <= symmetric_estimate.pessimistic

    def test_sample_shape(
        self, symmetric_estimate: PertEstimate, rng: np.random.Generator
    ) -> None:
        samples = sample_pert(estimate=symmetric_estimate, size=500, rng=rng)
        assert samples.shape == (500,)

    def test_mean_convergence_symmetric(
        self, symmetric_estimate: PertEstimate, rng: np.random.Generator
    ) -> None:
        samples = sample_pert(estimate=symmetric_estimate, size=100_000, rng=rng)
        # For symmetric PERT, mean should be close to most_likely
        assert samples.mean() == pytest.approx(10.0, abs=0.1)

    def test_point_estimate_constant(
        self, point_estimate: PertEstimate, rng: np.random.Generator
    ) -> None:
        samples = sample_pert(estimate=point_estimate, size=100, rng=rng)
        np.testing.assert_array_equal(samples, 5.0)

    def test_reproducibility(self, symmetric_estimate: PertEstimate) -> None:
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        s1 = sample_pert(estimate=symmetric_estimate, size=100, rng=rng1)
        s2 = sample_pert(estimate=symmetric_estimate, size=100, rng=rng2)
        np.testing.assert_array_equal(s1, s2)


class TestSamplePertBatch:
    def test_batch_shape(self, rng: np.random.Generator) -> None:
        estimates = [
            PertEstimate(optimistic=1, most_likely=2, pessimistic=5),
            PertEstimate(optimistic=3, most_likely=4, pessimistic=7),
        ]
        result = sample_pert_batch(estimates=estimates, size=1000, rng=rng)
        assert result.shape == (1000, 2)

    def test_batch_within_range(self, rng: np.random.Generator) -> None:
        estimates = [
            PertEstimate(optimistic=0, most_likely=5, pessimistic=10),
            PertEstimate(optimistic=2, most_likely=3, pessimistic=8),
        ]
        result = sample_pert_batch(estimates=estimates, size=5000, rng=rng)
        assert result[:, 0].min() >= 0
        assert result[:, 0].max() <= 10
        assert result[:, 1].min() >= 2
        assert result[:, 1].max() <= 8

    def test_empty_estimates(self, rng: np.random.Generator) -> None:
        result = sample_pert_batch(estimates=[], size=100, rng=rng)
        assert result.shape == (100, 0)
