"""PERT distribution sampling via Beta distribution."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from project_risk.models import DEFAULT_LAMBDA, PertEstimate

__all__ = [
    "pert_alpha_beta",
    "sample_pert",
    "sample_pert_batch",
]


def pert_alpha_beta(
    *,
    estimate: PertEstimate,
    pert_lambda: float = DEFAULT_LAMBDA,
) -> tuple[float, float]:
    """Compute Beta distribution shape parameters from a PERT estimate.

    Args:
        estimate: Three-point PERT estimate.
        pert_lambda: PERT weighting parameter (default 4.0).

    Returns:
        Tuple of (alpha, beta) shape parameters.
    """
    o, m, p = estimate.optimistic, estimate.most_likely, estimate.pessimistic
    spread = p - o
    if spread == 0:
        return (1.0, 1.0)
    alpha = 1.0 + pert_lambda * (m - o) / spread
    beta = 1.0 + pert_lambda * (p - m) / spread
    return (alpha, beta)


def sample_pert(
    *,
    estimate: PertEstimate,
    size: int,
    rng: np.random.Generator,
    pert_lambda: float = DEFAULT_LAMBDA,
) -> NDArray[np.floating]:
    """Sample from a single PERT distribution.

    Args:
        estimate: Three-point PERT estimate.
        size: Number of samples to draw.
        rng: NumPy random generator.
        pert_lambda: PERT weighting parameter.

    Returns:
        1-D array of samples in [optimistic, pessimistic].
    """
    o, p = estimate.optimistic, estimate.pessimistic
    if o == p:
        return np.full(size, o)
    alpha, beta = pert_alpha_beta(estimate=estimate, pert_lambda=pert_lambda)
    raw = rng.beta(alpha, beta, size=size)
    return o + raw * (p - o)


def sample_pert_batch(
    *,
    estimates: list[PertEstimate],
    size: int,
    rng: np.random.Generator,
    pert_lambda: float = DEFAULT_LAMBDA,
) -> NDArray[np.floating]:
    """Sample from multiple PERT distributions at once.

    Args:
        estimates: List of PERT estimates (one per task).
        size: Number of Monte Carlo iterations.
        rng: NumPy random generator.
        pert_lambda: PERT weighting parameter.

    Returns:
        2-D array of shape (size, len(estimates)).
    """
    n_tasks = len(estimates)
    result = np.empty((size, n_tasks))
    for i, est in enumerate(estimates):
        result[:, i] = sample_pert(
            estimate=est, size=size, rng=rng, pert_lambda=pert_lambda
        )
    return result
