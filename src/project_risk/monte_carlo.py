"""Monte Carlo simulation orchestration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from project_risk.dag import dag_critical_path_durations
from project_risk.models import (
    STANDARD_PERCENTILES,
    PercentileResult,
    PertEstimate,
    SimulationConfig,
    SimulationResult,
    Task,
    TaskDependency,
)
from project_risk.pert import sample_pert, sample_pert_batch

__all__ = [
    "compute_result",
    "simulate_single",
    "simulate_task_dag",
    "simulate_task_list",
]


def compute_result(
    *,
    samples: NDArray[np.floating],
    percentile_values: tuple[float, ...] = STANDARD_PERCENTILES,
) -> SimulationResult:
    """Compute summary statistics from raw simulation samples.

    Args:
        samples: 1-D array of simulated project durations.
        percentile_values: Percentiles to compute.

    Returns:
        A SimulationResult with percentiles, mean, and std_dev.
    """
    raw_pcts = np.percentile(samples, percentile_values)
    percentiles = tuple(
        PercentileResult(percentile=p, value=float(v))
        for p, v in zip(percentile_values, raw_pcts, strict=True)
    )
    return SimulationResult(
        samples=samples,
        percentiles=percentiles,
        mean=float(np.mean(samples)),
        std_dev=float(np.std(samples)),
    )


def simulate_single(
    *,
    estimate: PertEstimate,
    config: SimulationConfig,
) -> SimulationResult:
    """Run Monte Carlo simulation for a single PERT estimate (UC1).

    Args:
        estimate: Three-point PERT estimate.
        config: Simulation configuration.

    Returns:
        SimulationResult with sampled durations.
    """
    rng = np.random.default_rng(config.seed)
    samples = sample_pert(
        estimate=estimate,
        size=config.iterations,
        rng=rng,
        pert_lambda=config.pert_lambda,
    )
    return compute_result(samples=samples)


def simulate_task_list(
    *,
    tasks: list[Task],
    config: SimulationConfig,
) -> SimulationResult:
    """Run Monte Carlo simulation for sequential tasks (UC2).

    Total duration is the sum of all task durations.

    Args:
        tasks: Ordered list of tasks.
        config: Simulation configuration.

    Returns:
        SimulationResult with total project durations.
    """
    rng = np.random.default_rng(config.seed)
    estimates = [t.estimate for t in tasks]
    matrix = sample_pert_batch(
        estimates=estimates,
        size=config.iterations,
        rng=rng,
        pert_lambda=config.pert_lambda,
    )
    total = matrix.sum(axis=1)
    return compute_result(samples=total)


def simulate_task_dag(
    *,
    tasks: list[Task],
    dependencies: list[TaskDependency],
    config: SimulationConfig,
) -> SimulationResult:
    """Run Monte Carlo simulation with DAG-based critical path (UC3).

    Args:
        tasks: List of tasks.
        dependencies: Dependency edges.
        config: Simulation configuration.

    Returns:
        SimulationResult with critical-path project durations.
    """
    rng = np.random.default_rng(config.seed)
    estimates = [t.estimate for t in tasks]
    task_ids = [t.task_id for t in tasks]
    matrix = sample_pert_batch(
        estimates=estimates,
        size=config.iterations,
        rng=rng,
        pert_lambda=config.pert_lambda,
    )
    durations = dag_critical_path_durations(
        task_ids=task_ids,
        dependencies=dependencies,
        duration_matrix=matrix,
    )
    return compute_result(samples=durations)
