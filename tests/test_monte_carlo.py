"""Tests for project_risk.monte_carlo."""

import numpy as np
import pytest

from project_risk.models import (
    PertEstimate,
    SimulationConfig,
    Task,
    TaskDependency,
)
from project_risk.monte_carlo import (
    compute_result,
    simulate_single,
    simulate_task_dag,
    simulate_task_list,
)


class TestComputeResult:
    def test_percentile_ordering(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(50, 10, size=10_000)
        result = compute_result(samples=samples)
        values = [p.value for p in result.percentiles]
        assert values == sorted(values)

    def test_standard_percentiles_count(self) -> None:
        samples = np.arange(1000, dtype=float)
        result = compute_result(samples=samples)
        assert len(result.percentiles) == 5

    def test_mean_and_std(self) -> None:
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_result(samples=samples)
        assert result.mean == pytest.approx(3.0)
        assert result.std_dev == pytest.approx(np.std(samples))

    def test_custom_percentiles(self) -> None:
        samples = np.arange(100, dtype=float)
        result = compute_result(samples=samples, percentile_values=(10, 90))
        assert len(result.percentiles) == 2
        assert result.percentiles[0].percentile == 10
        assert result.percentiles[1].percentile == 90


class TestSimulateSingle:
    def test_result_structure(self, symmetric_estimate: PertEstimate) -> None:
        cfg = SimulationConfig(iterations=1000, seed=42)
        result = simulate_single(estimate=symmetric_estimate, config=cfg)
        assert result.samples.shape == (1000,)
        assert len(result.percentiles) == 5
        assert result.mean > 0

    def test_samples_within_bounds(self, symmetric_estimate: PertEstimate) -> None:
        cfg = SimulationConfig(iterations=5000, seed=42)
        result = simulate_single(estimate=symmetric_estimate, config=cfg)
        assert result.samples.min() >= symmetric_estimate.optimistic
        assert result.samples.max() <= symmetric_estimate.pessimistic

    def test_reproducibility(self, symmetric_estimate: PertEstimate) -> None:
        cfg = SimulationConfig(iterations=500, seed=99)
        r1 = simulate_single(estimate=symmetric_estimate, config=cfg)
        r2 = simulate_single(estimate=symmetric_estimate, config=cfg)
        np.testing.assert_array_equal(r1.samples, r2.samples)


class TestSimulateTaskList:
    def test_sum_of_tasks(self, linear_tasks: list[Task]) -> None:
        cfg = SimulationConfig(iterations=5000, seed=42)
        result = simulate_task_list(tasks=linear_tasks, config=cfg)
        assert result.samples.shape == (5000,)
        # Sum of optimistic values = 1+2+1 = 4
        assert result.samples.min() >= 4.0
        # Sum of pessimistic values = 5+6+7 = 18
        assert result.samples.max() <= 18.0

    def test_single_task_equals_single_sim(self) -> None:
        est = PertEstimate(optimistic=2, most_likely=5, pessimistic=10)
        task = Task(task_id="X", estimate=est)
        cfg = SimulationConfig(iterations=1000, seed=42)
        list_result = simulate_task_list(tasks=[task], config=cfg)
        single_result = simulate_single(estimate=est, config=cfg)
        np.testing.assert_array_almost_equal(list_result.samples, single_result.samples)


class TestSimulateTaskDag:
    def test_dag_result_structure(
        self,
        diamond_tasks: list[Task],
        diamond_deps: list[TaskDependency],
    ) -> None:
        cfg = SimulationConfig(iterations=2000, seed=42)
        result = simulate_task_dag(
            tasks=diamond_tasks, dependencies=diamond_deps, config=cfg
        )
        assert result.samples.shape == (2000,)
        assert len(result.percentiles) == 5

    def test_dag_bounds(
        self,
        diamond_tasks: list[Task],
        diamond_deps: list[TaskDependency],
    ) -> None:
        cfg = SimulationConfig(iterations=5000, seed=42)
        result = simulate_task_dag(
            tasks=diamond_tasks, dependencies=diamond_deps, config=cfg
        )
        # Min critical path: A_opt + max(B_opt, C_opt) + D_opt = 1+3+1 = 5
        assert result.samples.min() >= 5.0
        # Max critical path: A_pes + max(B_pes, C_pes) + D_pes = 3+9+3 = 15
        assert result.samples.max() <= 15.0

    def test_linear_dag_matches_task_list(self, linear_tasks: list[Task]) -> None:
        deps = [
            TaskDependency(predecessor="A", successor="B"),
            TaskDependency(predecessor="B", successor="C"),
        ]
        cfg = SimulationConfig(iterations=1000, seed=42)
        dag_result = simulate_task_dag(
            tasks=linear_tasks, dependencies=deps, config=cfg
        )
        list_result = simulate_task_list(tasks=linear_tasks, config=cfg)
        # For a linear chain, critical path = sum of all tasks
        np.testing.assert_array_almost_equal(dag_result.samples, list_result.samples)
