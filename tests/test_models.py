"""Tests for project_risk.models."""

import numpy as np
import pytest

from project_risk.models import (
    DEFAULT_ITERATIONS,
    DEFAULT_LAMBDA,
    STANDARD_PERCENTILES,
    PercentileResult,
    PertEstimate,
    SimulationConfig,
    SimulationResult,
    Task,
    TaskDependency,
)


class TestPertEstimate:
    def test_valid_symmetric(self) -> None:
        e = PertEstimate(optimistic=1, most_likely=5, pessimistic=10)
        assert e.optimistic == 1
        assert e.most_likely == 5
        assert e.pessimistic == 10

    def test_valid_point(self) -> None:
        e = PertEstimate(optimistic=3, most_likely=3, pessimistic=3)
        assert e.optimistic == e.most_likely == e.pessimistic == 3

    def test_negative_optimistic_raises(self) -> None:
        with pytest.raises(ValueError, match="optimistic must be >= 0"):
            PertEstimate(optimistic=-1, most_likely=5, pessimistic=10)

    def test_wrong_ordering_raises(self) -> None:
        with pytest.raises(ValueError, match="must satisfy"):
            PertEstimate(optimistic=10, most_likely=5, pessimistic=1)

    def test_most_likely_below_optimistic_raises(self) -> None:
        with pytest.raises(ValueError, match="must satisfy"):
            PertEstimate(optimistic=5, most_likely=3, pessimistic=10)

    def test_most_likely_above_pessimistic_raises(self) -> None:
        with pytest.raises(ValueError, match="must satisfy"):
            PertEstimate(optimistic=1, most_likely=11, pessimistic=10)

    def test_equal_bounds_different_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="must satisfy"):
            PertEstimate(optimistic=5, most_likely=3, pessimistic=5)

    def test_frozen(self) -> None:
        e = PertEstimate(optimistic=1, most_likely=2, pessimistic=3)
        with pytest.raises(AttributeError):
            e.optimistic = 99  # type: ignore[misc]


class TestTask:
    def test_valid_task(self) -> None:
        e = PertEstimate(optimistic=1, most_likely=2, pessimistic=3)
        t = Task(task_id="A", estimate=e)
        assert t.task_id == "A"
        assert t.estimate == e

    def test_empty_task_id_raises(self) -> None:
        e = PertEstimate(optimistic=1, most_likely=2, pessimistic=3)
        with pytest.raises(ValueError, match="non-empty"):
            Task(task_id="   ", estimate=e)


class TestTaskDependency:
    def test_valid_dependency(self) -> None:
        dep = TaskDependency(predecessor="A", successor="B")
        assert dep.predecessor == "A"
        assert dep.successor == "B"

    def test_self_dependency_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot depend on itself"):
            TaskDependency(predecessor="A", successor="A")


class TestSimulationConfig:
    def test_defaults(self) -> None:
        cfg = SimulationConfig()
        assert cfg.iterations == DEFAULT_ITERATIONS
        assert cfg.seed is None
        assert cfg.pert_lambda == DEFAULT_LAMBDA

    def test_custom_values(self) -> None:
        cfg = SimulationConfig(iterations=5000, seed=42, pert_lambda=6.0)
        assert cfg.iterations == 5000
        assert cfg.seed == 42
        assert cfg.pert_lambda == 6.0

    def test_zero_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            SimulationConfig(iterations=0)

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="pert_lambda must be > 0"):
            SimulationConfig(pert_lambda=-1.0)

    def test_zero_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="pert_lambda must be > 0"):
            SimulationConfig(pert_lambda=0.0)


class TestPercentileResult:
    def test_creation(self) -> None:
        pr = PercentileResult(percentile=50.0, value=10.5)
        assert pr.percentile == 50.0
        assert pr.value == 10.5


class TestSimulationResult:
    def test_creation(self) -> None:
        samples = np.array([1.0, 2.0, 3.0])
        pcts = (PercentileResult(percentile=50.0, value=2.0),)
        sr = SimulationResult(
            samples=samples, percentiles=pcts, mean=2.0, std_dev=0.816
        )
        assert sr.mean == 2.0
        assert len(sr.percentiles) == 1
        assert sr.samples is samples


class TestConstants:
    def test_standard_percentiles(self) -> None:
        assert STANDARD_PERCENTILES == (5, 25, 50, 75, 95)

    def test_default_lambda(self) -> None:
        assert DEFAULT_LAMBDA == 4.0

    def test_default_iterations(self) -> None:
        assert DEFAULT_ITERATIONS == 10_000
