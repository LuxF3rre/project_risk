"""Shared fixtures for project_risk tests."""

import numpy as np
import pytest

from project_risk.models import PertEstimate, Task, TaskDependency


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def symmetric_estimate() -> PertEstimate:
    """Symmetric PERT estimate: mode at midpoint."""
    return PertEstimate(optimistic=5.0, most_likely=10.0, pessimistic=15.0)


@pytest.fixture
def skewed_estimate() -> PertEstimate:
    """Right-skewed PERT estimate."""
    return PertEstimate(optimistic=2.0, most_likely=4.0, pessimistic=12.0)


@pytest.fixture
def point_estimate() -> PertEstimate:
    """Degenerate estimate where all values are equal."""
    return PertEstimate(optimistic=5.0, most_likely=5.0, pessimistic=5.0)


@pytest.fixture
def linear_tasks() -> list[Task]:
    """Three sequential tasks."""
    return [
        Task(
            task_id="A",
            estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=5),
        ),
        Task(
            task_id="B",
            estimate=PertEstimate(optimistic=2, most_likely=3, pessimistic=6),
        ),
        Task(
            task_id="C",
            estimate=PertEstimate(optimistic=1, most_likely=4, pessimistic=7),
        ),
    ]


@pytest.fixture
def diamond_tasks() -> list[Task]:
    """Diamond DAG: A -> B, A -> C, B -> D, C -> D."""
    return [
        Task(
            task_id="A",
            estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
        ),
        Task(
            task_id="B",
            estimate=PertEstimate(optimistic=3, most_likely=5, pessimistic=9),
        ),
        Task(
            task_id="C",
            estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
        ),
        Task(
            task_id="D",
            estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
        ),
    ]


@pytest.fixture
def diamond_deps() -> list[TaskDependency]:
    """Dependencies for the diamond DAG."""
    return [
        TaskDependency(predecessor="A", successor="B"),
        TaskDependency(predecessor="A", successor="C"),
        TaskDependency(predecessor="B", successor="D"),
        TaskDependency(predecessor="C", successor="D"),
    ]
