"""Shared fixtures for project_risk tests."""

import numpy as np
import pytest

from project_risk.models import PertEstimate, Task, TaskDependency, TaskTransition


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


@pytest.fixture
def workflow_tasks() -> list[Task]:
    """Software development workflow tasks."""
    return [
        Task(
            task_id="Requirements",
            estimate=PertEstimate(optimistic=2, most_likely=4, pessimistic=8),
        ),
        Task(
            task_id="Development",
            estimate=PertEstimate(optimistic=5, most_likely=10, pessimistic=20),
        ),
        Task(
            task_id="Code Review",
            estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=4),
        ),
        Task(
            task_id="Testing",
            estimate=PertEstimate(optimistic=2, most_likely=5, pessimistic=10),
        ),
        Task(
            task_id="Bug Fixes",
            estimate=PertEstimate(optimistic=1, most_likely=3, pessimistic=7),
        ),
        Task(
            task_id="User Acceptance",
            estimate=PertEstimate(optimistic=1, most_likely=3, pessimistic=5),
        ),
        Task(
            task_id="Deployment",
            estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=4),
        ),
    ]


@pytest.fixture
def workflow_transitions() -> list[TaskTransition]:
    """Transitions for the software development workflow."""
    return [
        TaskTransition(source="Requirements", target="Development", probability=1.0),
        TaskTransition(source="Development", target="Code Review", probability=1.0),
        TaskTransition(source="Code Review", target="Testing", probability=0.7),
        TaskTransition(source="Code Review", target="Development", probability=0.3),
        TaskTransition(source="Testing", target="User Acceptance", probability=0.6),
        TaskTransition(source="Testing", target="Bug Fixes", probability=0.4),
        TaskTransition(source="Bug Fixes", target="Testing", probability=1.0),
        TaskTransition(source="User Acceptance", target="Deployment", probability=0.8),
        TaskTransition(source="User Acceptance", target="Bug Fixes", probability=0.2),
    ]
