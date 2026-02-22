"""Domain models for project risk simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_LAMBDA",
    "DEFAULT_MAX_STEPS",
    "STANDARD_PERCENTILES",
    "PercentileResult",
    "PertEstimate",
    "SimulationConfig",
    "SimulationResult",
    "Task",
    "TaskDependency",
    "TaskTransition",
    "WorkflowResult",
]

DEFAULT_LAMBDA: Final = 4.0
DEFAULT_ITERATIONS: Final = 10_000
DEFAULT_MAX_STEPS: Final = 1000
STANDARD_PERCENTILES: Final[tuple[float, ...]] = (5, 25, 50, 75, 95)


@dataclass(frozen=True, slots=True)
class PertEstimate:
    """Three-point PERT estimate (optimistic, most likely, pessimistic)."""

    optimistic: float
    most_likely: float
    pessimistic: float

    def __post_init__(self) -> None:
        if self.optimistic < 0:
            msg = f"optimistic must be >= 0, got {self.optimistic}"
            raise ValueError(msg)
        if not (self.optimistic <= self.most_likely <= self.pessimistic):
            msg = (
                f"must satisfy optimistic <= most_likely <= pessimistic, "
                f"got {self.optimistic}, {self.most_likely}, {self.pessimistic}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Task:
    """A named task with a PERT estimate."""

    task_id: str
    estimate: PertEstimate

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            msg = "task_id must be a non-empty string"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TaskDependency:
    """A directed dependency edge: predecessor must finish before successor starts."""

    predecessor: str
    successor: str

    def __post_init__(self) -> None:
        if self.predecessor == self.successor:
            msg = f"task cannot depend on itself: {self.predecessor}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for a Monte Carlo simulation run."""

    iterations: int = DEFAULT_ITERATIONS
    seed: int | None = None
    pert_lambda: float = DEFAULT_LAMBDA

    def __post_init__(self) -> None:
        if self.iterations < 1:
            msg = f"iterations must be >= 1, got {self.iterations}"
            raise ValueError(msg)
        if self.pert_lambda <= 0:
            msg = f"pert_lambda must be > 0, got {self.pert_lambda}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PercentileResult:
    """A single percentile value from a simulation."""

    percentile: float
    value: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Aggregated results from a Monte Carlo simulation."""

    samples: NDArray[np.floating]
    percentiles: tuple[PercentileResult, ...]
    mean: float
    std_dev: float


@dataclass(frozen=True, slots=True)
class TaskTransition:
    """A probabilistic transition edge between two workflow tasks."""

    source: str
    target: str
    probability: float

    def __post_init__(self) -> None:
        if self.source == self.target:
            msg = f"self-loop not allowed: {self.source}"
            raise ValueError(msg)
        if not (0.0 < self.probability <= 1.0):
            msg = f"probability must be in (0, 1], got {self.probability}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Results from a cyclic workflow simulation."""

    duration_result: SimulationResult
    step_counts: NDArray[np.integer]
    visit_counts: dict[str, NDArray[np.integer]]
    visit_durations: dict[str, NDArray[np.floating]]
    max_steps_hit_count: int
