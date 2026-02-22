"""Tests for project_risk.workflow."""

import numpy as np
import pytest

from project_risk.models import (
    PertEstimate,
    SimulationConfig,
    Task,
    TaskTransition,
    WorkflowResult,
)
from project_risk.workflow import (
    build_workflow_graph,
    simulate_task_workflow,
    validate_workflow,
)


class TestBuildWorkflowGraph:
    def test_basic_construction(
        self,
        workflow_tasks: list[Task],
        workflow_transitions: list[TaskTransition],
    ) -> None:
        graph = build_workflow_graph(
            tasks=workflow_tasks, transitions=workflow_transitions
        )
        assert len(graph.nodes()) == len(workflow_tasks)
        assert len(graph.edges()) == len(workflow_transitions)

    def test_edge_probabilities(
        self,
        workflow_tasks: list[Task],
        workflow_transitions: list[TaskTransition],
    ) -> None:
        graph = build_workflow_graph(
            tasks=workflow_tasks, transitions=workflow_transitions
        )
        assert graph.edges["Code Review", "Testing"]["probability"] == 0.7
        assert graph.edges["Code Review", "Development"]["probability"] == 0.3

    def test_unknown_source_raises(self) -> None:
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
        ]
        transitions = [TaskTransition(source="X", target="A", probability=1.0)]
        with pytest.raises(ValueError, match="unknown source"):
            build_workflow_graph(tasks=tasks, transitions=transitions)

    def test_unknown_target_raises(self) -> None:
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
        ]
        transitions = [TaskTransition(source="A", target="X", probability=1.0)]
        with pytest.raises(ValueError, match="unknown target"):
            build_workflow_graph(tasks=tasks, transitions=transitions)


class TestValidateWorkflow:
    def test_valid_workflow(
        self,
        workflow_tasks: list[Task],
        workflow_transitions: list[TaskTransition],
    ) -> None:
        graph = build_workflow_graph(
            tasks=workflow_tasks, transitions=workflow_transitions
        )
        errors = validate_workflow(graph=graph, start_nodes=["Requirements"])
        assert errors == []

    def test_missing_start_node(
        self,
        workflow_tasks: list[Task],
        workflow_transitions: list[TaskTransition],
    ) -> None:
        graph = build_workflow_graph(
            tasks=workflow_tasks, transitions=workflow_transitions
        )
        errors = validate_workflow(graph=graph, start_nodes=["Nonexistent"])
        assert any("start node" in e for e in errors)

    def test_empty_start_nodes(
        self,
        workflow_tasks: list[Task],
        workflow_transitions: list[TaskTransition],
    ) -> None:
        graph = build_workflow_graph(
            tasks=workflow_tasks, transitions=workflow_transitions
        )
        errors = validate_workflow(graph=graph, start_nodes=[])
        assert any("at least one" in e for e in errors)

    def test_probability_sum_error(self) -> None:
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
            Task(
                task_id="B",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
            Task(
                task_id="C",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
        ]
        transitions = [
            TaskTransition(source="A", target="B", probability=0.5),
            TaskTransition(source="A", target="C", probability=0.3),
        ]
        graph = build_workflow_graph(tasks=tasks, transitions=transitions)
        errors = validate_workflow(graph=graph, start_nodes=["A"])
        assert any("sum to" in e for e in errors)

    def test_unreachable_nodes(self) -> None:
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
            Task(
                task_id="B",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
            Task(
                task_id="C",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
        ]
        transitions = [
            TaskTransition(source="A", target="B", probability=1.0),
        ]
        graph = build_workflow_graph(tasks=tasks, transitions=transitions)
        errors = validate_workflow(graph=graph, start_nodes=["A"])
        assert any("unreachable" in e for e in errors)

    def test_multiple_starts_cover_unreachable(self) -> None:
        """Two start nodes together reach all nodes."""
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
            Task(
                task_id="B",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
            Task(
                task_id="C",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
        ]
        transitions = [
            TaskTransition(source="A", target="B", probability=1.0),
        ]
        graph = build_workflow_graph(tasks=tasks, transitions=transitions)
        errors = validate_workflow(graph=graph, start_nodes=["A", "C"])
        assert not any("unreachable" in e for e in errors)

    def test_no_terminal_node(self) -> None:
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
            Task(
                task_id="B",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
        ]
        transitions = [
            TaskTransition(source="A", target="B", probability=1.0),
            TaskTransition(source="B", target="A", probability=1.0),
        ]
        graph = build_workflow_graph(tasks=tasks, transitions=transitions)
        errors = validate_workflow(graph=graph, start_nodes=["A"])
        assert any("no terminal" in e for e in errors)


class TestSimulateTaskWorkflow:
    def test_result_structure(
        self,
        workflow_tasks: list[Task],
        workflow_transitions: list[TaskTransition],
    ) -> None:
        config = SimulationConfig(iterations=100, seed=42)
        result = simulate_task_workflow(
            tasks=workflow_tasks,
            transitions=workflow_transitions,
            start_nodes=["Requirements"],
            config=config,
        )
        assert isinstance(result, WorkflowResult)
        assert len(result.duration_result.samples) == 100
        assert len(result.step_counts) == 100
        assert "Requirements" in result.visit_counts
        assert "Deployment" in result.visit_durations

    def test_terminal_always_reached(
        self,
        workflow_tasks: list[Task],
        workflow_transitions: list[TaskTransition],
    ) -> None:
        config = SimulationConfig(iterations=500, seed=7)
        result = simulate_task_workflow(
            tasks=workflow_tasks,
            transitions=workflow_transitions,
            start_nodes=["Requirements"],
            config=config,
        )
        assert result.max_steps_hit_count == 0
        assert all(result.step_counts > 0)

    def test_linear_workflow_equals_sum(self) -> None:
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=5, most_likely=5, pessimistic=5),
            ),
            Task(
                task_id="B",
                estimate=PertEstimate(optimistic=3, most_likely=3, pessimistic=3),
            ),
            Task(
                task_id="C",
                estimate=PertEstimate(optimistic=2, most_likely=2, pessimistic=2),
            ),
        ]
        transitions = [
            TaskTransition(source="A", target="B", probability=1.0),
            TaskTransition(source="B", target="C", probability=1.0),
        ]
        config = SimulationConfig(iterations=10, seed=1)
        result = simulate_task_workflow(
            tasks=tasks,
            transitions=transitions,
            start_nodes=["A"],
            config=config,
        )
        np.testing.assert_allclose(result.duration_result.samples, 10.0)
        np.testing.assert_array_equal(result.step_counts, 3)

    def test_reproducibility(
        self,
        workflow_tasks: list[Task],
        workflow_transitions: list[TaskTransition],
    ) -> None:
        config = SimulationConfig(iterations=200, seed=99)
        r1 = simulate_task_workflow(
            tasks=workflow_tasks,
            transitions=workflow_transitions,
            start_nodes=["Requirements"],
            config=config,
        )
        r2 = simulate_task_workflow(
            tasks=workflow_tasks,
            transitions=workflow_transitions,
            start_nodes=["Requirements"],
            config=config,
        )
        np.testing.assert_array_equal(
            r1.duration_result.samples, r2.duration_result.samples
        )

    def test_max_steps_safety(self) -> None:
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=1, most_likely=1, pessimistic=1),
            ),
            Task(
                task_id="B",
                estimate=PertEstimate(optimistic=1, most_likely=1, pessimistic=1),
            ),
        ]
        transitions = [
            TaskTransition(source="A", target="B", probability=1.0),
            TaskTransition(source="B", target="A", probability=1.0),
        ]
        config = SimulationConfig(iterations=5, seed=42)
        result = simulate_task_workflow(
            tasks=tasks,
            transitions=transitions,
            start_nodes=["A"],
            config=config,
            max_steps=10,
        )
        assert result.max_steps_hit_count == 5
        np.testing.assert_array_equal(result.step_counts, 10)

    def test_invalid_start_node_raises(self) -> None:
        tasks = [
            Task(
                task_id="A",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            ),
        ]
        config = SimulationConfig(iterations=10, seed=1)
        with pytest.raises(ValueError, match="start node"):
            simulate_task_workflow(
                tasks=tasks,
                transitions=[],
                start_nodes=["Z"],
                config=config,
            )

    def test_single_task(self) -> None:
        tasks = [
            Task(
                task_id="Solo",
                estimate=PertEstimate(optimistic=5, most_likely=5, pessimistic=5),
            ),
        ]
        config = SimulationConfig(iterations=10, seed=1)
        result = simulate_task_workflow(
            tasks=tasks,
            transitions=[],
            start_nodes=["Solo"],
            config=config,
        )
        np.testing.assert_allclose(result.duration_result.samples, 5.0)
        np.testing.assert_array_equal(result.step_counts, 1)
        assert result.max_steps_hit_count == 0

    def test_parallel_starts_takes_max_duration(self) -> None:
        """Two parallel start nodes — duration is the max of both walks."""
        tasks = [
            Task(
                task_id="Fast",
                estimate=PertEstimate(optimistic=1, most_likely=1, pessimistic=1),
            ),
            Task(
                task_id="Slow",
                estimate=PertEstimate(optimistic=10, most_likely=10, pessimistic=10),
            ),
        ]
        config = SimulationConfig(iterations=10, seed=1)
        result = simulate_task_workflow(
            tasks=tasks,
            transitions=[],
            start_nodes=["Fast", "Slow"],
            config=config,
        )
        # Duration should be max(1, 10) = 10
        np.testing.assert_allclose(result.duration_result.samples, 10.0)
        # Steps should be 2 (one per start node)
        np.testing.assert_array_equal(result.step_counts, 2)
