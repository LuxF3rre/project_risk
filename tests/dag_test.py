"""Tests for project_risk.dag."""

import networkx as nx
import numpy as np
import pytest

from project_risk.dag import (
    build_dag,
    build_linear_dag,
    dag_critical_path_durations,
    find_critical_path,
    validate_dag,
)
from project_risk.models import PertEstimate, Task, TaskDependency


class TestBuildDag:
    def test_simple_graph(self, linear_tasks: list[Task]) -> None:
        deps = [
            TaskDependency(predecessor="A", successor="B"),
            TaskDependency(predecessor="B", successor="C"),
        ]
        g = build_dag(tasks=linear_tasks, dependencies=deps)
        assert set(g.nodes) == {"A", "B", "C"}
        assert set(g.edges) == {("A", "B"), ("B", "C")}

    def test_no_dependencies(self, linear_tasks: list[Task]) -> None:
        g = build_dag(tasks=linear_tasks, dependencies=[])
        assert set(g.nodes) == {"A", "B", "C"}
        assert len(g.edges) == 0

    def test_unknown_predecessor_raises(self, linear_tasks: list[Task]) -> None:
        deps = [TaskDependency(predecessor="X", successor="A")]
        with pytest.raises(ValueError, match="unknown predecessor"):
            build_dag(tasks=linear_tasks, dependencies=deps)

    def test_unknown_successor_raises(self, linear_tasks: list[Task]) -> None:
        deps = [TaskDependency(predecessor="A", successor="Z")]
        with pytest.raises(ValueError, match="unknown successor"):
            build_dag(tasks=linear_tasks, dependencies=deps)


class TestValidateDag:
    def test_valid_dag(self, linear_tasks: list[Task]) -> None:
        deps = [
            TaskDependency(predecessor="A", successor="B"),
            TaskDependency(predecessor="B", successor="C"),
        ]
        g = build_dag(tasks=linear_tasks, dependencies=deps)
        errors = validate_dag(graph=g)
        assert errors == []

    def test_cycle_detected(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        errors = validate_dag(graph=g)
        assert any("cycle" in e for e in errors)

    def test_isolated_nodes(self) -> None:
        g = nx.DiGraph()
        g.add_nodes_from(["A", "B", "C"])
        g.add_edge("A", "B")
        errors = validate_dag(graph=g)
        assert any("isolated" in e for e in errors)

    def test_single_node_not_isolated(self) -> None:
        g = nx.DiGraph()
        g.add_node("A")
        errors = validate_dag(graph=g)
        assert errors == []


class TestDagCriticalPathDurations:
    def test_linear_chain(self) -> None:
        task_ids = ["A", "B", "C"]
        deps = [
            TaskDependency(predecessor="A", successor="B"),
            TaskDependency(predecessor="B", successor="C"),
        ]
        # Deterministic durations: A=2, B=3, C=4 → total=9
        matrix = np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
        result = dag_critical_path_durations(
            task_ids=task_ids, dependencies=deps, duration_matrix=matrix
        )
        np.testing.assert_array_almost_equal(result, [9.0, 6.0])

    def test_diamond_dag(self, diamond_deps: list[TaskDependency]) -> None:
        task_ids = ["A", "B", "C", "D"]
        # A=1, B=5, C=2, D=1 → critical: A->B->D = 1+5+1=7
        matrix = np.array([[1.0, 5.0, 2.0, 1.0]])
        result = dag_critical_path_durations(
            task_ids=task_ids, dependencies=diamond_deps, duration_matrix=matrix
        )
        np.testing.assert_array_almost_equal(result, [7.0])

    def test_parallel_paths(self) -> None:
        task_ids = ["start", "fast", "slow", "end"]
        deps = [
            TaskDependency(predecessor="start", successor="fast"),
            TaskDependency(predecessor="start", successor="slow"),
            TaskDependency(predecessor="fast", successor="end"),
            TaskDependency(predecessor="slow", successor="end"),
        ]
        # start=1, fast=2, slow=10, end=1 → critical: start->slow->end = 12
        matrix = np.array([[1.0, 2.0, 10.0, 1.0]])
        result = dag_critical_path_durations(
            task_ids=task_ids, dependencies=deps, duration_matrix=matrix
        )
        np.testing.assert_array_almost_equal(result, [12.0])


class TestBuildLinearDag:
    def test_linear_structure(self, linear_tasks: list[Task]) -> None:
        g = build_linear_dag(tasks=linear_tasks)
        assert set(g.edges) == {("A", "B"), ("B", "C")}

    def test_single_task(self) -> None:
        tasks = [
            Task(
                task_id="X",
                estimate=PertEstimate(optimistic=1, most_likely=2, pessimistic=3),
            )
        ]
        g = build_linear_dag(tasks=tasks)
        assert set(g.nodes) == {"X"}
        assert len(g.edges) == 0


class TestFindCriticalPath:
    def test_linear_path(self) -> None:
        task_ids = ["A", "B", "C"]
        deps = [
            TaskDependency(predecessor="A", successor="B"),
            TaskDependency(predecessor="B", successor="C"),
        ]
        durations = [2.0, 3.0, 4.0]
        cp = find_critical_path(
            task_ids=task_ids, dependencies=deps, durations=durations
        )
        assert cp == ["A", "B", "C"]

    def test_diamond_critical_path(self, diamond_deps: list[TaskDependency]) -> None:
        task_ids = ["A", "B", "C", "D"]
        # B is slower → A->B->D is critical
        durations = [1.0, 5.0, 2.0, 1.0]
        cp = find_critical_path(
            task_ids=task_ids, dependencies=diamond_deps, durations=durations
        )
        assert cp == ["A", "B", "D"]
