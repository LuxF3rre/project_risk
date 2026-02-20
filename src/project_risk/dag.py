"""DAG construction, validation, and vectorized critical-path computation."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from project_risk.models import Task, TaskDependency

__all__ = [
    "build_dag",
    "build_linear_dag",
    "dag_critical_path_durations",
    "find_critical_path",
    "validate_dag",
]


def build_dag(
    *,
    tasks: list[Task],
    dependencies: list[TaskDependency],
) -> nx.DiGraph:
    """Build a directed acyclic graph from tasks and dependencies.

    Args:
        tasks: List of tasks (nodes).
        dependencies: List of dependency edges.

    Returns:
        A networkx DiGraph with task_ids as nodes.

    Raises:
        ValueError: If a dependency references an unknown task_id.
    """
    task_ids = {t.task_id for t in tasks}
    graph = nx.DiGraph()
    graph.add_nodes_from(task_ids)

    for dep in dependencies:
        if dep.predecessor not in task_ids:
            msg = f"unknown predecessor task_id: {dep.predecessor!r}"
            raise ValueError(msg)
        if dep.successor not in task_ids:
            msg = f"unknown successor task_id: {dep.successor!r}"
            raise ValueError(msg)
        graph.add_edge(dep.predecessor, dep.successor)

    return graph


def validate_dag(*, graph: nx.DiGraph) -> list[str]:
    """Validate a DAG for common issues.

    Args:
        graph: The directed graph to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if not nx.is_directed_acyclic_graph(graph):
        cycles = list(nx.simple_cycles(graph))
        for cycle in cycles:
            errors.append(f"cycle detected: {' -> '.join(cycle)}")

    isolated = list(nx.isolates(graph))
    if isolated and graph.number_of_nodes() > 1:
        errors.append(f"isolated nodes: {', '.join(sorted(isolated))}")

    return errors


def dag_critical_path_durations(
    *,
    task_ids: list[str],
    dependencies: list[TaskDependency],
    duration_matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute project duration across all MC iterations via vectorized DP.

    Uses forward-pass earliest-finish calculation in topological order.

    Args:
        task_ids: Ordered list of task identifiers.
        dependencies: Dependency edges.
        duration_matrix: Shape (iterations, n_tasks) sampled durations.

    Returns:
        1-D array of shape (iterations,) with project durations.
    """
    n_iter, n_tasks = duration_matrix.shape
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_tasks))
    for dep in dependencies:
        graph.add_edge(id_to_idx[dep.predecessor], id_to_idx[dep.successor])

    topo_order = list(nx.topological_sort(graph))

    earliest_finish = np.zeros((n_iter, n_tasks))

    for node in topo_order:
        preds = list(graph.predecessors(node))
        if preds:
            max_pred_finish = np.max(earliest_finish[:, preds], axis=1)
        else:
            max_pred_finish = np.zeros(n_iter)
        earliest_finish[:, node] = max_pred_finish + duration_matrix[:, node]

    return np.max(earliest_finish, axis=1)


def build_linear_dag(*, tasks: list[Task]) -> nx.DiGraph:
    """Build a linear (sequential) DAG from an ordered task list.

    Args:
        tasks: Ordered list of tasks.

    Returns:
        A DiGraph where each task depends on the previous one.
    """
    deps = [
        TaskDependency(
            predecessor=tasks[i].task_id,
            successor=tasks[i + 1].task_id,
        )
        for i in range(len(tasks) - 1)
    ]
    return build_dag(tasks=tasks, dependencies=deps)


def find_critical_path(
    *,
    task_ids: list[str],
    dependencies: list[TaskDependency],
    durations: list[float],
) -> list[str]:
    """Find the deterministic critical path through the DAG.

    Args:
        task_ids: Ordered list of task identifiers.
        dependencies: Dependency edges.
        durations: Duration for each task (same order as task_ids).

    Returns:
        Ordered list of task_ids on the critical path.
    """
    n_tasks = len(task_ids)
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_tasks))
    for dep in dependencies:
        graph.add_edge(id_to_idx[dep.predecessor], id_to_idx[dep.successor])

    topo_order = list(nx.topological_sort(graph))

    # Forward pass: earliest start / earliest finish
    es = np.zeros(n_tasks)
    ef = np.zeros(n_tasks)
    for node in topo_order:
        preds = list(graph.predecessors(node))
        if preds:
            es[node] = max(ef[pred] for pred in preds)
        ef[node] = es[node] + durations[node]

    project_duration = max(ef)

    # Backward pass: latest start / latest finish
    ls = np.full(n_tasks, project_duration)
    lf = np.full(n_tasks, project_duration)
    for node in reversed(topo_order):
        succs = list(graph.successors(node))
        if succs:
            lf[node] = min(ls[succ] for succ in succs)
        ls[node] = lf[node] - durations[node]

    # Critical path: tasks with zero slack
    slack = ls - es
    critical_indices = [node for node in topo_order if np.isclose(slack[node], 0.0)]
    return [task_ids[i] for i in critical_indices]
