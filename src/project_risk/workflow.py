"""Cyclic probabilistic graph simulation for task workflows."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from project_risk.models import (
    DEFAULT_MAX_STEPS,
    SimulationConfig,
    Task,
    TaskTransition,
    WorkflowResult,
)
from project_risk.monte_carlo import compute_result
from project_risk.pert import sample_pert

__all__ = [
    "build_workflow_graph",
    "simulate_task_workflow",
    "validate_workflow",
]


def build_workflow_graph(
    *,
    tasks: list[Task],
    transitions: list[TaskTransition],
) -> nx.DiGraph:
    """Build a directed graph with probability-weighted edges.

    Args:
        tasks: List of workflow tasks (nodes).
        transitions: List of probabilistic transition edges.

    Returns:
        A networkx DiGraph with ``probability`` edge attributes.

    Raises:
        ValueError: If a transition references an unknown task_id.
    """
    task_ids = {t.task_id for t in tasks}
    graph = nx.DiGraph()
    graph.add_nodes_from(task_ids)

    for tr in transitions:
        if tr.source not in task_ids:
            msg = f"unknown source task_id: {tr.source!r}"
            raise ValueError(msg)
        if tr.target not in task_ids:
            msg = f"unknown target task_id: {tr.target!r}"
            raise ValueError(msg)
        graph.add_edge(tr.source, tr.target, probability=tr.probability)

    return graph


def validate_workflow(
    *,
    graph: nx.DiGraph,
    start_nodes: list[str],
) -> list[str]:
    """Validate a workflow graph for simulation readiness.

    Checks:
    - At least one start node is provided.
    - Every start node exists in the graph.
    - Outgoing probabilities sum to 1.0 for every non-terminal node.
    - At least one terminal (sink) node is reachable from each start node.
    - All nodes are reachable from the union of start nodes.

    Args:
        graph: The directed workflow graph.
        start_nodes: Nodes where simulation begins (run in parallel).

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if not start_nodes:
        errors.append("at least one start node is required")
        return errors

    for sn in start_nodes:
        if sn not in graph:
            errors.append(f"start node {sn!r} not in graph")

    if errors:
        return errors

    for node in graph.nodes():
        successors = list(graph.successors(node))
        if not successors:
            continue
        total_prob = sum(graph.edges[node, succ]["probability"] for succ in successors)
        if not np.isclose(total_prob, 1.0):
            errors.append(
                f"outgoing probabilities for {node!r} sum to "
                f"{total_prob:.4f}, expected 1.0"
            )

    reachable: set[str] = set()
    for sn in start_nodes:
        reachable |= nx.descendants(graph, sn) | {sn}

    for sn in start_nodes:
        sn_reachable = nx.descendants(graph, sn) | {sn}
        terminals = [n for n in sn_reachable if graph.out_degree(n) == 0]
        if not terminals:
            errors.append(f"no terminal node reachable from start node {sn!r}")

    unreachable = set(graph.nodes()) - reachable
    if unreachable:
        errors.append(f"unreachable nodes from start: {', '.join(sorted(unreachable))}")

    return errors


def _run_walk(
    *,
    start: str,
    task_map: dict[str, Task],
    adjacency: dict[str, tuple[list[str], NDArray[np.floating]]],
    rng: np.random.Generator,
    pert_lambda: float,
    max_steps: int,
) -> tuple[float, int, dict[str, int], dict[str, float], bool]:
    """Execute a single random walk from *start* to a terminal node."""
    visit_counts: dict[str, int] = {}
    visit_durs: dict[str, float] = {}
    current = start
    total_duration = 0.0
    steps = 0

    while steps < max_steps:
        task = task_map[current]
        duration = float(
            sample_pert(
                estimate=task.estimate,
                size=1,
                rng=rng,
                pert_lambda=pert_lambda,
            )[0]
        )
        total_duration += duration
        steps += 1
        visit_counts[current] = visit_counts.get(current, 0) + 1
        visit_durs[current] = visit_durs.get(current, 0.0) + duration

        if current not in adjacency:
            break

        successors, cum_probs = adjacency[current]
        r = rng.random()
        idx = int(np.searchsorted(cum_probs, r))
        idx = min(idx, len(successors) - 1)
        current = successors[idx]
    else:
        return total_duration, steps, visit_counts, visit_durs, True

    return total_duration, steps, visit_counts, visit_durs, False


def simulate_task_workflow(
    *,
    tasks: list[Task],
    transitions: list[TaskTransition],
    start_nodes: list[str],
    config: SimulationConfig,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> WorkflowResult:
    """Run Monte Carlo simulation on a cyclic probabilistic workflow.

    Each iteration runs a random walk from every start node in parallel.
    The iteration duration is the **maximum** across all walks (parallel
    execution). Step counts and visit statistics are summed.

    Args:
        tasks: List of workflow tasks.
        transitions: Probabilistic edges between tasks.
        start_nodes: Nodes where each iteration begins (run in parallel).
        config: Simulation configuration.
        max_steps: Safety limit on task executions per walk.

    Returns:
        WorkflowResult with duration statistics and per-task visit data.

    Raises:
        ValueError: If any start node is not a known task.
    """
    task_map = {t.task_id: t for t in tasks}
    for sn in start_nodes:
        if sn not in task_map:
            msg = f"start node {sn!r} is not a known task"
            raise ValueError(msg)

    graph = build_workflow_graph(tasks=tasks, transitions=transitions)

    # Pre-compute adjacency with cumulative probabilities.
    adjacency: dict[str, tuple[list[str], NDArray[np.floating]]] = {}
    for node in graph.nodes():
        successors = list(graph.successors(node))
        if successors:
            probs = np.array([graph.edges[node, s]["probability"] for s in successors])
            cum_probs = np.cumsum(probs)
            adjacency[node] = (successors, cum_probs)

    rng = np.random.default_rng(config.seed)
    n_iter = config.iterations
    task_ids = [t.task_id for t in tasks]

    durations = np.zeros(n_iter)
    step_counts = np.zeros(n_iter, dtype=np.int64)
    visit_counts_arr = {tid: np.zeros(n_iter, dtype=np.int64) for tid in task_ids}
    visit_durations_arr = {tid: np.zeros(n_iter) for tid in task_ids}
    max_steps_hit = 0

    for i in range(n_iter):
        iter_max_dur = 0.0
        iter_steps = 0
        iter_hit = False

        for sn in start_nodes:
            dur, steps, vc, vd, hit = _run_walk(
                start=sn,
                task_map=task_map,
                adjacency=adjacency,
                rng=rng,
                pert_lambda=config.pert_lambda,
                max_steps=max_steps,
            )
            iter_max_dur = max(iter_max_dur, dur)
            iter_steps += steps
            iter_hit = iter_hit or hit
            for tid, cnt in vc.items():
                visit_counts_arr[tid][i] += cnt
            for tid, d in vd.items():
                visit_durations_arr[tid][i] += d

        if iter_hit:
            max_steps_hit += 1

        durations[i] = iter_max_dur
        step_counts[i] = iter_steps

    duration_result = compute_result(samples=durations)

    return WorkflowResult(
        duration_result=duration_result,
        step_counts=step_counts,
        visit_counts=visit_counts_arr,
        visit_durations=visit_durations_arr,
        max_steps_hit_count=max_steps_hit,
    )
