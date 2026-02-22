"""Visualization helpers: Altair charts and Graphviz DOT generation."""

from __future__ import annotations

import altair as alt
import pandas as pd

from project_risk.models import SimulationResult, TaskTransition

__all__ = [
    "build_cdf",
    "build_dag_dot",
    "build_histogram",
    "build_workflow_dot",
]


def build_histogram(
    *,
    result: SimulationResult,
    bin_count: int = 50,
    title: str = "Duration Distribution",
) -> alt.LayerChart:
    """Build a histogram with a vertical mean line.

    Args:
        result: Simulation result containing samples.
        bin_count: Number of histogram bins.
        title: Chart title.

    Returns:
        An Altair LayerChart.
    """
    df = pd.DataFrame({"duration": result.samples})

    hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.7, color="#4c78a8")
        .encode(
            x=alt.X("duration:Q", bin=alt.Bin(maxbins=bin_count), title="Duration"),
            y=alt.Y("count()", title="Frequency"),
        )
    )

    mean_line = (
        alt.Chart(pd.DataFrame({"mean": [result.mean]}))
        .mark_rule(color="red", strokeWidth=2)
        .encode(x="mean:Q")
    )

    return (hist + mean_line).properties(title=title, width=600, height=400)


def build_cdf(
    *,
    result: SimulationResult,
    title: str = "Cumulative Distribution",
) -> alt.Chart:
    """Build an empirical CDF chart.

    Args:
        result: Simulation result containing samples.
        title: Chart title.

    Returns:
        An Altair Chart.
    """
    df = pd.DataFrame({"duration": result.samples})

    return (
        alt.Chart(df)
        .transform_window(
            cumulative_count="count()",
            sort=[{"field": "duration"}],
        )
        .transform_joinaggregate(total="count()")
        .transform_calculate(cdf="datum.cumulative_count / datum.total")
        .mark_line(color="#4c78a8")
        .encode(
            x=alt.X("duration:Q", title="Duration"),
            y=alt.Y("cdf:Q", title="Cumulative Probability"),
        )
        .properties(title=title, width=600, height=400)
    )


def build_dag_dot(
    *,
    task_ids: list[str],
    dependencies: list[tuple[str, str]],
    critical_path: list[str] | None = None,
) -> str:
    """Generate a Graphviz DOT string for DAG visualization.

    Args:
        task_ids: List of task identifiers.
        dependencies: List of (predecessor, successor) tuples.
        critical_path: Optional list of task_ids on the critical path.

    Returns:
        A DOT-language string.
    """
    critical_set = set(critical_path) if critical_path else set()
    lines = ["digraph {", "    rankdir=LR;", "    node [shape=box, style=filled];"]

    for tid in task_ids:
        if tid in critical_set:
            lines.append(f'    "{tid}" [fillcolor="#ff6b6b", fontcolor="white"];')
        else:
            lines.append(f'    "{tid}" [fillcolor="#e8e8e8"];')

    critical_edges = set()
    if critical_path and len(critical_path) > 1:
        for i in range(len(critical_path) - 1):
            critical_edges.add((critical_path[i], critical_path[i + 1]))

    for pred, succ in dependencies:
        if (pred, succ) in critical_edges:
            lines.append(f'    "{pred}" -> "{succ}" [color="red", penwidth=2.0];')
        else:
            lines.append(f'    "{pred}" -> "{succ}";')

    lines.append("}")
    return "\n".join(lines)


def build_workflow_dot(
    *,
    task_ids: list[str],
    transitions: list[TaskTransition],
    start_nodes: list[str],
) -> str:
    """Generate a Graphviz DOT string for a workflow graph.

    Args:
        task_ids: List of task identifiers.
        transitions: Probabilistic transition edges.
        start_nodes: Starting nodes (colored blue).

    Returns:
        A DOT-language string with probability labels on edges.
    """
    start_set = set(start_nodes)
    targets = {tr.target for tr in transitions}
    sources = {tr.source for tr in transitions}
    terminal_nodes = {tid for tid in task_ids if tid in targets and tid not in sources}
    terminal_nodes |= {
        tid for tid in task_ids if tid not in sources and tid not in targets
    }
    terminal_nodes -= start_set

    lines = [
        "digraph {",
        "    rankdir=LR;",
        "    node [shape=box, style=filled];",
    ]

    for tid in task_ids:
        if tid in start_set:
            lines.append(f'    "{tid}" [fillcolor="#4c78a8", fontcolor="white"];')
        elif tid in terminal_nodes:
            lines.append(f'    "{tid}" [fillcolor="#59a14f", fontcolor="white"];')
        else:
            lines.append(f'    "{tid}" [fillcolor="#e8e8e8"];')

    for tr in transitions:
        label = f"{tr.probability:.0%}"
        lines.append(f'    "{tr.source}" -> "{tr.target}" [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)
