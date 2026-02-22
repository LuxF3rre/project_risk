"""Smoke tests for project_risk.charts."""

import altair as alt
import numpy as np

from project_risk.charts import (
    build_cdf,
    build_dag_dot,
    build_histogram,
    build_workflow_dot,
)
from project_risk.models import PercentileResult, SimulationResult, TaskTransition


def _make_result() -> SimulationResult:
    rng = np.random.default_rng(0)
    samples = rng.normal(50, 10, size=1000)
    return SimulationResult(
        samples=samples,
        percentiles=(PercentileResult(percentile=50, value=50.0),),
        mean=float(np.mean(samples)),
        std_dev=float(np.std(samples)),
    )


class TestBuildHistogram:
    def test_returns_layer_chart(self) -> None:
        result = _make_result()
        chart = build_histogram(result=result)
        assert isinstance(chart, alt.LayerChart)

    def test_custom_title(self) -> None:
        result = _make_result()
        chart = build_histogram(result=result, title="Custom Title")
        assert chart.title == "Custom Title"


class TestBuildCdf:
    def test_returns_chart(self) -> None:
        result = _make_result()
        chart = build_cdf(result=result)
        assert isinstance(chart, alt.Chart)

    def test_custom_title(self) -> None:
        result = _make_result()
        chart = build_cdf(result=result, title="CDF Title")
        assert chart.title == "CDF Title"


class TestBuildDagDot:
    def test_basic_dot(self) -> None:
        dot = build_dag_dot(
            task_ids=["A", "B", "C"],
            dependencies=[("A", "B"), ("B", "C")],
        )
        assert "digraph" in dot
        assert '"A"' in dot
        assert '"A" -> "B"' in dot

    def test_critical_path_highlight(self) -> None:
        dot = build_dag_dot(
            task_ids=["A", "B", "C"],
            dependencies=[("A", "B"), ("B", "C")],
            critical_path=["A", "B"],
        )
        assert "ff6b6b" in dot  # Critical node color
        assert 'color="red"' in dot  # Critical edge

    def test_no_critical_path(self) -> None:
        dot = build_dag_dot(
            task_ids=["X"],
            dependencies=[],
        )
        assert '"X"' in dot
        assert "red" not in dot


class TestBuildWorkflowDot:
    def test_basic_dot(self) -> None:
        transitions = [
            TaskTransition(source="A", target="B", probability=0.6),
            TaskTransition(source="A", target="C", probability=0.4),
        ]
        dot = build_workflow_dot(
            task_ids=["A", "B", "C"],
            transitions=transitions,
            start_nodes=["A"],
        )
        assert "digraph" in dot
        assert '"A"' in dot
        assert '"A" -> "B"' in dot

    def test_start_node_colored_blue(self) -> None:
        transitions = [
            TaskTransition(source="A", target="B", probability=1.0),
        ]
        dot = build_workflow_dot(
            task_ids=["A", "B"],
            transitions=transitions,
            start_nodes=["A"],
        )
        assert "4c78a8" in dot

    def test_terminal_node_colored_green(self) -> None:
        transitions = [
            TaskTransition(source="A", target="B", probability=1.0),
        ]
        dot = build_workflow_dot(
            task_ids=["A", "B"],
            transitions=transitions,
            start_nodes=["A"],
        )
        assert "59a14f" in dot

    def test_probability_labels(self) -> None:
        transitions = [
            TaskTransition(source="A", target="B", probability=0.7),
            TaskTransition(source="A", target="C", probability=0.3),
        ]
        dot = build_workflow_dot(
            task_ids=["A", "B", "C"],
            transitions=transitions,
            start_nodes=["A"],
        )
        assert "70%" in dot
        assert "30%" in dot

    def test_regular_node_gray(self) -> None:
        transitions = [
            TaskTransition(source="A", target="B", probability=1.0),
            TaskTransition(source="B", target="C", probability=1.0),
        ]
        dot = build_workflow_dot(
            task_ids=["A", "B", "C"],
            transitions=transitions,
            start_nodes=["A"],
        )
        assert "e8e8e8" in dot
