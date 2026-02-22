"""UC3: Task dependency graph simulation."""

import pandas as pd
import streamlit as st

from project_risk.charts import build_cdf, build_dag_dot, build_histogram
from project_risk.dag import find_critical_path
from project_risk.models import (
    PertEstimate,
    SimulationConfig,
    Task,
    TaskDependency,
)
from project_risk.monte_carlo import compute_result, simulate_task_dag
from project_risk.ui import sidebar_config

st.set_page_config(page_title="Task DAG", layout="wide")
st.title("Task Dependency Graph")

st.markdown(
    """
Model your project as a network of tasks with **dependencies** —
some tasks can't start until others finish, and some can run in parallel.
The simulation finds the **critical path** (the longest chain of dependent
tasks) in each scenario and shows you how it drives the overall duration.
"""
)

# --- Task editor ---
EXAMPLE_TASKS = pd.DataFrame(
    {
        "Task": ["Design", "Backend", "Frontend", "Integration", "Testing"],
        "Optimistic": [2.0, 5.0, 4.0, 1.0, 2.0],
        "Most Likely": [4.0, 10.0, 8.0, 3.0, 4.0],
        "Pessimistic": [8.0, 18.0, 14.0, 5.0, 8.0],
    }
)

EXAMPLE_DEPS = pd.DataFrame(
    {
        "Predecessor": ["Design", "Design", "Backend", "Frontend", "Integration"],
        "Successor": ["Backend", "Frontend", "Integration", "Integration", "Testing"],
    }
)

EMPTY_TASKS = pd.DataFrame(
    {
        "Task": pd.Series(dtype="str"),
        "Optimistic": pd.Series(dtype="float"),
        "Most Likely": pd.Series(dtype="float"),
        "Pessimistic": pd.Series(dtype="float"),
    }
)

EMPTY_DEPS = pd.DataFrame(
    {
        "Predecessor": pd.Series(dtype="str"),
        "Successor": pd.Series(dtype="str"),
    }
)

if "dag_data" not in st.session_state:
    st.session_state["dag_data"] = EMPTY_TASKS

if "dag_deps_data" not in st.session_state:
    st.session_state["dag_deps_data"] = EMPTY_DEPS

load_col, clear_col, *_ = st.columns([3, 2, 5], gap="small")
if load_col.button("Load Example", use_container_width=True):
    st.session_state["dag_data"] = EXAMPLE_TASKS.copy()
    st.session_state["dag_deps_data"] = EXAMPLE_DEPS.copy()
    st.session_state.pop("dag_result", None)
    st.session_state.pop("dag_run_inputs", None)
    st.rerun()
if clear_col.button("Clear Data", use_container_width=True):
    st.session_state["dag_data"] = EMPTY_TASKS.copy()
    st.session_state["dag_deps_data"] = EMPTY_DEPS.copy()
    st.session_state.pop("dag_result", None)
    st.session_state.pop("dag_run_inputs", None)
    st.rerun()

st.subheader("Tasks")

edited_tasks = st.data_editor(
    st.session_state["dag_data"],
    num_rows="dynamic",
    width="stretch",
    key="task_editor",
    column_config={
        "Task": st.column_config.TextColumn("Task Name", required=True),
        "Optimistic": st.column_config.NumberColumn("Optimistic", min_value=0.0),
        "Most Likely": st.column_config.NumberColumn("Most Likely", min_value=0.0),
        "Pessimistic": st.column_config.NumberColumn("Pessimistic", min_value=0.0),
    },
)

# Build task name list for dependency editor
task_names = [
    str(row["Task"]).strip()
    for _, row in edited_tasks.iterrows()
    if str(row["Task"]).strip()
]

# --- Dependency editor (adjacency list) ---
st.subheader("Dependencies")
st.caption(
    "Each row defines one dependency: the successor cannot start "
    "until the predecessor finishes."
)

edited_deps = st.data_editor(
    st.session_state["dag_deps_data"],
    num_rows="dynamic",
    width="stretch",
    key="dep_editor",
    column_config={
        "Predecessor": st.column_config.SelectboxColumn(
            "Predecessor (must finish first)",
            options=task_names,
            required=True,
        ),
        "Successor": st.column_config.SelectboxColumn(
            "Successor (starts after)",
            options=task_names,
            required=True,
        ),
    },
)

# Build dependency tuples from the edge list
dep_tuples: list[tuple[str, str]] = []
for _, row in edited_deps.iterrows():
    pred = str(row.get("Predecessor", "")).strip()
    succ = str(row.get("Successor", "")).strip()
    if pred and succ and pred not in ("", "nan") and succ not in ("", "nan"):
        dep_tuples.append((pred, succ))

# --- Live DAG preview ---
if task_names:
    st.subheader("DAG Preview")
    preview_dot = build_dag_dot(
        task_ids=task_names,
        dependencies=dep_tuples,
    )
    st.graphviz_chart(preview_dot)

# --- Sidebar config ---
iterations, seed, pert_lambda = sidebar_config()

# --- Run ---
if st.button("Run Simulation", type="primary"):
    tasks: list[Task] = []
    for _, row in edited_tasks.iterrows():
        name = str(row["Task"]).strip()
        if not name:
            st.error("All tasks must have a name.")
            st.stop()
        try:
            est = PertEstimate(
                optimistic=float(row["Optimistic"]),
                most_likely=float(row["Most Likely"]),
                pessimistic=float(row["Pessimistic"]),
            )
        except ValueError as e:
            st.error(f"Task '{name}': {e}")
            st.stop()
        tasks.append(Task(task_id=name, estimate=est))

    if not tasks:
        st.warning("Add at least one task.")
        st.stop()

    dependencies: list[TaskDependency] = []
    task_id_set = {t.task_id for t in tasks}
    for pred, succ in dep_tuples:
        if pred in task_id_set and succ in task_id_set:
            dependencies.append(TaskDependency(predecessor=pred, successor=succ))

    config = SimulationConfig(iterations=iterations, seed=seed, pert_lambda=pert_lambda)

    try:
        result = simulate_task_dag(
            tasks=tasks, dependencies=dependencies, config=config
        )
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.stop()

    st.session_state["dag_result"] = result
    st.session_state["dag_tasks"] = tasks
    st.session_state["dag_dependencies"] = dependencies
    st.session_state["dag_dep_tuples"] = dep_tuples
    st.session_state["dag_lambda"] = pert_lambda
    st.session_state["dag_run_inputs"] = (
        edited_tasks.to_csv(index=False),
        edited_deps.to_csv(index=False),
        iterations,
        seed,
        pert_lambda,
    )

# --- Display results from session state ---
if "dag_result" in st.session_state:
    result = st.session_state["dag_result"]
    tasks = st.session_state["dag_tasks"]
    dependencies = st.session_state["dag_dependencies"]
    saved_dep_tuples = st.session_state["dag_dep_tuples"]
    lam = st.session_state["dag_lambda"]

    # --- Stale results warning ---
    current_inputs = (
        edited_tasks.to_csv(index=False),
        edited_deps.to_csv(index=False),
        iterations,
        seed,
        pert_lambda,
    )
    if st.session_state.get("dag_run_inputs") != current_inputs:
        st.warning(
            "Inputs have changed since last run. Click **Run Simulation** to update."
        )

    # --- Confidence deadline ---
    st.subheader("Confidence Deadline")
    confidence = st.slider(
        "How confident do you need to be?",
        min_value=1,
        max_value=99,
        value=80,
        format="%d%%",
    )
    deadline = compute_result(
        samples=result.samples, percentile_values=(float(confidence),)
    )
    st.metric(
        f"P{confidence} Duration",
        f"{deadline.percentiles[0].value:.2f}",
        help=f"{confidence}% chance the project finishes within this duration.",
    )

    # --- Plain-language interpretation ---
    st.subheader("What does this mean?")
    lines = []
    for p in result.percentiles:
        pct = int(p.percentile)
        lines.append(
            f"- There is a **{pct}% chance** the project will finish "
            f"in **{p.value:.1f} or less**."
        )
    lines.append(
        f"\nOn average, the project duration (driven by the critical "
        f"path) is expected to be about **{result.mean:.1f}**."
    )
    st.markdown("\n".join(lines))

    # --- Critical path ---
    task_ids = [t.task_id for t in tasks]
    pert_means = [
        (t.estimate.optimistic + lam * t.estimate.most_likely + t.estimate.pessimistic)
        / (lam + 2)
        for t in tasks
    ]
    try:
        critical_path = find_critical_path(
            task_ids=task_ids,
            dependencies=dependencies,
            durations=pert_means,
        )
    except Exception:
        critical_path = []

    # --- DAG with critical path ---
    st.subheader("DAG with Critical Path")
    result_dot = build_dag_dot(
        task_ids=task_ids,
        dependencies=saved_dep_tuples,
        critical_path=critical_path,
    )
    st.graphviz_chart(result_dot)

    if critical_path:
        st.info(f"Critical path: {' \u2192 '.join(critical_path)}")

    # --- Charts ---
    tab_hist, tab_cdf = st.tabs(["Histogram", "Cumulative Probability"])
    with tab_hist:
        st.altair_chart(build_histogram(result=result), width="stretch")
    with tab_cdf:
        st.altair_chart(build_cdf(result=result), width="stretch")

    # --- Percentile table ---
    st.subheader("Percentiles")
    pct_data = {
        "Percentile": [f"P{int(p.percentile)}" for p in result.percentiles],
        "Duration": [round(p.value, 2) for p in result.percentiles],
    }
    st.table(pct_data)

    # --- Per-task stats ---
    st.subheader("Per-Task Estimates")
    task_stats = {
        "Task": [t.task_id for t in tasks],
        "Optimistic": [t.estimate.optimistic for t in tasks],
        "Most Likely": [t.estimate.most_likely for t in tasks],
        "Pessimistic": [t.estimate.pessimistic for t in tasks],
        "PERT Mean": [round(m, 2) for m in pert_means],
        "On Critical Path": [
            "Yes" if t.task_id in critical_path else "" for t in tasks
        ],
    }
    st.table(pd.DataFrame(task_stats))

    # --- Summary ---
    st.subheader("Summary")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Mean Project Duration", f"{result.mean:.2f}")
    mcol2.metric("Std Dev", f"{result.std_dev:.2f}")
