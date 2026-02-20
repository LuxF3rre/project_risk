"""UC2: Sequential task list simulation."""

import pandas as pd
import streamlit as st

from project_risk.charts import build_cdf, build_dag_dot, build_histogram
from project_risk.models import PertEstimate, SimulationConfig, Task
from project_risk.monte_carlo import compute_result, simulate_task_list

st.set_page_config(page_title="Task List", layout="wide")
st.title("Task List (Sequential)")

st.markdown(
    """
Break your project into individual tasks that run **one after another**.
Give each task its own best-case, most-likely, and worst-case duration.
The simulation adds up all task durations in each scenario, showing you
how individual uncertainty compounds into overall project risk.
"""
)

# --- Task editor ---
st.subheader("Tasks")

EXAMPLE_DATA = pd.DataFrame(
    {
        "Task": ["Design", "Development", "Testing"],
        "Optimistic": [2.0, 5.0, 1.0],
        "Most Likely": [4.0, 10.0, 3.0],
        "Pessimistic": [8.0, 20.0, 6.0],
    }
)

EMPTY_DATA = pd.DataFrame(
    {
        "Task": pd.Series(dtype="str"),
        "Optimistic": pd.Series(dtype="float"),
        "Most Likely": pd.Series(dtype="float"),
        "Pessimistic": pd.Series(dtype="float"),
    }
)

if "tl_data" not in st.session_state:
    st.session_state["tl_data"] = EMPTY_DATA

load_col, clear_col, *_ = st.columns([2, 2, 6], gap="small")
if load_col.button("Load Example"):
    st.session_state["tl_data"] = EXAMPLE_DATA.copy()
    st.session_state.pop("tl_result", None)
    st.rerun()
if clear_col.button("Clear Data"):
    st.session_state["tl_data"] = EMPTY_DATA.copy()
    st.session_state.pop("tl_result", None)
    st.rerun()

edited_df = st.data_editor(
    st.session_state["tl_data"],
    num_rows="dynamic",
    width="stretch",
    column_config={
        "Task": st.column_config.TextColumn("Task Name", required=True),
        "Optimistic": st.column_config.NumberColumn("Optimistic", min_value=0.0),
        "Most Likely": st.column_config.NumberColumn("Most Likely", min_value=0.0),
        "Pessimistic": st.column_config.NumberColumn("Pessimistic", min_value=0.0),
    },
)

# --- Sidebar config ---
st.sidebar.header("Simulation Settings")
st.sidebar.markdown(
    """
**Iterations** — How many simulated scenarios to run. More iterations
give smoother, more reliable results.

**Random Seed** — A number that makes results reproducible. Using the
same seed always produces the same output. Set to 0 for a different
result each time.

**PERT Lambda** — Controls how strongly the simulation favors the
"most likely" value. Higher values produce a tighter curve around the
most likely estimate; lower values spread the results wider.
"""
)
iterations = st.sidebar.number_input(
    "Iterations", min_value=100, max_value=10_000, value=10_000, step=1000
)
seed_input = st.sidebar.number_input(
    "Random Seed (0 = none)", min_value=0, max_value=2**31 - 1, value=0
)
pert_lambda = st.sidebar.slider("PERT Lambda", min_value=1.0, max_value=10.0, value=4.0)

seed = int(seed_input) if seed_input > 0 else None

# --- Run ---
if st.button("Run Simulation", type="primary"):
    tasks: list[Task] = []
    for _, row in edited_df.iterrows():
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

    config = SimulationConfig(
        iterations=int(iterations), seed=seed, pert_lambda=pert_lambda
    )
    result = simulate_task_list(tasks=tasks, config=config)
    st.session_state["tl_result"] = result
    st.session_state["tl_tasks"] = tasks
    st.session_state["tl_lambda"] = pert_lambda

# --- Display results from session state ---
if "tl_result" in st.session_state:
    result = st.session_state["tl_result"]
    tasks = st.session_state["tl_tasks"]
    lam = st.session_state["tl_lambda"]

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
        f"\nOn average, the total project duration is expected to be "
        f"about **{result.mean:.1f}**."
    )
    st.markdown("\n".join(lines))

    # --- Linear DAG visualization ---
    st.subheader("Task Flow")
    task_ids = [t.task_id for t in tasks]
    deps = [(task_ids[i], task_ids[i + 1]) for i in range(len(task_ids) - 1)]
    dot = build_dag_dot(
        task_ids=task_ids,
        dependencies=deps,
        critical_path=task_ids,
    )
    st.graphviz_chart(dot)

    # --- Charts ---
    tab_hist, tab_cdf = st.tabs(["Histogram", "CDF"])
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

    # --- Custom percentile ---
    st.subheader("Custom Percentile")
    custom_pct = st.slider("Percentile", min_value=1, max_value=99, value=80)
    custom_result = compute_result(
        samples=result.samples, percentile_values=(float(custom_pct),)
    )
    st.metric(
        f"P{custom_pct}",
        f"{custom_result.percentiles[0].value:.2f}",
    )

    # --- Per-task stats ---
    st.subheader("Per-Task Estimates")
    task_stats = {
        "Task": [t.task_id for t in tasks],
        "Optimistic": [t.estimate.optimistic for t in tasks],
        "Most Likely": [t.estimate.most_likely for t in tasks],
        "Pessimistic": [t.estimate.pessimistic for t in tasks],
        "PERT Mean": [
            (
                t.estimate.optimistic
                + lam * t.estimate.most_likely
                + t.estimate.pessimistic
            )
            / (lam + 2)
            for t in tasks
        ],
    }
    st.table(pd.DataFrame(task_stats))

    # --- Summary ---
    st.subheader("Summary")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Mean Total Duration", f"{result.mean:.2f}")
    mcol2.metric("Std Dev", f"{result.std_dev:.2f}")
