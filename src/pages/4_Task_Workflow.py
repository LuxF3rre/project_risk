"""UC4: Cyclic probabilistic graph simulation."""

import numpy as np
import pandas as pd
import streamlit as st

from project_risk.charts import build_cdf, build_histogram, build_workflow_dot
from project_risk.models import (
    DEFAULT_MAX_STEPS,
    PertEstimate,
    SimulationConfig,
    Task,
    TaskTransition,
)
from project_risk.monte_carlo import compute_result
from project_risk.ui import sidebar_config
from project_risk.workflow import (
    build_workflow_graph,
    simulate_task_workflow,
    validate_workflow,
)

st.set_page_config(page_title="Task Workflow", layout="wide")
st.title("Task Workflow (Cyclic Graph)")

st.markdown(
    """
Model your project as a **probabilistic workflow** — tasks with weighted
transitions that can form **cycles** (e.g., code review loops back to
development, testing fails and returns to bug fixes). The simulation runs
a random walk through the graph for each iteration until a terminal node
is reached.
"""
)

# --- Example data ---
EXAMPLE_TASKS = pd.DataFrame(
    {
        "Task": [
            "Requirements",
            "Development",
            "Code Review",
            "Testing",
            "Bug Fixes",
            "User Acceptance",
            "Deployment",
        ],
        "Optimistic": [2.0, 5.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        "Most Likely": [4.0, 10.0, 2.0, 5.0, 3.0, 3.0, 2.0],
        "Pessimistic": [8.0, 20.0, 4.0, 10.0, 7.0, 5.0, 4.0],
    }
)

EXAMPLE_TRANSITIONS = pd.DataFrame(
    {
        "Source": [
            "Requirements",
            "Development",
            "Code Review",
            "Code Review",
            "Testing",
            "Testing",
            "Bug Fixes",
            "User Acceptance",
            "User Acceptance",
        ],
        "Target": [
            "Development",
            "Code Review",
            "Testing",
            "Development",
            "User Acceptance",
            "Bug Fixes",
            "Testing",
            "Deployment",
            "Bug Fixes",
        ],
        "Probability": [1.0, 1.0, 0.7, 0.3, 0.6, 0.4, 1.0, 0.8, 0.2],
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

EMPTY_TRANSITIONS = pd.DataFrame(
    {
        "Source": pd.Series(dtype="str"),
        "Target": pd.Series(dtype="str"),
        "Probability": pd.Series(dtype="float"),
    }
)

# --- Session state ---
if "wf_data" not in st.session_state:
    st.session_state["wf_data"] = EMPTY_TASKS

if "wf_trans_data" not in st.session_state:
    st.session_state["wf_trans_data"] = EMPTY_TRANSITIONS

# --- Load / Clear ---
load_col, clear_col, *_ = st.columns([3, 2, 5], gap="small")
if load_col.button("Load Example", use_container_width=True):
    st.session_state["wf_data"] = EXAMPLE_TASKS.copy()
    st.session_state["wf_trans_data"] = EXAMPLE_TRANSITIONS.copy()
    st.session_state.pop("wf_result", None)
    st.session_state.pop("wf_run_inputs", None)
    st.rerun()
if clear_col.button("Clear Data", use_container_width=True):
    st.session_state["wf_data"] = EMPTY_TASKS.copy()
    st.session_state["wf_trans_data"] = EMPTY_TRANSITIONS.copy()
    st.session_state.pop("wf_result", None)
    st.session_state.pop("wf_run_inputs", None)
    st.rerun()

# --- Task editor ---
st.subheader("Tasks")

edited_tasks = st.data_editor(
    st.session_state["wf_data"],
    num_rows="dynamic",
    width="stretch",
    key="wf_task_editor",
    column_config={
        "Task": st.column_config.TextColumn("Task Name", required=True),
        "Optimistic": st.column_config.NumberColumn("Optimistic", min_value=0.0),
        "Most Likely": st.column_config.NumberColumn("Most Likely", min_value=0.0),
        "Pessimistic": st.column_config.NumberColumn("Pessimistic", min_value=0.0),
    },
)

task_names = [
    str(row["Task"]).strip()
    for _, row in edited_tasks.iterrows()
    if str(row["Task"]).strip()
]

# --- Transition editor ---
st.subheader("Transitions")
st.caption(
    "Each row defines a probabilistic transition from Source to Target. "
    "Outgoing probabilities from each non-terminal node must sum to 1.0."
)

edited_trans = st.data_editor(
    st.session_state["wf_trans_data"],
    num_rows="dynamic",
    width="stretch",
    key="wf_trans_editor",
    column_config={
        "Source": st.column_config.SelectboxColumn(
            "Source",
            options=task_names,
            required=True,
        ),
        "Target": st.column_config.SelectboxColumn(
            "Target",
            options=task_names,
            required=True,
        ),
        "Probability": st.column_config.NumberColumn(
            "Probability",
            min_value=0.01,
            max_value=1.0,
            step=0.05,
            format="%.2f",
        ),
    },
)

# Build transition list
trans_list: list[TaskTransition] = []
for _, row in edited_trans.iterrows():
    src = str(row.get("Source", "")).strip()
    tgt = str(row.get("Target", "")).strip()
    prob = row.get("Probability")
    if (
        src
        and tgt
        and src not in ("", "nan")
        and tgt not in ("", "nan")
        and prob is not None
        and not np.isnan(prob)
    ):
        trans_list.append(
            TaskTransition(source=src, target=tgt, probability=float(prob))
        )

# --- Live probability-sum check ---
if trans_list:
    prob_sums: dict[str, float] = {}
    for tr in trans_list:
        prob_sums[tr.source] = prob_sums.get(tr.source, 0.0) + tr.probability
    bad_sources = {
        src: total for src, total in prob_sums.items() if not np.isclose(total, 1.0)
    }
    if bad_sources:
        for src, total in sorted(bad_sources.items()):
            st.warning(
                f"Outgoing probabilities for **{src}** sum to "
                f"**{total:.2f}** (expected 1.00)."
            )

# --- Start node detection (auto-inferred) ---
# Start nodes are tasks with no incoming transitions.
_targets_in_trans = {tr.target for tr in trans_list}
_sources_in_trans = {tr.source for tr in trans_list}
start_nodes = [t for t in task_names if t not in _targets_in_trans]

# Warn about isolated nodes (no transitions at all).
_isolated = [
    t for t in start_nodes if t not in _sources_in_trans and t not in _targets_in_trans
]
if _isolated and len(task_names) > 1:
    st.warning(
        f"Isolated node(s) with no transitions: **{', '.join(_isolated)}**. "
        f"They will be simulated as single-step start nodes."
    )

# --- Live graph preview ---
if task_names and trans_list:
    st.subheader("Workflow Preview")
    preview_dot = build_workflow_dot(
        task_ids=task_names,
        transitions=trans_list,
        start_nodes=start_nodes,
    )
    st.graphviz_chart(preview_dot)

# --- Sidebar config ---
iterations, seed, pert_lambda = sidebar_config()

max_steps = int(
    st.sidebar.number_input(
        "Max Steps per Iteration",
        min_value=10,
        max_value=100_000,
        value=DEFAULT_MAX_STEPS,
        step=100,
        help="Safety limit: stop a single iteration after this many task executions.",
    )
)

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

    if not start_nodes:
        st.error("Select at least one start node.")
        st.stop()

    # Validate
    task_id_set = {t.task_id for t in tasks}
    valid_trans = [
        t for t in trans_list if t.source in task_id_set and t.target in task_id_set
    ]

    graph = build_workflow_graph(tasks=tasks, transitions=valid_trans)
    errors = validate_workflow(graph=graph, start_nodes=start_nodes)
    if errors:
        for err in errors:
            st.error(f"Validation error: {err}")
        st.stop()

    config = SimulationConfig(iterations=iterations, seed=seed, pert_lambda=pert_lambda)

    try:
        result = simulate_task_workflow(
            tasks=tasks,
            transitions=valid_trans,
            start_nodes=start_nodes,
            config=config,
            max_steps=max_steps,
        )
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.stop()

    st.session_state["wf_result"] = result
    st.session_state["wf_tasks"] = tasks
    st.session_state["wf_transitions"] = valid_trans
    st.session_state["wf_start_nodes"] = start_nodes
    st.session_state["wf_lambda"] = pert_lambda
    st.session_state["wf_max_steps"] = max_steps
    st.session_state["wf_run_inputs"] = (
        edited_tasks.to_csv(index=False),
        edited_trans.to_csv(index=False),
        tuple(start_nodes),
        iterations,
        seed,
        pert_lambda,
        max_steps,
    )

# --- Display results ---
if "wf_result" in st.session_state:
    result = st.session_state["wf_result"]
    tasks = st.session_state["wf_tasks"]
    saved_trans = st.session_state["wf_transitions"]
    saved_starts = st.session_state["wf_start_nodes"]
    saved_max_steps = st.session_state["wf_max_steps"]
    dur = result.duration_result

    # --- Stale results warning ---
    current_inputs = (
        edited_tasks.to_csv(index=False),
        edited_trans.to_csv(index=False),
        tuple(start_nodes),
        iterations,
        seed,
        pert_lambda,
        max_steps,
    )
    if st.session_state.get("wf_run_inputs") != current_inputs:
        st.warning(
            "Inputs have changed since last run. Click **Run Simulation** to update."
        )

    # --- Max-steps warning ---
    if result.max_steps_hit_count > 0:
        pct = result.max_steps_hit_count / len(dur.samples) * 100
        st.warning(
            f"{result.max_steps_hit_count} iteration(s) ({pct:.1f}%) hit the "
            f"max-steps limit ({saved_max_steps}). Results may underestimate "
            f"true durations. Consider increasing the limit or reviewing the "
            f"workflow for degenerate cycles."
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
        samples=dur.samples, percentile_values=(float(confidence),)
    )
    st.metric(
        f"P{confidence} Duration",
        f"{deadline.percentiles[0].value:.2f}",
        help=f"{confidence}% chance the workflow finishes within this duration.",
    )

    # --- Interpretation ---
    st.subheader("What does this mean?")
    lines = []
    for p in dur.percentiles:
        pct = int(p.percentile)
        lines.append(
            f"- There is a **{pct}% chance** the workflow will finish "
            f"in **{p.value:.1f} or less**."
        )
    lines.append(
        f"\nOn average, the workflow duration is expected to be about "
        f"**{dur.mean:.1f}**."
    )
    st.markdown("\n".join(lines))

    # --- Workflow graph ---
    st.subheader("Workflow Graph")
    result_dot = build_workflow_dot(
        task_ids=[t.task_id for t in tasks],
        transitions=saved_trans,
        start_nodes=saved_starts,
    )
    st.graphviz_chart(result_dot)

    # --- Duration charts ---
    tab_hist, tab_cdf = st.tabs(["Histogram", "Cumulative Probability"])
    with tab_hist:
        st.altair_chart(build_histogram(result=dur), width="stretch")
    with tab_cdf:
        st.altair_chart(build_cdf(result=dur), width="stretch")

    # --- Percentile table ---
    st.subheader("Percentiles")
    pct_data = {
        "Percentile": [f"P{int(p.percentile)}" for p in dur.percentiles],
        "Duration": [round(p.value, 2) for p in dur.percentiles],
    }
    st.table(pct_data)

    # --- Per-task visit stats ---
    st.subheader("Per-Task Visit Statistics")
    visit_rows = []
    for t in tasks:
        tid = t.task_id
        counts = result.visit_counts[tid]
        durations = result.visit_durations[tid]
        n_iter = len(counts)
        visited_mask = counts > 0
        visit_rate = float(np.sum(visited_mask)) / n_iter * 100
        visit_rows.append(
            {
                "Task": tid,
                "Avg Visits": round(float(np.mean(counts)), 2),
                "Max Visits": int(np.max(counts)),
                "Avg Time Spent": round(float(np.mean(durations)), 2),
                "Visit Rate (%)": round(visit_rate, 1),
            }
        )
    st.table(pd.DataFrame(visit_rows))

    # --- Path length distribution ---
    st.subheader("Path Length Distribution")
    step_result = compute_result(samples=result.step_counts.astype(np.float64))
    st.altair_chart(
        build_histogram(
            result=step_result,
            title="Steps per Iteration",
        ),
        width="stretch",
    )
    st.caption(
        f"Average path length: **{step_result.mean:.1f}** steps "
        f"(std dev: {step_result.std_dev:.1f})"
    )

    # --- Summary ---
    st.subheader("Summary")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Mean Duration", f"{dur.mean:.2f}")
    mcol2.metric("Std Dev", f"{dur.std_dev:.2f}")
    mcol3.metric("Avg Steps", f"{step_result.mean:.1f}")

st.divider()
st.markdown(
    "<div style='text-align: center'>Made with ❤ by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
