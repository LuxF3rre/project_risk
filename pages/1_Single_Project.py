"""UC1: Single three-point estimate simulation."""

import streamlit as st

from project_risk.charts import build_cdf, build_histogram
from project_risk.models import PertEstimate, SimulationConfig
from project_risk.monte_carlo import compute_result, simulate_single
from project_risk.ui import sidebar_config

st.set_page_config(page_title="Single Project", layout="wide")
st.title("Single Project Estimate")

st.markdown(
    """
Estimate the duration of a single project or activity. Provide three
values — the **best case**, **most likely**, and **worst case** duration —
and the simulation will show you the full range of possible outcomes.
"""
)

# --- Inputs ---
load_col, clear_col, *_ = st.columns([2, 2, 6], gap="small")
if load_col.button("Load Example"):
    st.session_state["sp_optimistic"] = 5.0
    st.session_state["sp_most_likely"] = 10.0
    st.session_state["sp_pessimistic"] = 20.0
    st.rerun()
if clear_col.button("Clear Data"):
    st.session_state["sp_optimistic"] = 0.0
    st.session_state["sp_most_likely"] = 0.0
    st.session_state["sp_pessimistic"] = 0.0
    st.session_state.pop("sp_result", None)
    st.session_state.pop("sp_run_inputs", None)
    st.rerun()

col1, col2, col3 = st.columns(3)
with col1:
    optimistic = st.number_input(
        "Optimistic", min_value=0.0, value=0.0, step=0.5, key="sp_optimistic"
    )
with col2:
    most_likely = st.number_input(
        "Most Likely", min_value=0.0, value=0.0, step=0.5, key="sp_most_likely"
    )
with col3:
    pessimistic = st.number_input(
        "Pessimistic", min_value=0.0, value=0.0, step=0.5, key="sp_pessimistic"
    )

# --- Sidebar config ---
iterations, seed, pert_lambda = sidebar_config()

# --- Run ---
if st.button("Run Simulation", type="primary"):
    try:
        estimate = PertEstimate(
            optimistic=optimistic,
            most_likely=most_likely,
            pessimistic=pessimistic,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    config = SimulationConfig(iterations=iterations, seed=seed, pert_lambda=pert_lambda)
    result = simulate_single(estimate=estimate, config=config)
    st.session_state["sp_result"] = result
    st.session_state["sp_run_inputs"] = (
        optimistic,
        most_likely,
        pessimistic,
        iterations,
        seed,
        pert_lambda,
    )

# --- Display results from session state ---
if "sp_result" in st.session_state:
    result = st.session_state["sp_result"]

    # --- Stale results warning ---
    current_inputs = (
        optimistic,
        most_likely,
        pessimistic,
        iterations,
        seed,
        pert_lambda,
    )
    if st.session_state.get("sp_run_inputs") != current_inputs:
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
        f"\nOn average, the project is expected to take about **{result.mean:.1f}**."
    )
    st.markdown("\n".join(lines))

    # --- Visualizations ---
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

    # --- Summary stats ---
    st.subheader("Summary")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Mean", f"{result.mean:.2f}")
    mcol2.metric("Std Dev", f"{result.std_dev:.2f}")
