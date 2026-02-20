"""UC1: Single three-point estimate simulation."""

import streamlit as st

from project_risk.charts import build_cdf, build_histogram
from project_risk.models import PertEstimate, SimulationConfig
from project_risk.monte_carlo import compute_result, simulate_single

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
if st.button("Load Example"):
    st.session_state["sp_optimistic"] = 5.0
    st.session_state["sp_most_likely"] = 10.0
    st.session_state["sp_pessimistic"] = 20.0
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
    try:
        estimate = PertEstimate(
            optimistic=optimistic,
            most_likely=most_likely,
            pessimistic=pessimistic,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    config = SimulationConfig(
        iterations=int(iterations), seed=seed, pert_lambda=pert_lambda
    )
    result = simulate_single(estimate=estimate, config=config)
    st.session_state["sp_result"] = result

# --- Display results from session state ---
if "sp_result" in st.session_state:
    result = st.session_state["sp_result"]

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

    # --- Summary stats ---
    st.subheader("Summary")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Mean", f"{result.mean:.2f}")
    mcol2.metric("Std Dev", f"{result.std_dev:.2f}")
