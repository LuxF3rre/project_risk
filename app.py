"""Project Risk Management - Home page."""

import streamlit as st

st.set_page_config(
    page_title="Project Risk Management",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Project Risk Management")
st.subheader("PERT + Monte Carlo Simulation")

st.markdown(
    """
    Estimate project durations under uncertainty using **PERT distributions**
    and **Monte Carlo simulation**.

    ### Use Cases

    1. **Single Project** — Enter a single three-point estimate (optimistic,
       most likely, pessimistic) and visualize the probability distribution of
       project duration.

    2. **Task List** — Define a sequence of tasks, each with its own
       three-point estimate. The simulation sums all task durations to produce
       the total project duration distribution.

    3. **Task DAG** — Model task dependencies as a directed acyclic graph.
       The simulation computes the critical-path duration across all Monte
       Carlo iterations, capturing the effect of parallel paths.

    ---

    Use the **sidebar** to navigate between pages.
    """
)
