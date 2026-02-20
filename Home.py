"""Project Risk Management - Home page."""

import streamlit as st

st.set_page_config(
    page_title="Project Risk Management",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Home")

st.markdown(
    """
This app helps you **understand and quantify project risks** — the
uncertainties that affect whether a project finishes on time, within
budget, or meets its targets. Instead of relying on single-point
estimates, you explore the full range of possible outcomes so you can
make better-informed decisions.

Different types of risk call for different analysis methods. The app
is organized into modules, each focused on a specific technique.
"""
)

st.header("PERT + Monte Carlo Simulation")

st.markdown(
    """
The first module focuses on **schedule risk** — how long will the
project really take?

**PERT** (Program Evaluation and Review Technique) is a way to turn your
three-point estimate (best case, most likely, worst case) into a realistic
probability curve. It gives more weight to the "most likely" value while
still considering the extremes.

**Monte Carlo simulation** takes that probability curve and rolls the dice
thousands of times. Each roll picks a random duration from the curve. After
all the rolls, you get a distribution that answers questions like *"What's
the chance we finish in under 20 days?"* or *"What duration covers 80% of
scenarios?"*.

**Use cases:**

1. **Single Project** — You have one overall estimate for a project.
   Enter your best-case, most likely, and worst-case durations to see the
   range of possible outcomes.

2. **Task List** — Your project has several tasks that run one after
   another. Enter estimates for each task and see how the total project
   duration distributes when individual task uncertainty adds up.

3. **Task DAG** — Your project has tasks that depend on each other, and
   some can run in parallel. Define the dependencies and see how the
   *critical path* (the longest chain of dependent tasks) drives the
   overall project duration.
"""
)

st.divider()
st.caption("Use the **sidebar** to navigate between pages.")
