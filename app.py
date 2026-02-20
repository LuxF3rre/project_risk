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
This app helps you **estimate how long a project will really take** by
accounting for uncertainty in your time estimates. Instead of giving a
single number, you provide a range — the best case, the most likely case,
and the worst case — and the app runs thousands of simulated scenarios to
show you the full picture of possible outcomes.
"""
)

st.header("PERT + Monte Carlo Simulation")

st.markdown(
    """
**PERT** (Program Evaluation and Review Technique) is a way to turn your
three-point estimate (best case, most likely, worst case) into a realistic
probability curve. It gives more weight to the "most likely" value while
still considering the extremes.

**Monte Carlo simulation** takes that probability curve and rolls the dice
thousands of times. Each roll picks a random duration from the curve. After
all the rolls, you get a distribution that answers questions like *"What's
the chance we finish in under 20 days?"* or *"What duration covers 80% of
scenarios?"*.
"""
)

st.header("Use Cases")

st.markdown(
    """
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
