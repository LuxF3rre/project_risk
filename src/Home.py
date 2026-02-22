"""Project Risk Management - Home page."""

import streamlit as st

st.set_page_config(
    page_title="Project Risk Management",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Project Risk Management")

st.markdown(
    """
This app helps you **understand and quantify project risks** — the
uncertainties that affect whether a project finishes on time. Instead of
relying on single-point estimates, you explore the full range of possible
outcomes using **PERT distributions** and **Monte Carlo simulation**.
"""
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Single Project")
    st.markdown(
        "One overall estimate — enter best-case, most-likely, and "
        "worst-case durations to see the range of possible outcomes."
    )
    st.page_link(
        "pages/1_Single_Project.py",
        label="Open Single Project",
        icon="📊",
    )

with col2:
    st.subheader("Task List")
    st.markdown(
        "Sequential tasks — enter estimates for each task and see how "
        "individual uncertainty compounds into overall project risk."
    )
    st.page_link("pages/2_Task_List.py", label="Open Task List", icon="📋")

with col3:
    st.subheader("Task DAG")
    st.markdown(
        "Parallel tasks with dependencies — define the network and see how "
        "the *critical path* drives overall project duration."
    )
    st.page_link("pages/3_Task_DAG.py", label="Open Task DAG", icon="🔀")

with col4:
    st.subheader("Task Workflow")
    st.markdown(
        "Cyclic workflows with probabilistic transitions — model loops "
        "like rework and retesting to see how they affect duration."
    )
    st.page_link(
        "pages/4_Task_Workflow.py",
        label="Open Task Workflow",
        icon="🔄",
    )

st.divider()
st.markdown(
    "<div style='text-align: center'>Made with ❤ by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
