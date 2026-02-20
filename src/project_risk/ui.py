"""Shared UI helpers for Streamlit pages."""

from __future__ import annotations

import streamlit as st

__all__ = ["sidebar_config"]


def sidebar_config() -> tuple[int, int | None, float]:
    """Render shared simulation settings in the sidebar.

    Returns:
        Tuple of (iterations, seed_or_none, pert_lambda).
    """
    st.sidebar.header("Simulation Settings")

    with st.sidebar.expander("What do these settings mean?"):
        st.markdown(
            """
**Iterations** — How many simulated scenarios to run. More iterations
give smoother, more reliable results.

**Seed** — Makes results reproducible. The same seed always produces
the same output.

**PERT Lambda** — Controls how strongly the simulation favors the
"most likely" value. Lower = wider spread, higher = tighter around
the most likely estimate. Default is 4.
"""
        )

    iterations = st.sidebar.number_input(
        "Iterations",
        min_value=100,
        max_value=10_000,
        value=10_000,
        step=1000,
    )

    lock = st.sidebar.checkbox("Lock results (reproducible)")
    seed: int | None = None
    if lock:
        seed_val = st.sidebar.number_input(
            "Seed",
            min_value=1,
            max_value=2**31 - 1,
            value=42,
        )
        seed = int(seed_val)

    pert_lambda = float(
        st.sidebar.number_input(
            "PERT Lambda",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Lower = wider spread, higher = tighter around most likely.",
        )
    )

    return int(iterations), seed, pert_lambda
