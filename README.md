# Project Risk Management

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-risk-luxf3rre.streamlit.app/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LuxF3rre/project_risk)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type%20checker-ty-blue.svg)](https://github.com/astral-sh/ty)
[![Build](https://github.com/LuxF3rre/project_risk/actions/workflows/test.yml/badge.svg)](https://github.com/LuxF3rre/project_risk/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/LuxF3rre/project_risk/graph/badge.svg?token=OuwuSu1lm4)](https://codecov.io/gh/LuxF3rre/project_risk)

A Streamlit app for quantifying project schedule risk
using **PERT distributions** and **Monte Carlo simulation**.

Instead of relying on single-point estimates, enter best-case,
most-likely, and worst-case durations to explore the full range
of possible outcomes.

## Use Cases

- **Single Project** — One overall three-point estimate.
  See the probability distribution of possible durations.
- **Task List** — Sequential tasks that run one after another.
  See how individual uncertainty compounds.
- **Task DAG** — Tasks with dependencies and parallelism.
  Identifies the critical path driving overall duration.
- **Task Workflow** — Cyclic probabilistic workflows with feedback loops.
  Simulates random walks through graphs where tasks can repeat (e.g. rework, retesting).

## Quick Start

```bash
uv sync
uv run streamlit run src/Home.py
```

## Tech Stack

- **Python 3.12+**
- **Streamlit** — Web UI
- **NumPy** — Vectorized PERT sampling and critical path
- **NetworkX** — DAG construction, topological sort

## Project Structure

```text
project_risk/
├── src/
│   ├── Home.py                  # Landing page
│   ├── pages/
│   │   ├── 1_Single_Project.py  # Single three-point estimate
│   │   ├── 2_Task_List.py       # Sequential tasks
│   │   ├── 3_Task_DAG.py        # Task dependency graph
│   │   └── 4_Task_Workflow.py   # Cyclic probabilistic workflow
│   └── project_risk/
│       ├── models.py            # Domain models
│       ├── pert.py              # PERT distribution sampling
│       ├── monte_carlo.py       # Simulation orchestration
│       ├── dag.py               # Graph algorithms
│       ├── workflow.py          # Cyclic workflow simulation
│       ├── charts.py            # Altair charts & Graphviz
│       └── ui.py                # Shared sidebar config
└── tests/                       # pytest suite
```

## Development

```bash
uv run ruff check --fix          # lint
uv run ruff format               # format
uv run ty check src tests        # type check
uv run pytest -v --cov --cov-branch
```

## License

MIT
