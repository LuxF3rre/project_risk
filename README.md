# Project Risk Management

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

## Quick Start

```bash
uv sync
uv run streamlit run Home.py
```

## Tech Stack

- **Python 3.12+**
- **Streamlit** — Web UI
- **NumPy** — Vectorized PERT sampling and critical path
- **NetworkX** — DAG construction, topological sort

## Project Structure

```text
project_risk/
├── Home.py                      # Landing page
├── pages/
│   ├── 1_Single_Project.py      # Single three-point estimate
│   ├── 2_Task_List.py           # Sequential tasks
│   └── 3_Task_DAG.py            # Task dependency graph
├── src/project_risk/
│   ├── models.py                # Domain models
│   ├── pert.py                  # PERT distribution sampling
│   ├── monte_carlo.py           # Simulation orchestration
│   ├── dag.py                   # Graph algorithms
│   ├── charts.py                # Altair charts & Graphviz
│   └── ui.py                    # Shared sidebar config
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
