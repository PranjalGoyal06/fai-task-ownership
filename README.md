# Deciding Task Ownership from Private Evidence under Misaligned Incentives

Multi-Agent Systems course project at IIT Gandhinagar. We study how to assign tasks when agents have private evidence about who should own a task and may benefit from overstating or understating their responsibility.

**Authors:** Pranjal Goyal, Ujas Shah, Harshal Kundanlal Rana, Satyakam Mishra

---

## Project Structure

```
fai-task-ownership/
├── .gitignore                  # Ignores .venv, caches, and generated outputs/plots
├── pyproject.toml              # Project metadata, pytest configuration, dependencies
├── requirements.txt            # Python requirements (numpy, scipy, networkx, pytest, matplotlib)
├── AGENTS.md                   # Repository conventions and verification rules
├── FAI_Project_Idea_Review.md  # Research questions, model assumptions, baselines, and math specs
├── demo_simulation.py          # End-to-end benchmark comparison script
├── src/
│   ├── __init__.py
│   ├── types.py                # Core dataclasses: Task, Evidence, PublicReport, MechanismConfig
│   ├── environment.py          # Repository dependency graph G, module profiles, weighted task generation
│   ├── evidence.py             # Semantic similarity, Bayesian belief calibration, overconfidence map
│   ├── agents.py               # Strategic agent reasoning, dynamic costs, report shading, online learning
│   ├── mechanisms.py           # Escalation, conjunctive set selection, centered Brier scoring, graph audits, Shapley credit
│   ├── metrics.py              # Social welfare, Oracle benchmark, Miss Loss, Excess Cost, Regret decomposition
│   └── baselines.py            # B1 (Description Router), B2 (Polling Router), B3 (Voting), B4 (Oracle)
└── tests/
    ├── test_environment.py     # Verification of graph topology, cluster sampling, and weighted embeddings
    ├── test_scoring.py         # Verification of strictly proper Brier score properties
    ├── test_mechanisms.py      # Verification of escalation order, conjunctive selection, and graph audits
    ├── test_credit.py          # Verification of Shapley efficiency & dummy player axiom vs Equal Split
    ├── test_learning.py        # Verification of agent shading dynamics, score scaling, and audit deterrence
    ├── test_metrics.py         # Verification of Oracle optimality and exact regret decomposition
    └── test_baselines.py       # Verification of B1 stale description degradation and B3 voting
```

---

## Installation & Setup

1. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running Verification Tests

Run the complete automated test suite from the repository root (`fai-task-ownership/`):

```bash
pytest tests -v
```

---

## Running the Interactive Streamlit UI

Launch the interactive web application to tune all system constants, test benchmarks, inspect module topologies, step through individual task rounds, and run live sensitivity sweeps:

```bash
streamlit run app.py
```

Features included in the web UI:
- **🎛️ Tunable Parameters with One-Liner Descriptions**: Tune every single constant across the environment, evidence calibration, scoring rules, graph audits, coalition credits, strategic learning, and baselines.
- **⚡ Benchmark Presets**: 1-click execution for standard Mode 1 benchmarks, Mode 2 dynamic workloads, deflection crises, acquisition hoarding, and high-staleness stress tests.
- **📊 Comparative Benchmark Results**: Top-level KPI scorecards, comparison tables matching `demo_simulation.py`, and interactive Plotly regret decomposition stacks ($\Delta V$ Miss Loss vs $\Delta C$ Excess Cost).
- **📈 Convergence Dynamics**: Multi-round cumulative welfare, running regret trajectories, and dynamic workload fatigue evolution.
- **🔬 Strategic Shading & Audits**: Live scatter plots showing how ex-post proper scoring ($\lambda$) contracts strategic misreporting cones back toward truthful beliefs, along with audit deterrence metrics.
- **🕸️ Repository Topology & Round Inspector**: Interactive 2D dependency graph $G$ and round-by-round task inspection with root-cause tracking and assignment comparisons.
- **🎛️ Sensitivity Sweeps & Grid Search**: Run repeated one-dimensional sweeps or vary multiple numeric parameters simultaneously over user-defined ranges and steps. Results include averages, variability, regret recovery, incentive overhead, report shading, tampering, coalition coverage, and padding for the project research questions.
- **🧪 Interpretable Experiment Cards**: Run labeled, deterministic stress scenarios on demand (population size, deflection, acquisition, scoring, auditing, penalties, and missing evidence), inspect their parameters, and compare them with B2 and B4.
- **📈 One-Parameter Comparison Cards**: Run compact 3–5 value comparisons for population size, deflection strength, audit rate, audit penalty, or scoring strength with tables and plots.
- **📖 FAI Theoretical Guide**: Full reference of mathematical formulas, assumptions, and research questions.

---

## Running the Simulation via CLI

Execute the multi-round benchmark simulation comparing Oracle (B4), Stale Description Router (B1), Polling Router without incentives (B2), Voting (B3), and our Incentive Mechanism (M):

```bash
python demo_simulation.py
```

### Default Benchmark Parameters & Reproducibility
`demo_simulation.py` runs deterministically with `seed=42`, `num_rounds=100`, `num_agents=8`, `k_owners=2`, `lambda_scale=2.5`, `audit_rate=0.2`, `staleness_rho=0.3`, and `Rule=CONJUNCTIVE`. Both **Mode 1 (`dynamic_mode=False`, static types)** and **Mode 2 (`dynamic_mode=True`, workload fatigue & credit EMA)** are supported via `run_experiment(...)`.

## Historical issue experiment

The [grounded replay report](reports/e1/REPORT.md) gives the results, uncertainty and limitations. The [reproduction guide](reports/e1/REPRODUCE.md) covers setup, verification and the complete run.
