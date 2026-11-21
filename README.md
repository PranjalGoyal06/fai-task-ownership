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

## Task ownership lab

Run the unified app from the repository root. The historical page ships with a frozen Django preset and works offline. The sidebar also opens the existing synthetic simulator.

```sh
python -m streamlit run lab.py
```

**Explore the published replay.** The opening comparison uses the original six-acquirer settings and seed 331. Choose **Inspect a fix** to compare both policies’ reports and assignments against the patch. Switch between newly covered and newly missed fixes to see both sides of the coverage result. The project import graph includes every module at the displayed pre-fix revision. Select a module to highlight its connections or a connection to see its source citations. A two-headed arrow carries citations for both directions.

**Change the experiment** opens scoring strength first. Expand **Advanced settings** for private incentives, audits, credit, reward and seed. The Django preset has nine fixed owners. The agent-count control changes how many receive a private incentive.

The lab keeps the ownership probabilities fixed and refits numerical reporting policies on the earlier validation period each time settings change. Press **Run replay** to apply edits. Both policies face the same later fixes and private type draws. The pooled benchmark for each reward uses pre-fix evidence. Changed files serve as the ownership proxy. Trying settings on the held-out issues is exploratory.

**Save and replay.** Under **Reproduce this run**, download the settings as JSON or the complete ZIP with both policies’ issue and agent tables. Restore the JSON under **Restore a replay recipe** in the experiment view or run it from the command line. The recipe records the frozen input digest. Use the same checkout and the [recorded Python environment](reports/e1/environment.txt) for byte comparisons.

```sh
python -m src.lab.replay --recipe recipe.json --output replay.zip
```

**How the code fits together.** The Streamlit page calls the headless runner in [src/lab](src/lab). The [reporting game](src/e1/game.py) fits the strategy and computes reports, audits and assignment. The [E1 replay](src/e1/replay.py) settles the assignment against the patch. [Mechanisms](src/mechanisms.py) supplies centered scoring and credit allocation. The synthetic page continues to use its own simulation runner. The lab uses the same historical engine for interactive and command-line runs.

The portable supplement contains validation estimates and graph degrees, pre-fix evidence summaries, and pooled assignments at rewards 5, 8 and 12. It joins the published cohort, predictions and method summaries in manifest order. To regenerate it, follow the [E1 reproduction guide](reports/e1/REPRODUCE.md) through evidence extraction, then run the following commands with the pinned environment. The preparation step refits the original predictors and checks their held-out estimates against the release.

```sh
python -m src.lab.prepare results/inputs reports/e1 results/replay-inputs.json.gz
cmp results/replay-inputs.json.gz presets/django/replay-inputs.json.gz
```

See the [lab progress log](LAB_PROGRESS.md) for verification and the next milestone.

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
