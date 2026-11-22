# Deciding Task Ownership from Private Evidence under Misaligned Incentives

Multi-Agent Systems course project at IIT Gandhinagar. We study how to assign tasks when agents have private evidence about who should own a task and may benefit from overstating or understating their responsibility.

**Authors:** Pranjal Goyal, Ujas Shah, Harshal Kundanlal Rana, Satyakam Mishra

---

## Project Structure

```
fai-task-ownership/
├── .gitignore                  # Ignores .venv, caches, and generated outputs/plots
├── pyproject.toml              # Project metadata, pytest configuration, dependencies
├── requirements.txt            # Python requirements (numpy, scipy, networkx, pandas, scikit-learn, streamlit, plotly, pyarrow)
├── AGENTS.md                   # Repository conventions and verification rules
├── FAI_Project_Idea_Review.md  # Research questions, model assumptions, baselines, and math specs
├── lab.py                      # Unified Streamlit entry point (Historical Replay, Repository Experiment, Synthetic Simulation)
├── app.py                      # Standalone Streamlit application for synthetic simulations & sensitivity sweeps
├── app/
│   └── lab/
│       ├── historical.py       # Streamlit UI for historical replay, fix inspector & paper research grid tabs
│       ├── setup.py            # Streamlit UI for public GitHub repository onboarding & model selection
│       ├── import_graph.css    # D3 layout styling for repository import dependency graph
│       └── import_graph.js     # D3 interactive graph visualization script
├── presets/
│   └── django/
│       └── replay-inputs.json.gz # Frozen pre-fix AST features, imports, and validation estimates for offline replay
├── reports/
│   ├── e1/                     # Lexical classifier outputs (methods_benchmark.csv, RQ1-RQ3 sweeps, manifest, cohort)
│   └── e2/                     # Direct LLM outputs (methods_benchmark.csv, RQ1-RQ3 sweeps, table4_e2.tex)
├── scratch/
│   ├── run_benchmarks.py       # Headless script to reproduce Table 3 benchmark presets
│   └── benchmark_results.json  # Raw verified output metrics for all operational regimes
├── src/
│   ├── __init__.py
│   ├── types.py                # Core dataclasses: Task, Evidence, PublicReport, MechanismConfig, Enums
│   ├── environment.py          # Dependency graph G generation (Scale-free, Erdős-Rényi), task generator
│   ├── evidence.py             # Semantic similarity, Bayesian belief calibration, overconfidence map
│   ├── agents.py               # Strategic agent reasoning, dynamic costs, report shading, online learning
│   ├── mechanisms.py           # Escalation, exact combinatorial conjunctive solver, Brier scoring, graph audits, Shapley credit
│   ├── metrics.py              # Social welfare, Oracle benchmark, Miss Loss, Excess Cost, Regret decomposition
│   ├── baselines.py            # B1 (Description Router), B2 (Polling Router), B3 (Voting), B4 (Oracle)
│   ├── simulation_runner.py    # Configurable simulation engine, parameter specifications, multi-round loop, sweeps
│   ├── experiment_cards.py     # Labeled stress-test scenario definitions and single-parameter comparison cards
│   ├── e1/                     # Empirical lexical pipeline: dataset intake, AST features, game equilibrium, settlement
│   ├── e2/                     # Direct LLM pipeline: research_grid sweeps, Ollama providers, semantic extraction
│   ├── lab/                    # Historical replay engine, repository scanner, offline cache management, preset loader
│   └── lab_ui/                 # Network visualization bridge and D3 component wrappers
└── tests/
    ├── test_environment.py     # Verification of graph topology, cluster sampling, and weighted embeddings
    ├── test_scoring.py         # Verification of strictly proper Brier score properties
    ├── test_mechanisms.py      # Verification of escalation order, conjunctive selection, and graph audits
    ├── test_credit.py          # Verification of Shapley efficiency & dummy player axiom vs Equal Split
    ├── test_learning.py        # Verification of agent shading dynamics, score scaling, and audit deterrence
    ├── test_metrics.py         # Verification of Oracle optimality and exact regret decomposition
    ├── test_baselines.py       # Verification of B1 stale description degradation and B3 voting
    ├── test_app.py             # Automated tests for simulation runner and app configuration
    ├── test_e1.py              # Verification of E1 lexical feature extraction, selection, and settlement
    ├── test_e2.py              # Verification of E2 LLM beliefs, prompt contracts, and paired controls
    ├── test_research_grid.py   # Verification of RQ1-RQ3 sweep reproducibility and monotonicity
    ├── test_lab_replay.py      # Verification of offline replay reproducibility and recipe serialization
    ├── test_lab_direct.py      # Verification of direct LLM probability caches and fallback logic
    ├── test_lab_imports.py     # Verification of AST import extraction across Python module roots
    ├── test_lab_ui.py          # Verification of Streamlit lab helpers, presets, and state management
    ├── test_public_repository.py # Verification of Git history ingestion and production root filtering
    └── test_repository_pilot.py  # Verification of multi-agent ownership mapping and PR validation rules
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

## 🔬 Complete Scientific Reproducibility Guide

This section provides the exact workflows, parameters, random seeds, and Streamlit navigation steps required to reproduce every single numerical result reported in [`main.tex`](main.tex).

### Quick Seed & Provenance Summary

| Report Table | Description | Environment / Data | Random Seed | Primary Workflow / UI Preset |
| :--- | :--- | :--- | :--- | :--- |
| **Table 3** | Benchmark Performance across Operational Regimes | Synthetic Simulation ($T=100$ rounds) | `seed = 42` | `streamlit run app.py` (Presets 1, 2, 5, 6, 9, 10) OR `python scratch/run_benchmarks.py` |
| **Table 4 (Panel A)** | Empirical Repository Benchmarks: E1 (Lexical) | Django 2022 Test Split (234 issues, $N=9$) | `seed = 331` | `streamlit run lab.py` (Historical Replay $\to$ Grid Tables) OR `python -m src.e2.research_grid` |
| **Table 4 (Panel B)** | Empirical Repository Benchmarks: E2 (LLM) | Django 2022 Test Split (234 issues, $N=9$) | `seed = 331` | `reports/e2/methods_benchmark.csv` / Grid Sweeps |
| **Section 6.2 RQ1–RQ3** | Padding Reduction ($665 \to 483$), Multi-module ($6/27$ vs $9/27$), Zero Audits | Django 2022 Test Split (234 issues) | `seed = 331` | `streamlit run lab.py` (Opening Overview Scorecards) |

---

### 1. Reproducing Synthetic Simulation Results (Table 3)

Table 3 compares Mechanism $\mathcal{M}$ against Oracle ($\mathcal{M}^\circ$), Stale Description Routing (B1), Naive Strategic Polling (B2), and Plurality Voting (B3) across six canonical operational regimes over $T=100$ sequential rounds with **`seed = 42`**.

#### Option A: Interactive Streamlit Web UI (Visual Replay)
1. Launch the simulation app from the repository root:
   ```bash
   streamlit run app.py
   # Or run: python -m streamlit run lab.py (and choose "Synthetic simulation" from sidebar)
   ```
2. In the left sidebar under **⚡ Choose Benchmark Preset**, select the corresponding preset:
   - **Regime 1 (Multi-Service Incident):** Select `1. Multi-Service Incident (Conjunctive Default)`
   - **Regime 2 (Deflection Crisis):** Select `5. Multi-Service Deflection Crisis (Conjunctive — High Miss Loss)`
   - **Regime 3 (Acquisition Hoarding):** Select `6. Multi-Service Acquisition Hoarding (Conjunctive — High Excess Cost)`
   - **Regime 4 (Specialist Dispatch):** Select `2. Specialist & Reviewer Dispatch (Disjunctive — Shows Voting B3)`
   - **Regime 5 (Legacy Catalog Drift):** Select `9. Legacy Catalog Failure: High Description Staleness (ρ = 0.70)`
   - **Regime 6 (Fully Aligned Control):** Select `10. Fully Aligned Baseline: Truthful Reporting (ℓ = 0)`
3. Ensure **Random Seed** is set to `42` (default for all presets).
4. Click the large button: **🚀 Run Full Simulation (100 rounds)**.
5. In the main pane, inspect the **Comparative Benchmark Results** KPI scorecard and summary table. The realized numbers match Table 3:

| Operational Regime | System | Welfare ($W$) | Eff. Ratio ($\eta$) | Miss Loss ($\Delta V$) | Excess Cost ($\Delta C$) | Resolution Rate | Net Welfare |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Multi-Service Incident** | Oracle ($\mathcal{M}^\circ$) | 11.36 | 1.000 | 0.00 | 0.00 | 100.0% | 11.36 |
| *(Conjunctive Default, $N=10$, $k \in [1, 5]$)* | **Mechanism $\mathcal{M}$** | **4.78** | **0.421** | **3.15** | **3.43** | **66.0%** | **4.56** |
| | Description (B1) | 1.47 | 0.129 | 7.85 | 2.04 | 46.0% | 1.47 |
| | Polling Router (B2) | -1.44 | -0.127 | 11.00 | 1.80 | 27.0% | -1.44 |
| **2. Deflection Crisis** | Oracle ($\mathcal{M}^\circ$) | 11.36 | 1.000 | 0.00 | 0.00 | 100.0% | 11.36 |
| *(50% Deflectors $\ell=+3.0$)* | **Mechanism $\mathcal{M}$** | **4.58** | **0.403** | **4.20** | **2.58** | **64.0%** | **2.77** |
| | Description (B1) | 1.47 | 0.129 | 7.85 | 2.04 | 46.0% | 1.47 |
| | Polling Router (B2) | -1.11 | -0.098 | 12.50 | -0.03 | 21.0% | -1.11 |
| **3. Acquisition Hoarding** | Oracle ($\mathcal{M}^\circ$) | 11.36 | 1.000 | 0.00 | 0.00 | 100.0% | 11.36 |
| *(50% Acquirers $\ell=-3.0$)* | **Mechanism $\mathcal{M}$** | **5.57** | **0.490** | **2.70** | **3.09** | **60.0%** | **9.31** |
| | Polling Router (B2) | 4.98 | 0.438 | 3.10 | 3.28 | 65.0% | 4.98 |
| | Description (B1) | 1.47 | 0.129 | 7.85 | 2.04 | 46.0% | 1.47 |
| **4. Specialist Dispatch** | Oracle ($\mathcal{M}^\circ$) | 4.00 | 1.000 | 0.00 | 0.00 | 100.0% | 4.00 |
| *(Disjunctive Single-Owner $N=8, k=1$)* | Voting (B3) | 3.65 | 0.912 | 0.35 | 0.00 | 93.0% | 3.65 |
| | **Mechanism $\mathcal{M}$** | **3.38** | **0.845** | **0.00** | **0.62** | **100.0%** | **3.68** |
| | Description (B1) | 2.95 | 0.738 | 0.00 | 1.05 | 100.0% | 2.95 |
| | Polling Router (B2)$^*$ | 0.32 | 0.080 | 1.30 | 2.38 | 74.0% | 0.32 |
| **5. Legacy Catalog Drift** | Oracle ($\mathcal{M}^\circ$) | 11.36 | 1.000 | 0.00 | 0.00 | 100.0% | 11.36 |
| *(Staleness $\rho = 0.70$)* | **Mechanism $\mathcal{M}$** | **4.78** | **0.421** | **3.15** | **3.43** | **66.0%** | **4.56** |
| | Description (B1) | -1.21 | -0.107 | 11.15 | 1.42 | 19.0% | -1.21 |
| | Polling Router (B2) | -1.44 | -0.127 | 11.00 | 1.80 | 27.0% | -1.44 |
| **6. Fully Aligned Control** | Oracle ($\mathcal{M}^\circ$) | 11.36 | 1.000 | 0.00 | 0.00 | 100.0% | 11.36 |
| *(100% Honest $\ell = 0$)* | Polling Router (B2) | 5.79 | 0.510 | 3.20 | 2.37 | 77.0% | 5.79 |
| | **Mechanism $\mathcal{M}$** | **5.57** | **0.490** | **2.65** | **3.14** | **61.0%** | **5.00** |
| | Description (B1) | 1.47 | 0.129 | 7.85 | 2.04 | 46.0% | 1.47 |

> [!NOTE]
> **Deterministic vs. Disjunctive Tie-Breaking ($^*$):** Regimes 1, 2, 3, 5, and 6 evaluate exact combinatorial conjunctive selection and are 100% bitwise deterministic across all baselines. In Regime 4 (Disjunctive), Oracle, Description (B1), Voting (B3), and Mechanism $\mathcal{M}$ are also 100% deterministic ($W=3.38$). For Polling Router (B2), both acquiring agents shade bids to $r_i=1.0$ without penalty, tying on $r_i/c_i=1.0$; escalation resolves exact ties via randomized tie-breaking, yielding mean welfare centered at $W=0.32$ (seed 3 tie-break yields the exact $W=0.32, \Delta V=1.30, \Delta C=2.38$).

#### Option B: Headless Command Line (CLI Replay)
To run all benchmark regimes simultaneously and verify output metrics without the browser UI:
```bash
python scratch/run_benchmarks.py
```
This script executes all preset configurations with `seed=42`, prints summary metrics to stdout, and saves the verified results to `scratch/benchmark_results.json`.

---

### 2. Reproducing Empirical Software Repository Results (Table 4 & RQ1–RQ3)

Table 4 reports performance on the canonical SWE-bench Django testbed ($N=9$ module custodian agents, $R=8.0$, $c_i=1.0$, 234 held-out evaluation issues from calendar year 2022). All empirical reporting games use **`seed = 331`**.

#### Option A: Interactive Streamlit Web UI (Historical Replay)
1. Launch the unified task ownership lab:
   ```bash
   python -m streamlit run lab.py
   ```
2. The app opens directly to the **Historical replay** view, pre-loaded with the published Django preset (`presets/django/replay-inputs.json.gz`).
3. **Verify Overview Scorecards (Section 6.2 Key Takeaways):**
   - **Padding Elimination:** Polling assignments $665 \to$ Scoring assignments $483$ (exact elimination of 182 redundant assignments; $181/182 = 99.5\%$ untouched).
   - **Resolution Coverage:** Scoring covers **114 fixes (48.7%)**, up from **110 fixes (47.0%)** under strategic polling.
   - **Multi-Module Drop:** Coverage on collaborative fixes ($|S^*| \ge 2$) falls from **$9/27 \to 6/27$** ($33.3\% \to 22.2\%$) due to conjunctive Brier risk.
4. **Inspect Table 4 Panel A (Methods Benchmark):**
   - Click **Change the experiment** in the UI.
   - Scroll down and expand **📊 Paper Research Questions (Grid Replay & Tables)**.
   - Open Tab **Methods Benchmark (Table 4)** to view the exact Panel A table:

| Policy / Mechanism | Welfare ($W$) | Eff. Ratio ($\eta$) | Miss Loss ($\Delta V$) | Excess Cost ($\Delta C$) | Coverage (%) | Padding | Net Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Proxy ($\mathcal{M}^\circ$)** | 3.71 | 1.000 | 0.00 | 0.00 | 76.5% | 0.00 | 0.00 |
| **Centralized Text Classifier** | 3.54 | 0.954 | 1.41 | 0.32 | 60.7% | 0.32 | 0.00 |
| **Honest Decentralized Polling ($\ell = 0$)** | 2.81 | 0.757 | 2.02 | 0.46 | 50.0% | 0.65 | 0.00 |
| **Plurality Voting (B3)** | 1.32 | 0.356 | 3.78 | 0.18 | 29.1% | 0.18 | 0.00 |
| **Stale Description Router (B1)** | 1.24 | 0.334 | 4.38 | 0.08 | 19.2% | 0.08 | 0.00 |
| **Strategic Polling (B2, $\phi=2/3, \ell=-1.0$)** | 0.38 | 0.102 | 2.26 | 2.84 | 47.0% | 2.84 | 0.00 |
| **Mechanism $\mathcal{M}$ (Proper Scoring $\lambda=8$)** | **1.29** | **0.348** | **2.14** | **1.47** | **48.7%** | **2.06** | **+0.40** |
| **Mechanism $\mathcal{M}$ + Audit ($\lambda=8, \alpha=0.25$)** | 1.29 | 0.348 | 2.14 | 1.47 | 48.7% | 2.06 | +0.63 |
| **Mechanism $\mathcal{M}$ + Shapley ($\lambda=8, \alpha=0.25$)** | 0.69 | 0.186 | 3.66 | 2.52 | 54.3% | 3.04 | -0.50 |

   - Inspect Tab **RQ1: Misalignment** to view scaling with $\ell$ and $\phi$ (`reports/e1/rq1_misalignment.csv`).
   - Inspect Tab **RQ2: Scoring & Audit** to view padding suppression across $\lambda \in [0, 16]$ and audit rates $\alpha \in [0, 0.50]$ (`reports/e1/rq2_scoring_audit.csv`).
   - Inspect Tab **RQ3: Coalition Credit** to view Equal Split vs. Shapley value across $R \in \{5, 8, 12\}$ (`reports/e1/rq3_credit_coalitions.csv`).

#### Option B: Headless Command Line (CLI Replay)
To execute the standardized evaluation harness and generate the benchmark tables offline:
```bash
# Generate Table 4 methods benchmark and RQ1-RQ3 sweeps directly
python -m src.e2.research_grid presets/django/replay-inputs.json.gz --output reports/e1
```
Or verify the numbers instantly via Python:
```bash
python -c "from src.lab.preset import load_preset; from src.e2.research_grid import generate_methods_benchmark; print(generate_methods_benchmark(load_preset(), seed=331).to_string())"
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

### Run E1 on a public Python repository

Open **New repository experiment**, enter a GitHub repository URL and choose the merge dates. New scans use the GitHub CLI (`gh`) with authenticated public read access. The initial check reads file metadata and pins a revision. Confirm the production roots and review the estimated Git history size and available storage. For repositories that fit, checking history acquires one compressed bare clone without a working checkout. Larger repositories use shallow pre-fix trees and bulk Git packs of the required file versions. GitHub supplies issue and pull request metadata. Estimates can differ from actual disk use. The lab monitors transfers against the repository, cache and free-space limits. Completed Git objects and older per-file caches are reusable.

The generic scan supports flat modules, packages and `src/` layouts. Repositories with eligible production Python can also contain compiled extensions or other languages. Setup warns about those files and excludes any fix that changes unsupported production source, including fixes that also change Python. The exclusion table lists the affected pull requests. Replay results cover eligible Python fixes, and the graph resolves imports within the selected Python source. The scan recognizes tests, documentation and packaging files by paths and names. It also excludes changes with unverifiable patch paths, ambiguous issue links or unavailable issue text. An incomplete GitHub tree or an exhausted quota interrupts the scan. Retry the same URL and dates to reuse completed batches.

The initial ownership map comes from directories and filenames in the earliest usable pre-fix tree. Review per-rule file counts before accepting. A rule that first matches later in the history is allowed. A rule that never owns an eligible file is flagged. The timeline, training results, directed import graph, inspector and ZIP export use the same E1 engine as the curated experiments. **Open a saved experiment** reopens trained results offline.

After choosing dates, select **Trained classifier** or **Direct model probabilities**. The direct option asks the local Ollama model for one probability per issue and owner. Each request contains the issue, the owner's earlier rate, and short excerpts from that owner's pre-fix source. It never includes a later patch or source owned by another agent. **Try a small run** uses up to ten issues for fitting reporting strategies and ten issues for checking results. Completed responses are cached and a retry resumes an interrupted run. Game settings reuse saved probabilities.

The direct option defaults to `qwen3.5:9b-q4_K_M` through a local Ollama server. It uses structured JSON, temperature zero, and disables reasoning output. The saved inputs record the model tag, installed-model digest when available, prompt version, source revision, and run settings. Direct model results are exploratory. A model may already have encountered public issues or source code before the experiment. [E2-D design](E2D_DESIGN.md) records the corresponding evidence boundaries and evaluation limits.

| Storage setting | Default | Environment override in MiB |
| --- | ---: | --- |
| Production source in one snapshot | 100 MiB | `LAB_SNAPSHOT_MIB` |
| Local data for one repository | 512 MiB | `LAB_REPOSITORY_MIB` |
| Total lab cache | 2 GiB | `LAB_TOTAL_MIB` |
| Disk space kept free | 2 GiB | `LAB_RESERVE_MIB` |

The lab checks space again under a shared write lock before publishing each cached file. Generic data lives under `results/repositories/`. The total budget also counts existing scikit-learn caches. **Manage downloaded data** removes a repository’s downloaded objects, scan metadata and extracted evidence while keeping its prepared replay inputs.

For a headless run, the first command prints the history estimate and available budget. Add `--prepare` to acquire history and print the scan path, then pass that path to training and the resulting inputs path to replay.

```sh
python -m src.lab.repository https://github.com/OWNER/PROJECT --start 2025-01-01 --end 2025-12-31
python -m src.lab.repository https://github.com/OWNER/PROJECT --start 2025-01-01 --end 2025-12-31 --prepare
python -m src.lab.training --scan SCAN_PATH
python -m src.lab.replay --inputs INPUTS_PATH --output results/experiment.zip
```

### Run the scikit-learn pilot

Choose **Review scikit-learn pilot** in setup to revisit its cached 2022 scan, or **Open a saved experiment** to open a trained replay. The pilot retains its original seven-group suggestion and mixed-patch policy. Its separate reproduction commands below use Git, the GitHub CLI and an ignored cache under `results/sklearn-2022/`.

1. **Review suitability.** The lab scans PRs merged in 2022. Each retained fix has one explicit link to a same-repository issue, a pre-fix source revision and a verified production Python change. Changed Python files stand in for the owners needed to resolve an issue. Mixed patches can also change compiled code outside that measure.
2. **Define agents.** Edit the names and production paths in the ownership table. A specific file or folder overrides its parent. The preview shows the resulting file owners, an actual patch and its pre-fix imports. A fix with an unmapped eligible path is excluded in full.
3. **Choose dates.** The proposed dates aim for a 60/20/20 split. The replay time is the later of issue creation and the first parent’s commit time. Issues with observed edits after that time are excluded. GitHub’s current issue record does not reconstruct every historical title. A training fix must merge before validation starts, and a validation fix must merge before evaluation starts. Repeated issue IDs, merge revisions and identical normalized report texts are excluded.
4. **Train models** to see example counts, probability errors and any fallback for each agent. **Open reporting game** compares policies using the saved probabilities. Changing the ownership map requires retraining. Changing scoring reuses the trained inputs.

To prepare the same suggested map and dates without the UI, run these commands from the worktree root. Both commands accept `--cache DIRECTORY`. Training also accepts `--map owners.json` (a list of objects with `name` and `paths`) and `--cutoffs YYYY-MM-DD YYYY-MM-DD` for the starts of validation and evaluation. Intake accepts `--cross-check PATH` to compare a local SWE-bench parquet against the finished scan.

```sh
python -m src.lab.intake
python -m src.lab.training
```

The training command prints the path to `inputs.json.gz`. Pass it to `python -m src.lab.replay --inputs PATH --output experiment.zip` to export the default comparison. To reproduce an exported scikit-learn run offline in the same software environment, extract its frozen inputs and settings and compare the complete output.

```sh
unzip -o experiment.zip inputs.json.gz recipe.json -d results/reproduction
python -m src.lab.replay --inputs results/reproduction/inputs.json.gz --recipe results/reproduction/recipe.json --output results/reproduction/replayed.zip
cmp experiment.zip results/reproduction/replayed.zip
```

## Running the Interactive Streamlit UI

The repository includes two complementary web interfaces:

1. **Unified Task Ownership Lab** (contains Historical Replay, New Repository Experiments, and Synthetic Simulation):
   ```bash
   streamlit run lab.py
   ```
2. **Dedicated Synthetic Simulation Engine** (direct access to full parameter tuning, live sensitivity sweeps, and Plotly KPI stacks):
   ```bash
   streamlit run app.py
   ```

Features included in the synthetic web UI (`app.py`):
- **🎛️ Tunable Parameters with One-Liner Descriptions**: Tune every single constant across the environment, evidence calibration, scoring rules, graph audits, coalition credits, strategic learning, and baselines.
- **⚡ Benchmark Presets**: 1-click execution for standard Mode 1 benchmarks, Mode 2 dynamic workloads, deflection crises, acquisition hoarding, and high-staleness stress tests.
- **📊 Comparative Benchmark Results**: KPI scorecards, benchmark comparison tables, and interactive Plotly regret decomposition stacks ($\Delta V$ Miss Loss vs $\Delta C$ Excess Cost).
- **📈 Convergence Dynamics**: Multi-round cumulative welfare, running regret trajectories, and dynamic workload fatigue evolution.
- **🔬 Strategic Shading & Audits**: Live scatter plots showing how ex-post proper scoring ($\lambda$) contracts strategic misreporting cones back toward truthful beliefs, along with audit deterrence metrics.
- **🕸️ Repository Topology & Round Inspector**: Interactive 2D dependency graph $G$ and round-by-round task inspection with root-cause tracking and assignment comparisons.
- **🎛️ Sensitivity Sweeps & Grid Search**: Run repeated one-dimensional sweeps or vary multiple numeric parameters simultaneously over user-defined ranges and steps. Results include averages, variability, regret recovery, incentive overhead, report shading, tampering, coalition coverage, and padding for the project research questions.
- **🧪 Interpretable Experiment Cards**: Run labeled, deterministic stress scenarios on demand (population size, deflection, acquisition, scoring, auditing, penalties, and missing evidence), inspect their parameters, and compare them with B2 and B4.
- **📈 One-Parameter Comparison Cards**: Run compact 3–5 value comparisons for population size, deflection strength, audit rate, audit penalty, or scoring strength with tables and plots.
- **📖 FAI Theoretical Guide**: Full reference of mathematical formulas, assumptions, and research questions.

---

## Historical issue experiment

The [grounded replay report](reports/e1/REPORT.md) gives the results, uncertainty and limitations. The [reproduction guide](reports/e1/REPRODUCE.md) covers setup, verification and the complete run.
