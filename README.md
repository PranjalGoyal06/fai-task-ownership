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

Launch the interactive web application to tune all system constants, test benchmarks, inspect module topologies, step through individual task rounds, and run live sensitivity sweeps:

```bash
streamlit run app.py
```

Features included in the web UI:
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
