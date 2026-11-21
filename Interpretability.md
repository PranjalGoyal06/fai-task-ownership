# Interpretability Guide

This document explains how to read the simulator from the outside in:

1. choose sidebar parameters;
2. understand how one task is generated;
3. follow one round through the agents and mechanisms;
4. understand how the repeated-round results become charts and tables.

The implementation is a synthetic simulator for the proposal. It is not a
production task router and it does not learn from a real repository yet.

## 1. The complete simulation flow

The main implementation is
[`run_full_simulation`](./src/simulation_runner.py), which creates the world
once and then repeats the round loop `T` times.

### 1.1 Initialization before the loop

The simulator first creates:

1. **A repository environment**
   - `N` agents/modules.
   - One random capability/profile vector per agent.
   - A dependency graph `G`.
   - Shortest-path distances used when tasks are clustered around a root module.

2. **A task generator**
   - Samples the hidden required-owner set `S*`.
   - Chooses a root module.
   - Expands the owner set through graph neighbors or random fallback.
   - Builds an issue embedding from the owner profiles.
   - Adds task-generation noise.
   - Assigns the reward and social cost.

3. **An evidence engine**
   - Compares each agent profile with the issue embedding.
   - Adds observation noise.
   - Converts the signal into a local probability `pi_true`.
   - Applies calibration distortion to produce `pi_hat`.
   - Sometimes removes the signal entirely according to `gamma`.

4. **Mechanism M**
   - Configures scoring, audits, detection, penalties, and credit.

5. **Baselines**
   - B1: stale description router.
   - B2: polling router without incentives.
   - B3: voting router.
   - B4: oracle.

6. **Two agent populations**
   - Mechanism M agents receive incentives.
   - B2 agents are isolated and do not receive the incentive layer.
   - This prevents learning state from leaking between systems.

### 1.2 One round of the loop

For each round `t` from `1` to `T`:

#### Step A: Generate a hidden task

The task generator creates `S*`, the true set of required owners. The agents
do not receive `S*` directly. They receive only their own evidence later.

The task also contains:

- issue embedding;
- root module;
- reward `R`;
- per-agent social costs `c_i`;
- conjunctive or disjunctive resolution rule.

#### Step B: Generate private evidence

Every agent receives a private evidence object:

- `raw_signal`: its noisy similarity to the task;
- `pi_true`: calibrated probability estimate;
- `pi_hat`: subjective, possibly miscalibrated probability;
- private evidence payload.

An agent with missing evidence receives the base-rate probability instead of a
signal.

#### Step C: Agents submit reports

Each strategic agent converts `pi_hat` into a public report `r_i`.

- Aligned agents tend to report near their belief.
- Deflectors (`ell > 0`) prefer not to be assigned and shade reports downward.
- Acquirers (`ell < 0`) benefit from assignment and shade reports upward.
- Scoring pulls reports toward truthful beliefs.
- Agents may tamper with their payload if the expected benefit exceeds the
  expected audit penalty.

The report and payload are stored separately. The report affects selection;
the payload is relevant to audits.

#### Step D: Select owners under each system

The same task is evaluated through different routers:

- B1 uses stale capability descriptions.
- B2 uses reports without scoring or audits.
- B3 uses noisy peer voting.
- M uses the configured scoring/audit/credit mechanism.
- The honest benchmark uses `pi_true`.
- B4 uses the oracle implementation.

For disjunctive tasks, escalation stops when a required owner is reached.
For conjunctive tasks, the mechanism chooses a set using expected resolution
value minus assignment cost.

#### Step E: Calculate welfare and regret

For each assignment, the simulator computes:

- whether the task resolved;
- social cost of assigned agents;
- gross social welfare;
- regret against B4;
- miss loss from failing to resolve;
- excess cost from assigning too much;
- efficiency ratio.

The important distinction is:

- **Miss loss:** correct owners were not assigned.
- **Excess cost:** unnecessary owners were assigned.

#### Step F: Apply audits, scores, and credit

After selection:

- audits inspect submitted payloads;
- detected tampering receives a penalty;
- reports receive centered Brier scores;
- resolved assigned agents receive equal-split or Shapley credit;
- audit cost and score transfers are recorded.

Because audits are currently ex-post, they mainly affect tampering, penalties,
and incentive overhead in static mode. They do not retroactively change the
assignment already made in that round.

#### Step G: Update dynamic agent state

When dynamic mode is enabled:

- assignment load changes fatigue;
- credits and penalties update agent state;
- assignment observations update each agent's internal assignment model;
- later reports can therefore differ from earlier reports.

When dynamic mode is disabled, agents still maintain bookkeeping, but fatigue
does not affect the current strategic cost.

#### Step H: Store history

The simulator stores:

- per-system round metrics;
- task owner set;
- all reports;
- private beliefs;
- all assignments;
- audited and contradicted agents;
- agent-level shading, assignment, credit, penalty, and tampering counts.

This history powers the UI tabs and the experiment cards.

## 2. What the main loop variables mean

### `T`: Number of rounds

`T` controls how many independent tasks are presented sequentially.

- Larger `T` gives more observations and smoother averages.
- Larger `T` makes running simulations slower.
- In dynamic mode, larger `T` also gives agents more opportunities to learn,
  accumulate fatigue, receive credit, and receive penalties.
- A small `T` is useful for quick debugging but can produce noisy results.

### `N`: Number of agents

`N` controls the number of candidate owners/modules.

- It changes the size of the dependency graph.
- It changes how many private reports are generated each round.
- It changes how difficult conjunctive selection becomes.
- It changes the prior owner probability because the same coalition is spread
  across more candidates.
- It can increase padding opportunities and computation cost.

In the current synthetic model, very large `N` can make conjunctive tasks hard
to resolve because the mechanism must identify enough required owners among
many candidates.

### `s`: Random seed

The seed controls reproducibility. With the same seed and parameters, the same
environment, tasks, signals, reports, and outcomes are regenerated.

Change the seed to test whether a result is robust or is caused by one random
draw.

## 3. Sidebar parameters

### General and simulation

| Parameter | Meaning | Main effect |
|---|---|---|
| `num_rounds` (`T`) | Number of task rounds | Controls runtime, averaging quality, and learning horizon |
| `num_agents` (`N`) | Candidate agents/modules | Controls population size, graph size, report count, and selection difficulty |
| `seed` (`s`) | Master random seed | Makes a run reproducible |

### Population and misalignment

| Parameter | Meaning | Main effect |
|---|---|---|
| `agent_regime` | Mixed, deflection, acquisition, or custom population | Determines how private ownership incentives are distributed |
| `deflector_ratio` | Fraction of positive-cost agents | More deflectors usually increase under-reporting and miss loss |
| `acquirer_ratio` | Fraction of negative-cost agents | More acquirers usually increase over-reporting and padding |
| `neutral_ratio` | Fraction of aligned agents | More aligned agents provide a truthful comparison population |
| `deflector_ell` | Private cost of being assigned to a deflector | Larger positive values create stronger downward shading |
| `acquirer_ell` | Private benefit of being assigned to an acquirer | More negative values create stronger upward shading and possible hoarding |

The proposal's signed incentive parameter is represented by `ell`:

- positive `ell`: deflection;
- negative `ell`: acquisition;
- zero `ell`: alignment.

### Dynamic learning

| Parameter | Meaning | Main effect |
|---|---|---|
| `dynamic_mode` | Enables fatigue and online learning | Makes behavior depend on earlier rounds |
| `beta_load` | Sensitivity to recent workload | Larger values make repeated assignment more costly |
| `load_decay` | How quickly workload memory fades | Larger values reduce persistence of old fatigue more quickly |
| `alpha_credit_ema` | Credit moving-average rate | Larger values make expected credit adapt faster |

These parameters are extensions around the proposal's static mechanism. They
are most meaningful with large `T`.

### Task and value structure

| Parameter | Meaning | Main effect |
|---|---|---|
| `rule` | Conjunctive or disjunctive resolution | Conjunctive needs all required owners; disjunctive needs one |
| `coalition_size_k` (`k`) | Fixed required-owner count | Larger coalitions increase coverage requirements |
| `use_coalition_range` | Randomizes coalition size | Makes task difficulty vary between rounds |
| `coalition_min` | Minimum coalition size | Sets the lower task complexity |
| `coalition_max` | Maximum coalition size | Sets the upper task complexity |
| `reward_per_owner` (`R_k`) | Reward multiplier per required owner | Makes reward grow with coalition size |
| `resolution_reward` (`R`) | Flat resolution reward | Used when reward-per-owner mode is disabled |
| `default_cost` (`c_i`) | Social cost per assigned agent | Larger values discourage padding and can make resolution unprofitable |
| `cluster_prob` | Probability that owners cluster in the graph | Larger values make graph evidence more useful |
| `hop_decay` | Embedding contribution decay by graph distance | Smaller values make distant owners less visible in the task signal |
| `min_member_weight` | Minimum contribution of a co-owner | Larger values make all coalition members more observable |
| `task_noise_sigma` | Noise in task embedding construction | Larger values make tasks harder to identify |

The coalition range takes precedence over `coalition_size_k` when enabled.
Therefore, to study a fixed `k`, disable `use_coalition_range`.

### Repository environment and graph

| Parameter | Meaning | Main effect |
|---|---|---|
| `embedding_dim` (`d`) | Profile-vector dimension | Larger dimensions generally separate random profiles more clearly |
| `graph_type` | Scale-free, Erdos-Renyi, complete, or empty | Changes task clustering and audit neighborhoods |
| `avg_degree` | Target graph degree | Changes reachable neighbors and audit detection opportunities |

The graph is used in two places:

1. task owner coalitions can be clustered around graph neighbors;
2. audits can cross-check neighboring payloads.

### Evidence and calibration

| Parameter | Meaning | Main effect |
|---|---|---|
| `obs_noise_sigma` | Noise in private observations | Larger values make `pi_true` less reliable |
| `missing_evidence_prob` (`gamma`) | Probability of no signal | Missing agents fall back to the prior |
| `signal_sensitivity` (`beta_1`) | Converts similarity into probability | Larger values make evidence probabilities more decisive |
| `signal_midpoint_override` | Evidence decision midpoint | Moves the threshold between owner and non-owner evidence |
| `calibration_kappa` (`kappa`) | Probability distortion exponent | `1` calibrated, greater than `1` overconfident, less than `1` underconfident |

Evidence quality affects the information available before strategic behavior
is applied. This is different from misalignment: an honest but poorly
calibrated agent can still be wrong.

### Mechanism M

| Parameter | Meaning | Main effect |
|---|---|---|
| `score_scale` (`lambda`) | Weight of centered Brier score | Larger values usually reduce strategic report shading |
| `audit_rate` (`alpha`) | Chance of auditing each payload | Larger values increase detection and audit overhead |
| `audit_penalty` (`F`) | Penalty after detected tampering | Larger values make tampering less attractive |
| `audit_detection_delta` (`delta`) | Per-neighbor detection probability | Larger values increase contradiction detection |
| `cost_per_audit` (`c_a`) | Cost of each audit | Larger values increase incentive overhead |
| `credit_rule` (`chi_i`) | Equal split or Shapley credit | Changes how coalition rewards are distributed |

Scoring primarily affects reports. Auditing primarily affects payload
tampering. Credit primarily affects assigned agents and is especially
important in coalition tasks.

### Baseline controls

| Parameter | Meaning | Main effect |
|---|---|---|
| `b1_staleness_rho` (`rho`) | Fraction of stale descriptions | Higher values should degrade B1 |
| `b1_desc_noise_sigma` | Noise in description similarity | Makes B1 routing less reliable |
| `b3_peer_noise_sigma` | Noise in peer perception | Makes B3 voting less reliable |

These parameters should primarily change their corresponding baselines, not
Mechanism M.

### Agent internal controls

| Parameter | Meaning | Main effect |
|---|---|---|
| `agent_theta0` | Initial assignment-model intercept | Sets the initial assignment tendency |
| `agent_theta1` | Initial assignment-model slope | Controls how strongly reports affect expected assignment |
| `agent_online_lr` | Online assignment-model learning rate | Controls how quickly behavior adapts from outcomes |
| `agent_evasion_benefit_factor` | Benefit of fabricating evidence | Larger values make tampering more attractive |
| `agent_tamper_deflect_signal` | Fabricated signal used by deflectors | More negative values strengthen ownership denial |
| `agent_tamper_acquire_signal` | Fabricated signal used by acquirers | More positive values strengthen ownership claims |
| `agent_fp_iterations` | Fixed-point iterations for report choice | More iterations can make strategic report calculation more stable but slower |

These parameters affect the agent's decision process rather than the task
generator or the mechanism's selection formula.

## 4. How to interpret the UI outputs

### Gross welfare

Gross welfare is resolution value minus social assignment cost. It excludes
the mechanism's score and audit transfers.

### Net welfare

Net welfare subtracts incentive overhead from gross welfare. A mechanism can
improve assignment quality while still having lower net welfare if incentives
are too expensive.

### Regret

Regret is the gap between the oracle's welfare and the system's welfare.
Lower is better.

### Miss loss

Miss loss measures value lost because required owners were not assigned.
Deflection and poor evidence tend to increase it.

### Excess cost

Excess cost measures assignment cost beyond the oracle's assignment.
Acquisition and padding tend to increase it.

### Report shading

Shading is the difference between the mechanism report and the underlying
belief. Large negative values indicate deflection; large positive values
indicate acquisition.

### Tampering rate

Tampering rate is the fraction of reports whose payload was marked as altered
by the simulator. It is affected by the agent's tampering decision, audit
probability, detection probability, and penalty.

### Coalition coverage and padding

- **Coverage:** fraction of required owners included in the assigned set.
- **Padding:** number of assigned agents who are not required owners.

Coverage is especially important for conjunctive tasks. Padding is especially
important for acquisition incentives and equal-split credit.

## 5. Experiment-card interpretation

The cards in the `Experiment Cards` tab are intentionally small behavioral
tests:

- **No incentives:** should produce more shading and tampering.
- **Strong scoring:** should reduce shading and improve resolution.
- **High deflection:** should increase miss loss.
- **High acquisition:** should increase padding and excess cost.
- **High auditing:** should reduce tampering, but may not change an already
  completed static assignment.
- **Weak penalty:** should make tampering more attractive.
- **Missing evidence:** should reduce information quality.
- **Overconfident evidence:** should increase calibration error and often regret.
- **Coalition-size comparison:** should increase coverage requirements and
  conjunctive difficulty.
- **Disjunctive routing:** should be easier to resolve because one correct
  owner is sufficient.

If a card does not show the expected direction, first increase `T`, repeat
with several seeds, and check whether the changed parameter actually affects
the selected system. For example, B1 staleness should be evaluated against
B1, while audit parameters may primarily change tampering and overhead rather
than static assignment welfare.

## 6. Important implementation caveats

1. The current oracle uses the hidden owner set directly and is therefore an
   optimistic upper benchmark.
2. Static audits happen after selection, so they cannot repair the current
   round's assignment.
3. The repository and issue embeddings are synthetic; the planned real
   issue-to-module data experiment is not yet implemented.
4. The dynamic workload model is an extension and should be interpreted
   separately from the proposal's static analytical model.