# Test Description

The repository has **57 pytest test items** across 8 test modules. The items cover configuration, simulation, baselines, credit, environment generation, strategic learning, mechanisms, metrics, and scoring.

## `tests/test_app.py` - 15 items

| Test item | Purpose |
|---|---|
| `test_simulation_runner_default_static` | Runs a static 15-round simulation and checks benchmark systems, task history, agent rows, Oracle properties, and Mechanism metrics. |
| `test_simulation_runner_dynamic_mode` | Runs dynamic Mode 2 and verifies fatigue and credit EMA fields. |
| `test_simulation_runner_disjunctive_rule` | Runs disjunctive resolution with Equal-Split credit and checks the summary. |
| `test_simulation_config_rejects_invalid_domains[kwargs0-num_agents]` | Rejects `num_agents=101`. |
| `test_simulation_config_rejects_invalid_domains[kwargs1-deflector_ratio]` | Rejects deflector/acquirer fractions whose sum exceeds 1. |
| `test_simulation_config_rejects_invalid_domains[kwargs2-cost_per_audit]` | Rejects a negative audit cost. |
| `test_simulation_config_rejects_invalid_domains[kwargs3-task_noise_sigma]` | Rejects negative task embedding noise. |
| `test_parameter_sweep_runner` | Checks one-dimensional score-scale sweeps and required output columns. |
| `test_parameter_sweep_averages_replicates_reproducibly` | Checks repeated-replicate sweep averages and reproducibility. |
| `test_parameter_sweep_rejects_invalid_replicate_count` | Rejects zero replicates. |
| `test_parameter_grid_search_averages_metrics_across_combinations` | Checks 2D parameter grids, replicate counts, and research metrics. |
| `test_parameter_grid_search_rejects_unknown_parameters` | Rejects unknown grid parameter names. |
| `test_parameter_grid_search_can_be_cancelled_between_runs` | Verifies grid cancellation. |
| `test_experiment_cards_use_known_parameters_and_override_base_config` | Checks experiment-card overrides without mutating the base config. |
| `test_comparison_card_is_reproducible` | Checks deterministic comparison-card output. |

## `tests/test_baselines.py` - 2 items

| Test item | Purpose |
|---|---|
| `test_description_router_degrades_with_staleness` | Verifies fresh descriptions outperform fully stale descriptions over 20 tasks. |
| `test_voting_decider_returns_plurality_winner_without_report_leakage` | Checks plurality selection and that peer voting is not derived from private reports. |

## `tests/test_credit.py` - 11 items

| Test item | Purpose |
|---|---|
| `test_shapley_credit_dummy_player_gets_zero` | Gives no Shapley credit to an unneeded dummy agent. |
| `test_equal_split_rewards_dummy_player` | Equal Split distributes credit, including to an unneeded dummy. |
| `test_closed_form_shapley_matches_combinatorial[assigned_set0-ResolutionRule.CONJUNCTIVE]` | Matches closed-form Shapley credit with brute-force credit under conjunctive resolution. |
| `test_closed_form_shapley_matches_combinatorial[assigned_set0-ResolutionRule.DISJUNCTIVE]` | Matches closed-form Shapley credit with brute-force credit under disjunctive resolution. |
| `test_closed_form_shapley_matches_combinatorial[assigned_set1-ResolutionRule.CONJUNCTIVE]` | Tests combinatorial Shapley correctness for assigned set 1 under conjunctive resolution. |
| `test_closed_form_shapley_matches_combinatorial[assigned_set1-ResolutionRule.DISJUNCTIVE]` | Tests combinatorial Shapley correctness for assigned set 1 under disjunctive resolution. |
| `test_closed_form_shapley_matches_combinatorial[assigned_set2-ResolutionRule.CONJUNCTIVE]` | Tests combinatorial Shapley correctness for assigned set 2 under conjunctive resolution. |
| `test_closed_form_shapley_matches_combinatorial[assigned_set2-ResolutionRule.DISJUNCTIVE]` | Tests combinatorial Shapley correctness for assigned set 2 under disjunctive resolution. |
| `test_closed_form_shapley_matches_combinatorial[assigned_set3-ResolutionRule.CONJUNCTIVE]` | Tests combinatorial Shapley correctness for assigned set 3 under conjunctive resolution. |
| `test_closed_form_shapley_matches_combinatorial[assigned_set3-ResolutionRule.DISJUNCTIVE]` | Tests combinatorial Shapley correctness for assigned set 3 under disjunctive resolution. |
| `test_credit_rules_induce_distinct_reporting_under_static_mode` | Verifies Shapley and Equal-Split produce different learned reports in static mode. |

## `tests/test_environment.py` - 5 items

| Test item | Purpose |
|---|---|
| `test_repository_environment_initialization` | Checks profile dimensions, unit-normalized vectors, nodes, and graph edges. |
| `test_task_generator_weighted_embedding` | Checks coalition size, root membership, task unit norm, and root similarity. |
| `test_task_generator_random_coalition_and_scaled_reward` | Checks random coalition bounds, rewards, and multiple coalition sizes. |
| `test_evidence_engine_beliefs_and_miscalibration` | Checks calibrated probabilities and overconfident belief transformation. |
| `test_evidence_engine_calibration_ece` | Checks calibrated ECE and that stronger overconfidence increases ECE. |

## `tests/test_learning.py` - 10 items

| Test item | Purpose |
|---|---|
| `test_aligned_agent_is_truthful` | Aligned agents report their calibrated belief without tampering. |
| `test_deflector_shades_down` | Deflectors report below their true belief. |
| `test_acquirer_shades_up` | Acquirers report above their true belief. |
| `test_score_scale_disciplines_shading` | Stronger scoring reduces strategic shading. |
| `test_no_scoring_extreme_reports` | Without scoring, deflectors report 0 and acquirers report 1. |
| `test_audit_penalty_deters_tampering` | High audit rate and penalty deter tampering. |
| `test_missing_evidence_cannot_tamper` | Missing evidence cannot be tampered or falsely reported. |
| `test_tampering_alters_claimed_signal` | Tampering lowers deflector claims and raises acquirer claims. |
| `test_online_learning_assignment_curve` | Assignment-history learning produces a bounded positive slope. |
| `test_dynamic_workload_stays_bounded_and_tracks_credit` | Recent workload reaches the expected cap, credit EMA converges, and workload decays. |

## `tests/test_mechanisms.py` - 6 items

| Test item | Purpose |
|---|---|
| `test_escalation_rule_stops_at_first_required` | Disjunctive escalation stops after assigning the first true owner. |
| `test_conjunctive_set_selection_balances_value_and_costs` | Conjunctive selection assigns required agents while avoiding low-report agents. |
| `test_graph_audits_detection_frequency` | Checks empirical graph-audit contradiction frequency against theory. |
| `test_escalation_skips_zero_reports` | Zero-report agents are not assigned. |
| `test_escalation_tiebreak_is_randomized` | Equal reports produce randomized tie-breaking. |
| `test_vectorized_conjunctive_matches_combinatorial_search` | Vectorized conjunctive selection matches brute-force optimal selection. |

## `tests/test_metrics.py` - 5 items

| Test item | Purpose |
|---|---|
| `test_oracle_optimality_and_regret_nonnegativity` | Checks conjunctive Oracle optimality, welfare, and regret decomposition. |
| `test_disjunctive_oracle` | Checks disjunctive Oracle chooses the lowest-cost owner. |
| `test_unprofitable_task_efficiency_and_empty_s_star` | Checks unprofitable tasks, efficiency, and empty-owner handling. |
| `test_truthful_benchmark_and_net_welfare` | Checks honest welfare, information/strategic gaps, and net welfare. |
| `test_calibration_and_incentive_summary` | Checks ECE, mean Brier score, tampering rate, credit share, and principal spend. |

## `tests/test_scoring.py` - 3 items

| Test item | Purpose |
|---|---|
| `test_brier_score_expected_value_strictly_proper` | The Brier score expected value is maximized by reporting the true probability. |
| `test_centered_brier_score_zero_expectation_at_prior` | Reporting the prior has zero expected centered-Brier score. |
| `test_compute_all_scores_accepts_vector_priors` | Multi-agent score calculation supports heterogeneous vector priors. |
