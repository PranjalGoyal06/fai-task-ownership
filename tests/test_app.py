"""Automated tests for simulation runner and app configuration."""

import pytest
import pandas as pd
import time
from app import (
    build_fixed_factor_table,
    build_model_parameter_table,
    mean_pivot,
    research_grid_cache_key,
    start_research_grid_worker,
    summarize_rq1,
    summarize_rq2,
    summarize_rq3,
)
from src.experiment_cards import (
    COMPARISON_CARDS,
    EXPERIMENT_CARDS,
    RESEARCH_QUESTION_CARDS,
    build_experiment_config,
    run_comparison_card,
    run_research_question_card,
)
from src.simulation_runner import (
    SimulationConfig,
    SimulationCancelled,
    _benchmark_worker_count,
    run_full_simulation,
    run_parameter_grid_search,
    run_parameter_sweep,
)
from src.types import CreditRule, ResolutionRule


def test_simulation_runner_default_static():
    """Verify that default static benchmark simulation runs and yields complete outputs."""
    config = SimulationConfig(num_rounds=15, num_agents=8, seed=42)
    results = run_full_simulation(config)

    assert results is not None
    assert len(results.summary_df) == 4
    assert set(results.summary_df["System"]) == {
        "Oracle (B4)",
        "Description (B1)",
        "Polling Router (B2)",
        "Mechanism M",
    }
    assert len(results.task_history) == 15
    assert len(results.agents_df) == 8

    # Oracle must have 0 regret and non-negative welfare
    oracle_row = results.summary_df[results.summary_df["System"] == "Oracle (B4)"].iloc[0]
    assert oracle_row["Regret"] == 0.0
    assert oracle_row["Welfare (Gross)"] >= 0.0

    # Mechanism M must have calculated metrics
    mech_row = results.summary_df[results.summary_df["System"] == "Mechanism M"].iloc[0]
    assert "Regret" in mech_row
    assert "Net Welfare" in mech_row
    assert "Incentive Overhead" in mech_row


def test_simulation_runner_dynamic_mode():
    """Verify that Mode 2 dynamic mode tracks fatigue and updates credit EMA."""
    config = SimulationConfig(
        num_rounds=20,
        num_agents=6,
        dynamic_mode=True,
        beta_load=0.8,
        load_decay=0.5,
        alpha_credit_ema=0.2,
        seed=101,
    )
    results = run_full_simulation(config)

    assert results.config.dynamic_mode is True
    assert "Recent Fatigue Load" in results.agents_df.columns
    assert "Total Credits M" in results.agents_df.columns


def test_simulation_runner_disjunctive_rule():
    """Verify simulation runner with Disjunctive resolution rule and Equal Split credit."""
    config = SimulationConfig(
        num_rounds=10,
        num_agents=6,
        rule=ResolutionRule.DISJUNCTIVE,
        credit_rule=CreditRule.EQUAL_SPLIT,
        seed=77,
    )
    results = run_full_simulation(config)
    assert len(results.summary_df) == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_agents": 101}, "num_agents"),
        ({"deflector_ratio": 0.7, "acquirer_ratio": 0.5}, "deflector_ratio"),
        ({"cost_per_audit": -0.1}, "cost_per_audit"),
        ({"task_noise_sigma": -0.2}, "task_noise_sigma"),
    ],
)
def test_simulation_config_rejects_invalid_domains(kwargs, message):
    with pytest.raises(ValueError, match=message):
        config_kwargs = {"num_rounds": 1, "num_agents": 4}
        config_kwargs.update(kwargs)
        SimulationConfig(**config_kwargs)


def test_parameter_sweep_runner():
    """Verify that 1D sensitivity parameter sweep produces correct DataFrame."""
    base_cfg = SimulationConfig(num_rounds=10, num_agents=6, seed=42)
    df_sweep = run_parameter_sweep("score_scale", [0.0, 2.5], base_cfg)

    assert len(df_sweep) == 2
    assert set(df_sweep["Runs"]) == {1}
    assert "score_scale" in df_sweep.columns
    assert "Mechanism Welfare" in df_sweep.columns
    assert "Mechanism Regret" in df_sweep.columns
    assert "Mechanism Welfare Std" in df_sweep.columns


def test_parameter_sweep_averages_replicates_reproducibly():
    """Verify repeated runs produce reproducible means and between-run spread."""
    base_cfg = SimulationConfig(num_rounds=4, num_agents=4, seed=42)
    first = run_parameter_sweep(
        "score_scale",
        [0.0, 2.5],
        base_cfg,
        num_replicates=3,
    )
    second = run_parameter_sweep(
        "score_scale",
        [0.0, 2.5],
        base_cfg,
        num_replicates=3,
    )

    assert list(first["Runs"]) == [3, 3]
    assert first.equals(second)
    assert (first["Mechanism Welfare Std"] >= 0).all()


def test_parameter_sweep_rejects_invalid_replicate_count():
    """Verify a sweep cannot silently run without any replicates."""
    base_cfg = SimulationConfig(num_rounds=1, num_agents=3)

    with pytest.raises(ValueError, match="num_replicates"):
        run_parameter_sweep("score_scale", [1.0], base_cfg, num_replicates=0)


def test_benchmark_worker_count_is_machine_scaled():
    """Verify benchmark workers scale to available CPUs and never exceed eight."""
    worker_count = _benchmark_worker_count()

    assert 1 <= worker_count <= 8


def test_research_heatmap_averages_duplicate_nuisance_factor_rows():
    """A full factorial grid must reshape after averaging nuisance-factor duplicates."""
    grid = pd.DataFrame(
        [
            {"ell": 1.0, "phi": 0.25, "metric": 2.0},
            {"ell": 1.0, "phi": 0.25, "metric": 4.0},
            {"ell": 2.0, "phi": 0.25, "metric": 6.0},
            {"ell": 2.0, "phi": 0.50, "metric": 8.0},
        ]
    )

    pivot = mean_pivot(grid, "ell", "phi", "metric")

    assert pivot.loc[1.0, 0.25] == 3.0
    assert pivot.loc[2.0, 0.25] == 6.0
    assert pivot.loc[2.0, 0.50] == 8.0


def test_research_conclusion_summaries_are_data_derived():
    rq1 = pd.DataFrame(
        [{"deflector_ratio": 0.0, "deflector_ell": 1.0, "B2 Regret": 1.0, "B2 Miss Loss": 2.0, "B2 Excess Cost": 3.0},
         {"deflector_ratio": 0.5, "deflector_ell": 2.0, "B2 Regret": 4.0, "B2 Miss Loss": 5.0, "B2 Excess Cost": 6.0}]
    )
    rq2 = pd.DataFrame(
        [{"score_scale": 1.0, "audit_rate": 0.2, "Regret Recovery Fraction": 0.4,
          "Incentive Overhead": 0.3, "Tampering Rate": 0.1, "Mean Absolute Shading M": 0.2}]
    )
    rq3 = pd.DataFrame(
        [{"rule": "conjunctive", "credit_rule": "shapley", "Mechanism Coalition Coverage": 0.8,
          "Mechanism Coalition Padding": 0.2, "Required-Owner Underreporting": 0.1,
          "Non-Required-Owner Overreporting": 0.1},
         {"rule": "conjunctive", "credit_rule": "equal_split", "Mechanism Coalition Coverage": 0.7,
          "Mechanism Coalition Padding": 0.3, "Required-Owner Underreporting": 0.2,
          "Non-Required-Owner Overreporting": 0.2}]
    )

    assert len(summarize_rq1(rq1)) == 2
    assert "λ=1.0" in summarize_rq2(rq2)[0]
    assert len(summarize_rq3(rq3)) == 1


def test_fixed_factor_table_is_arrow_serializable():
    """Fixed-factor display values must remain uniformly typed for Streamlit/Arrow."""
    base = SimulationConfig(num_rounds=2, num_agents=4, seed=42)

    table = build_fixed_factor_table(base, "rq1_misalignment")

    assert table["Factor"].map(type).eq(str).all()
    assert table["Treatment"].map(type).eq(str).all()


def test_model_parameter_inventory_lists_every_simulation_config_field():
    base = SimulationConfig(num_rounds=2, num_agents=4, seed=42)
    inventory = build_model_parameter_table(base)

    assert set(inventory["Parameter"]) == set(SimulationConfig.__dataclass_fields__)


def test_rq_primary_grid_sizes_keep_nuisance_parameters_fixed():
    rq1 = next(card for card in RESEARCH_QUESTION_CARDS if card.key == "rq1_misalignment")
    rq2 = next(card for card in RESEARCH_QUESTION_CARDS if card.key == "rq2_incentive_recovery")
    rq3 = next(card for card in RESEARCH_QUESTION_CARDS if card.key == "rq3_coalition_credit")

    rq1_product = 10 * 10
    rq2_product = len(rq2.parameter_grid["score_scale"]) * len(rq2.parameter_grid["audit_rate"])
    rq3_product = (
        len(rq3.parameter_grid["credit_rule"])
        * len(rq3.parameter_grid["rule"])
        * len(rq3.parameter_grid["coalition_size_k"])
    )

    for card in (rq2, rq3):
        for name, values in card.parameter_grid.items():
            if name not in (
                "score_scale",
                "audit_rate",
                "credit_rule",
                "rule",
                "coalition_size_k",
            ):
                assert len(values) == 1

    assert rq1_product == 100
    assert rq2_product == 100
    assert rq3_product == 40


def test_research_grid_cache_key_is_exact_and_repeatable():
    base = SimulationConfig(num_rounds=60, num_agents=12, seed=42)
    grid = {"deflector_ratio": [0.0, 0.5], "deflector_ell": [1.0, 3.0]}

    first = research_grid_cache_key("rq1_misalignment", base, 42, 5, grid)
    second = research_grid_cache_key("rq1_misalignment", base, 42, 5, grid)
    changed_seed = research_grid_cache_key("rq1_misalignment", base, 43, 5, grid)

    assert first == second
    assert first != changed_seed


def test_research_grid_worker_returns_renderable_result():
    base = SimulationConfig(num_rounds=2, num_agents=4, seed=42)
    card = next(card for card in RESEARCH_QUESTION_CARDS if card.key == "rq1_misalignment")

    status = start_research_grid_worker(
        card,
        base,
        42,
        1,
        {"deflector_ratio": [0.0, 0.5], "deflector_ell": [0.0, 1.0]},
        "smoke-rq1",
    )
    deadline = time.monotonic() + 10
    while status["running"] and time.monotonic() < deadline:
        time.sleep(0.02)

    assert status["running"] is False
    assert status["error"] is None
    assert isinstance(status["result"], pd.DataFrame)
    assert len(status["result"]) == 4
    assert "B2 Regret" in status["result"]


def test_parameter_grid_search_averages_metrics_across_combinations():
    """Verify a multi-parameter grid varies only selected fields and returns RQ metrics."""
    base_cfg = SimulationConfig(num_rounds=3, num_agents=4, seed=42, audit_rate=0.25)
    grid = run_parameter_grid_search(
        {"score_scale": [0.0, 2.5], "audit_rate": [0.0, 0.5]},
        base_cfg,
        num_replicates=2,
    )

    assert len(grid) == 4
    assert set(grid["Runs"]) == {2}
    assert set(grid["audit_rate"]) == {0.0, 0.5}
    assert set(grid["score_scale"]) == {0.0, 2.5}
    assert "B2 Regret" in grid
    assert "Regret Recovery Fraction" in grid
    assert "Mechanism ECE" in grid
    assert "Mechanism Coalition Coverage" in grid
    assert "Mechanism Welfare Std" in grid


def test_parameter_grid_search_rejects_unknown_parameters():
    """Verify invalid grid names fail explicitly instead of being silently ignored."""
    with pytest.raises(ValueError, match="Unknown simulation parameter"):
        run_parameter_grid_search({"not_a_parameter": [1]}, SimulationConfig(num_rounds=1))


def test_parameter_grid_search_can_be_cancelled_between_runs():
    """Verify a cancellation request stops before starting the next run."""
    calls = 0

    def cancel_after_first_run() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    with pytest.raises(SimulationCancelled, match="cancelled"):
        run_parameter_grid_search(
            {"score_scale": [0.0, 2.5]},
            SimulationConfig(num_rounds=2, num_agents=4),
            num_replicates=1,
            cancel_check=cancel_after_first_run,
        )


def test_experiment_cards_use_known_parameters_and_override_base_config():
    base = SimulationConfig(num_rounds=2, num_agents=6, seed=42)
    card = next(card for card in EXPERIMENT_CARDS if card.key == "small_population")
    config = build_experiment_config(base, card.overrides)

    assert config.num_agents == 4
    assert config.num_rounds == base.num_rounds
    assert base.num_agents == 6


def test_comparison_card_is_reproducible():
    base = SimulationConfig(num_rounds=2, num_agents=4, seed=42)
    card = next(card for card in COMPARISON_CARDS if card.parameter == "audit_rate")

    first = run_comparison_card(card, base)
    second = run_comparison_card(card, base)

    assert first.equals(second)


def test_disjunctive_runs_are_reproducible_with_equal_seed():
    config = SimulationConfig(
        num_rounds=2,
        num_agents=12,
        seed=42,
        rule=ResolutionRule.DISJUNCTIVE,
        credit_rule=CreditRule.SHAPLEY,
        use_coalition_range=False,
        coalition_size_k=1,
    )

    first = run_full_simulation(config)
    second = run_full_simulation(config)

    assert first.summary_df.equals(second.summary_df)


def test_research_question_cards_are_reproducible_and_cover_all_metrics():
    base = SimulationConfig(num_rounds=2, num_agents=4, seed=42)

    for card in RESEARCH_QUESTION_CARDS:
        if card.key == "rq1_misalignment":
            smoke_grid = {
                "deflector_ratio": [0.0, 0.5],
                "deflector_ell": [0.0, 3.0],
            }
        elif card.key == "rq2_incentive_recovery":
            smoke_grid = {
                "score_scale": [0.0, 2.5],
                "audit_rate": [0.0, 0.5],
                "deflector_ratio": [0.1, 0.5],
                "deflector_ell": [1.0, 3.0],
                "acquirer_ratio": [0.1, 0.3],
            }
        else:
            smoke_grid = {
                "credit_rule": [CreditRule.SHAPLEY, CreditRule.EQUAL_SPLIT],
                "rule": [ResolutionRule.CONJUNCTIVE, ResolutionRule.DISJUNCTIVE],
                "coalition_size_k": [1, 2],
            }
        first = run_research_question_card(
            card,
            base,
            num_rounds=2,
            num_replicates=2,
            parameter_grid=smoke_grid,
        )
        second = run_research_question_card(
            card,
            base,
            num_rounds=2,
            num_replicates=2,
            parameter_grid=smoke_grid,
        )
        assert first.equals(second)
        assert set(first["Runs"]) == {2}

    rq1 = next(card for card in RESEARCH_QUESTION_CARDS if card.key == "rq1_misalignment")
    rq1_results = run_research_question_card(
        rq1,
        base,
        num_rounds=2,
        num_replicates=2,
        parameter_grid={
            "deflector_ratio": [0.0, 0.5],
            "deflector_ell": [0.0, 3.0],
        },
    )
    assert "B2 Miss Loss" in rq1_results
    assert "B2 Excess Cost" in rq1_results

    rq3 = next(card for card in RESEARCH_QUESTION_CARDS if card.key == "rq3_coalition_credit")
    rq3_results = run_research_question_card(
        rq3,
        base,
        num_rounds=2,
        num_replicates=2,
        parameter_grid={
            "credit_rule": [CreditRule.SHAPLEY, CreditRule.EQUAL_SPLIT],
            "rule": [ResolutionRule.CONJUNCTIVE, ResolutionRule.DISJUNCTIVE],
            "coalition_size_k": [1, 2],
        },
    )
    assert "Required-Owner Underreporting" in rq3_results
    assert "Non-Required-Owner Overreporting" in rq3_results

