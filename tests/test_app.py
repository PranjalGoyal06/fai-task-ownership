"""Automated tests for simulation runner and app configuration."""

import pytest
from src.simulation_runner import (
    SimulationConfig,
    SimulationCancelled,
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
    assert len(results.summary_df) == 6
    assert set(results.summary_df["System"]) == {
        "Oracle (B4)",
        "Honest (pi_true)",
        "Description (B1)",
        "Polling Router (B2)",
        "Voting (B3)",
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
    assert len(results.summary_df) == 6


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

