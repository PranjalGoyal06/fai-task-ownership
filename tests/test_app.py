"""Automated tests for simulation runner and app configuration."""

import pytest
from src.simulation_runner import SimulationConfig, run_full_simulation, run_parameter_sweep
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
    assert "score_scale" in df_sweep.columns
    assert "Mechanism Welfare" in df_sweep.columns
    assert "Mechanism Regret" in df_sweep.columns

