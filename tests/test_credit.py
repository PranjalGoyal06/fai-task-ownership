"""Tests for credit rules: Equal Split vs Shapley Value."""

import itertools
import math

import numpy as np
import pytest

from src.mechanisms import AssignmentMechanism
from src.types import CreditRule, MechanismConfig, ResolutionRule, Task


@pytest.fixture
def conjunctive_task():
    costs = np.full(5, 1.0)
    return Task(
        task_id=1,
        t_embedding=np.zeros(5),
        root_module=0,
        S_star={0, 1},
        z=np.array([1, 1, 0, 0, 0]),
        R=12.0,
        costs=costs,
        rule=ResolutionRule.CONJUNCTIVE,
    )


def test_shapley_credit_dummy_player_gets_zero(conjunctive_task):
    """If coalition A = {0, 1, 2} has padded unneeded agent 2, agent 2's Shapley credit must be 0."""
    config = MechanismConfig(credit_rule=CreditRule.SHAPLEY)
    mech = AssignmentMechanism(config)

    assigned_padded = {0, 1, 2}
    credits = mech.divide_credit(conjunctive_task, assigned_padded)

    # Agent 0 and 1 are critical: each gets 6.0
    assert np.isclose(credits[0], 6.0)
    assert np.isclose(credits[1], 6.0)
    # Agent 2 is a dummy player: gets exactly 0.0
    assert np.isclose(credits[2], 0.0)
    # Total sum equals R
    assert np.isclose(sum(credits.values()), conjunctive_task.R)


def test_equal_split_rewards_dummy_player(conjunctive_task):
    """Under Equal Split, padding coalition A with unneeded agent 2 dilutes credit and rewards agent 2."""
    config = MechanismConfig(credit_rule=CreditRule.EQUAL_SPLIT)
    mech = AssignmentMechanism(config)

    assigned_padded = {0, 1, 2}
    credits = mech.divide_credit(conjunctive_task, assigned_padded)

    # Equal split divides 12.0 equally among 3 members: 4.0 each
    assert np.isclose(credits[0], 4.0)
    assert np.isclose(credits[1], 4.0)
    assert np.isclose(credits[2], 4.0)  # Free rider gets 4.0!


def _brute_force_shapley(task: Task, assigned_set: set[int]) -> dict[int, float]:
    N = len(task.costs)
    credits = {i: 0.0 for i in range(N)}
    A = sorted(assigned_set)
    m = len(A)
    if m == 0:
        return credits

    def v(coalition: set[int]) -> float:
        if task.rule == ResolutionRule.CONJUNCTIVE:
            return task.R if task.S_star.issubset(coalition) else 0.0
        return task.R if bool(task.S_star.intersection(coalition)) else 0.0

    for i in A:
        others = [j for j in A if j != i]
        val = 0.0
        for s in range(len(others) + 1):
            weight = (math.factorial(s) * math.factorial(m - s - 1)) / math.factorial(m)
            for subset in itertools.combinations(others, s):
                T = set(subset)
                val += weight * (v(T | {i}) - v(T))
        credits[i] = float(val)
    return credits


@pytest.mark.parametrize("rule", [ResolutionRule.CONJUNCTIVE, ResolutionRule.DISJUNCTIVE])
@pytest.mark.parametrize("assigned_set", [{0, 1}, {0, 1, 2, 4}, {1, 3}, {2, 3, 4}])
def test_closed_form_shapley_matches_combinatorial(rule, assigned_set):
    task = Task(
        task_id=9,
        t_embedding=np.zeros(5),
        root_module=0,
        S_star={0, 1},
        z=np.array([1, 1, 0, 0, 0]),
        R=15.0,
        costs=np.full(5, 1.0),
        rule=rule,
    )
    mech = AssignmentMechanism(MechanismConfig(credit_rule=CreditRule.SHAPLEY))
    fast_credits = mech.divide_credit(task, assigned_set)
    exact_credits = _brute_force_shapley(task, assigned_set)

    for i in range(5):
        assert np.isclose(fast_credits[i], exact_credits[i])


def test_credit_rules_induce_distinct_reporting_under_static_mode():
    """Regression test for RQ3: Shapley vs Equal Split must induce distinct
    equilibrium reporting behavior under static mode (dynamic_mode=False)
    because agents learn expected credit bar_chi_i via credit_ema.
    """
    from src.simulation_runner import SimulationConfig, run_full_simulation

    cfg_shapley = SimulationConfig(
        num_rounds=25,
        num_agents=6,
        credit_rule=CreditRule.SHAPLEY,
        dynamic_mode=False,
        seed=42,
    )
    res_shapley = run_full_simulation(cfg_shapley)

    cfg_equal = SimulationConfig(
        num_rounds=25,
        num_agents=6,
        credit_rule=CreditRule.EQUAL_SPLIT,
        dynamic_mode=False,
        seed=42,
    )
    res_equal = run_full_simulation(cfg_equal)

    shapley_reports = [t["reports_M"] for t in res_shapley.task_history]
    equal_reports = [t["reports_M"] for t in res_equal.task_history]

    mean_r_shapley = np.array([np.mean([r[i] for r in shapley_reports]) for i in range(6)])
    mean_r_equal = np.array([np.mean([r[i] for r in equal_reports]) for i in range(6)])

    assert not np.allclose(mean_r_shapley, mean_r_equal, atol=1e-3), (
        f"Shapley reports ({mean_r_shapley}) and Equal Split reports ({mean_r_equal}) "
        "must diverge under expected credit learning in static mode."
    )
