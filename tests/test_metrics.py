"""Tests for metrics, oracle optimality, and regret decomposition."""

import numpy as np
import pytest

from src.baselines import OracleBaselineB4
from src.metrics import MetricsEvaluator
from src.types import ResolutionRule, Task


@pytest.fixture
def sample_task():
    costs = np.array([1.0, 1.5, 2.0, 1.0])
    return Task(
        task_id=1,
        t_embedding=np.zeros(4),
        root_module=0,
        S_star={0, 1},
        z=np.array([1, 1, 0, 0]),
        R=15.0,
        costs=costs,
        rule=ResolutionRule.CONJUNCTIVE,
    )


def test_oracle_optimality_and_regret_nonnegativity(sample_task):
    A_oracle = OracleBaselineB4.select(sample_task)
    assert A_oracle == {0, 1}

    # Oracle assigns {0, 1}: cost is 1.0 + 1.5 = 2.5, welfare is 15 - 2.5 = 12.5
    _, _, oracle_welfare = MetricsEvaluator.evaluate_assignment(sample_task, A_oracle)
    assert np.isclose(oracle_welfare, 12.5)

    # Test arbitrary candidate subsets: none should exceed oracle welfare
    for candidate in [{0}, {1}, {2}, {0, 2}, {0, 1, 2}, set()]:
        _, _, w = MetricsEvaluator.evaluate_assignment(sample_task, candidate)
        assert w <= oracle_welfare + 1e-6
        metrics = MetricsEvaluator.compute_regret_decomposition(sample_task, candidate)
        assert metrics["regret"] >= -1e-6
        assert np.isclose(metrics["regret"], metrics["miss_loss"] + metrics["excess_cost"])


def test_disjunctive_oracle():
    costs = np.array([3.0, 1.0, 2.0])
    task = Task(
        task_id=2,
        t_embedding=np.zeros(3),
        root_module=0,
        S_star={0, 1},  # either 0 (cost 3.0) or 1 (cost 1.0)
        z=np.array([1, 1, 0]),
        R=10.0,
        costs=costs,
        rule=ResolutionRule.DISJUNCTIVE,
    )

    A_oracle = OracleBaselineB4.select(task)
    # Oracle should pick lowest-cost owner: agent 1
    assert A_oracle == {1}
    _, cost, welfare = MetricsEvaluator.evaluate_assignment(task, A_oracle)
    assert np.isclose(cost, 1.0)
    assert np.isclose(welfare, 9.0)
