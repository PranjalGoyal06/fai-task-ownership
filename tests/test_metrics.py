"""Tests for metrics, oracle optimality, and regret decomposition."""

import numpy as np
import pytest

from src.baselines import OracleBaselineB4
from src.metrics import MetricsEvaluator
from src.types import Evidence, EvidencePayload, PublicPayload, PublicReport, ResolutionRule, Task


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


def test_unprofitable_task_efficiency_and_empty_s_star():
    # Unprofitable task: R=1.0 < cost=2.0 -> oracle abstains (oracle_welfare=0.0)
    unprofitable = Task(
        task_id=3,
        t_embedding=np.zeros(2),
        root_module=0,
        S_star={0},
        z=np.array([1, 0]),
        R=1.0,
        costs=np.array([2.0, 2.0]),
        rule=ResolutionRule.CONJUNCTIVE,
    )
    res = MetricsEvaluator.compute_regret_decomposition(unprofitable, set())
    assert np.isclose(res["oracle_welfare"], 0.0)
    assert np.isclose(res["efficiency_ratio"], 1.0)

    empty_task = Task(
        task_id=4,
        t_embedding=np.zeros(2),
        root_module=0,
        S_star=set(),
        z=np.array([0, 0]),
        R=10.0,
        costs=np.array([1.0, 1.0]),
        rule=ResolutionRule.CONJUNCTIVE,
    )
    assert MetricsEvaluator.evaluate_assignment(empty_task, {0})[0] is False
    assert MetricsEvaluator.compute_oracle(empty_task) == (set(), False, 0.0, 0.0)


def test_truthful_benchmark_and_net_welfare(sample_task):
    # Honest evidence clearly identifies owners {0, 1}
    evidence = {
        0: Evidence(0, 0.8, 0.95, 0.95, EvidencePayload(0.8)),
        1: Evidence(1, 0.7, 0.90, 0.90, EvidencePayload(0.7)),
        2: Evidence(2, -0.5, 0.05, 0.05, EvidencePayload(-0.5)),
        3: Evidence(3, -0.6, 0.05, 0.05, EvidencePayload(-0.6)),
    }
    res = MetricsEvaluator.compute_regret_decomposition(
        sample_task,
        assigned_set={0, 1, 2},  # includes dummy agent 2 (cost 2.0)
        audit_cost=0.4,
        evidence_dict=evidence,
    )
    assert np.isclose(res["honest_welfare"], 12.5)
    assert np.isclose(res["information_gap"], 0.0)
    assert np.isclose(res["strategic_gap"], 2.0)
    assert np.isclose(res["information_gap"] + res["strategic_gap"], res["regret"])
    assert np.isclose(res["net_welfare"], res["social_welfare"] - 0.4)


def test_calibration_and_incentive_summary(sample_task):
    ece = MetricsEvaluator.compute_ece([0.9, 0.1, 0.9, 0.1], [1, 0, 1, 0], n_bins=5)
    assert np.isclose(ece, 0.1)

    brier = MetricsEvaluator.compute_mean_brier([1.0, 0.0], [1, 0])
    assert np.isclose(brier, 1.0)

    reports = {
        0: PublicReport(0, 0.9, PublicPayload(0.9)),
        1: PublicReport(1, 0.9, PublicPayload(0.9)),
        2: PublicReport(2, 0.8, PublicPayload(0.8)),
        3: PublicReport(3, 0.1, PublicPayload(0.1)),
    }
    tampered_flags = {0: False, 1: False, 2: True, 3: False}
    summary = MetricsEvaluator.compute_incentive_summary(
        task=sample_task,
        reports=reports,
        scores={0: 0.5, 1: 0.5, 2: -0.2, 3: 0.1},
        credits={0: 5.0, 1: 5.0, 2: 5.0, 3: 0.0},
        penalties={0: 0.0, 1: 0.0, 2: 5.0, 3: 0.0},
        audit_cost=0.2,
        tampered_flags=tampered_flags,
    )
    assert np.isclose(summary["tampering_rate"], 0.25)
    assert np.isclose(summary["owner_credit_share"], 10.0 / 15.0)
    assert np.isclose(summary["net_principal_spend"], 0.9 + 15.0 - 5.0 + 0.2)
