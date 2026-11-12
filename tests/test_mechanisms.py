"""Tests for selection rules and graph audits."""

import itertools

import networkx as nx
import numpy as np
import pytest

from src.mechanisms import AssignmentMechanism
from src.types import (
    EvidencePayload,
    MechanismConfig,
    PublicReport,
    ResolutionRule,
    Task,
)


def test_escalation_rule_stops_at_first_required():
    costs = np.array([1.0, 1.0, 1.0, 1.0])
    task = Task(
        task_id=1,
        t_embedding=np.zeros(4),
        root_module=2,
        S_star={2},
        z=np.array([0, 0, 1, 0]),
        R=10.0,
        costs=costs,
        rule=ResolutionRule.DISJUNCTIVE,
    )

    # Agents 0, 1, 2, 3 have reports [0.9, 0.4, 0.8, 0.1]
    # Sorted order of r/c: Agent 0 (0.9), Agent 2 (0.8), Agent 1 (0.4), Agent 3 (0.1)
    # Agent 2 is in S_star, so escalation should stop after assigning Agent 0 and Agent 2!
    reports = {
        0: PublicReport(0, 0.9, EvidencePayload(0.9)),
        1: PublicReport(1, 0.4, EvidencePayload(0.4)),
        2: PublicReport(2, 0.8, EvidencePayload(0.8)),
        3: PublicReport(3, 0.1, EvidencePayload(0.1)),
    }

    mech = AssignmentMechanism(MechanismConfig())
    assigned = mech.select_agents(task, reports)
    assert assigned == {0, 2}


def test_conjunctive_set_selection_balances_value_and_costs():
    costs = np.array([1.0, 1.0, 1.0, 1.0])
    task = Task(
        task_id=2,
        t_embedding=np.zeros(4),
        root_module=0,
        S_star={0, 1},
        z=np.array([1, 1, 0, 0]),
        R=20.0,
        costs=costs,
        rule=ResolutionRule.CONJUNCTIVE,
    )

    # Agent 0 and 1 have high reports (0.95), while Agent 2 and 3 have near 0 reports (0.01)
    reports = {
        0: PublicReport(0, 0.95, EvidencePayload(0.95)),
        1: PublicReport(1, 0.95, EvidencePayload(0.95)),
        2: PublicReport(2, 0.01, EvidencePayload(0.01)),
        3: PublicReport(3, 0.01, EvidencePayload(0.01)),
    }

    mech = AssignmentMechanism(MechanismConfig())
    assigned = mech.select_agents(task, reports)
    assert assigned == {0, 1}, f"Expected { {0, 1} }, got {assigned}"


def test_graph_audits_detection_frequency():
    G = nx.star_graph(5)  # Center node 0 with 5 leaves (nodes 1..5)
    config = MechanismConfig(audit_rate=1.0, audit_detection_delta=0.4, audit_penalty=10.0, cost_per_audit=0.5)
    mech = AssignmentMechanism(config)

    # Center node 0 (degree 5) tampered with its payload
    reports = {
        0: PublicReport(0, 0.9, EvidencePayload(0.9, is_tampered_ground_truth=True)),
        1: PublicReport(1, 0.1, EvidencePayload(0.1, is_tampered_ground_truth=False)),
    }

    rng = np.random.default_rng(123)
    contradiction_count = 0
    runs = 1000
    for _ in range(runs):
        penalties, audited, contradicted, cost = mech.execute_audits(reports, G, rng)
        if 0 in contradicted:
            contradiction_count += 1
        assert 1 not in contradicted, "Untampered agent must never be contradicted"

    empirical_q0 = contradiction_count / runs
    # Theoretical q0 = 1.0 * [1 - (1 - 0.4)^5] = 1 - 0.6^5 = 1 - 0.07776 = 0.92224
    assert np.isclose(empirical_q0, 0.922, atol=0.03)


def test_escalation_tiebreak_is_randomized():
    costs = np.ones(6)
    # Only agent 5 is the true owner; all reports are tied at 0.0
    task = Task(
        task_id=3,
        t_embedding=np.zeros(6),
        root_module=5,
        S_star={5},
        z=np.array([0, 0, 0, 0, 0, 1]),
        R=10.0,
        costs=costs,
        rule=ResolutionRule.DISJUNCTIVE,
    )
    reports = {i: PublicReport(i, 0.0, EvidencePayload(0.0)) for i in range(6)}
    mech = AssignmentMechanism(MechanismConfig())

    assigned_sizes = {
        len(mech.select_agents(task, reports, rng=np.random.default_rng(seed)))
        for seed in range(25)
    }
    # Deterministic index tie-breaking would always visit 0..5 -> size 6 every time.
    assert len(assigned_sizes) > 1


def test_vectorized_conjunctive_matches_combinatorial_search():
    rng = np.random.default_rng(42)
    N = 6
    costs = rng.uniform(0.5, 2.5, size=N)
    r_vals = rng.uniform(0.05, 0.95, size=N)
    task = Task(
        task_id=4,
        t_embedding=np.zeros(N),
        root_module=0,
        S_star={0, 2},
        z=np.array([1, 0, 1, 0, 0, 0]),
        R=18.0,
        costs=costs,
        rule=ResolutionRule.CONJUNCTIVE,
    )
    reports = {i: PublicReport(i, float(r_vals[i]), EvidencePayload(float(r_vals[i]))) for i in range(N)}
    mech = AssignmentMechanism(MechanismConfig())

    assigned = mech.select_agents(task, reports)

    # Brute-force objective check
    best_obj = -float("inf")
    best_set = set()
    for k in range(N + 1):
        for subset in itertools.combinations(range(N), k):
            A = set(subset)
            prob = 1.0
            for j in range(N):
                if j not in A:
                    prob *= (1.0 - r_vals[j])
            obj = task.R * prob - sum(costs[i] for i in A)
            if obj > best_obj:
                best_obj = obj
                best_set = A

    assigned_prob = 1.0
    for j in range(N):
        if j not in assigned:
            assigned_prob *= (1.0 - r_vals[j])
    assigned_obj = task.R * assigned_prob - sum(costs[i] for i in assigned)

    assert np.isclose(assigned_obj, best_obj)
    assert assigned == best_set
