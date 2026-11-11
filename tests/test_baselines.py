"""Tests for comparison baselines B1-B4."""

import numpy as np
import pytest

from src.baselines import DescriptionRouterB1, PollingRouterB2, VotingDeciderB3
from src.environment import RepositoryEnvironment, TaskGenerator
from src.types import EvidencePayload, PublicReport, ResolutionRule


def test_description_router_degrades_with_staleness():
    env = RepositoryEnvironment(num_agents=8, embedding_dim=16, seed=42)
    gen = TaskGenerator(env, coalition_size_k=2, rule=ResolutionRule.CONJUNCTIVE, seed=42)

    b1_fresh = DescriptionRouterB1(env.profiles, staleness_rho=0.0, seed=42)
    b1_stale = DescriptionRouterB1(env.profiles, staleness_rho=1.0, seed=42)

    # Fresh descriptions should match env.profiles identically
    np.testing.assert_allclose(b1_fresh.stale_profiles, env.profiles)

    # Fully stale descriptions should differ from env.profiles
    assert not np.allclose(b1_stale.stale_profiles, env.profiles)

    # Verify accuracy over 20 tasks: fresh router resolves more tasks than 100% stale router
    fresh_hits = 0
    stale_hits = 0
    for t_id in range(20):
        task = gen.generate_task(t_id)
        if task.S_star.issubset(b1_fresh.select(task)):
            fresh_hits += 1
        if task.S_star.issubset(b1_stale.select(task)):
            stale_hits += 1

    assert fresh_hits >= stale_hits


def test_voting_decider_returns_plurality_winner():
    env = RepositoryEnvironment(num_agents=5, embedding_dim=8, seed=42)
    gen = TaskGenerator(env, coalition_size_k=1, rule=ResolutionRule.DISJUNCTIVE, seed=42)
    task = gen.generate_task(0)

    reports = {
        0: PublicReport(0, 0.1, EvidencePayload(0.1)),
        1: PublicReport(1, 0.95, EvidencePayload(0.95)),
        2: PublicReport(2, 0.1, EvidencePayload(0.1)),
        3: PublicReport(3, 0.1, EvidencePayload(0.1)),
        4: PublicReport(4, 0.1, EvidencePayload(0.1)),
    }
    b3 = VotingDeciderB3(seed=42)
    assert b3.select(task, reports) == {1}

