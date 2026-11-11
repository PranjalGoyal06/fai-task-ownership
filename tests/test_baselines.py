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


def test_voting_decider_returns_plurality_winner_without_report_leakage():
    env = RepositoryEnvironment(num_agents=5, embedding_dim=16, seed=42)
    gen = TaskGenerator(env, coalition_size_k=1, rule=ResolutionRule.DISJUNCTIVE, seed=42)
    task = gen.generate_task(0)
    owner = task.root_module

    # Even if non-owners don't know owner's report, public profiles point toward owner
    reports = {
        i: PublicReport(i, 0.95 if i == owner else 0.05, EvidencePayload(0.5))
        for i in range(5)
    }
    b3 = VotingDeciderB3(env.profiles, peer_noise_sigma=0.05, seed=42)
    assert b3.select(task, reports) == {owner}

    # Verify agent i does NOT leak peer reports r_j: if we change owner's report to 0.0,
    # non-owners (j != owner) still vote for owner based on public capability match!
    reports_deflect = dict(reports)
    reports_deflect[owner] = PublicReport(owner, 0.0, EvidencePayload(0.0))
    b3_2 = VotingDeciderB3(env.profiles, peer_noise_sigma=0.05, seed=42)
    assert b3_2.select(task, reports_deflect) == {owner}

