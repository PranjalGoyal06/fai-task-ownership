"""Tests for RepositoryEnvironment, TaskGenerator, and EvidenceEngine."""

import numpy as np
import pytest

from src.environment import RepositoryEnvironment, TaskGenerator
from src.evidence import EvidenceEngine
from src.types import ResolutionRule


def test_repository_environment_initialization():
    env = RepositoryEnvironment(num_agents=8, embedding_dim=16, graph_type="scale_free", avg_degree=2, seed=42)
    assert env.num_agents == 8
    assert env.profiles.shape == (8, 16)
    # Unit norm verification
    norms = np.linalg.norm(env.profiles, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    # Graph check
    assert env.G.number_of_nodes() == 8
    assert env.G.number_of_edges() > 0


def test_task_generator_weighted_embedding():
    env = RepositoryEnvironment(num_agents=10, embedding_dim=16, seed=123)
    generator = TaskGenerator(env, coalition_size_k=3, cluster_prob=1.0, hop_decay=0.5, seed=123)
    task = generator.generate_task(task_id=1)

    assert len(task.S_star) == 3
    assert task.root_module in task.S_star
    assert task.z.sum() == 3
    assert np.isclose(np.linalg.norm(task.t_embedding), 1.0, atol=1e-5)

    # Cosine similarities: Root module should have the highest similarity with task embedding
    sims = env.profiles @ task.t_embedding
    root_sim = sims[task.root_module]
    non_s_star = list(set(range(10)) - task.S_star)
    for j in non_s_star:
        assert root_sim > sims[j], f"Root module {task.root_module} sim ({root_sim}) should exceed non-owner {j} sim ({sims[j]})"


def test_evidence_engine_beliefs_and_miscalibration():
    env = RepositoryEnvironment(num_agents=6, embedding_dim=16, seed=42)
    generator = TaskGenerator(env, coalition_size_k=2, seed=42)
    task = generator.generate_task(task_id=1)

    engine = EvidenceEngine(env, obs_noise_sigma=0.05, seed=42)

    # Perfectly calibrated case (kappa = 1.0)
    ev_calibrated = engine.generate_evidence(task, calibration_kappas=np.ones(6))
    for i in range(6):
        assert 0.0 < ev_calibrated[i].pi_true < 1.0
        assert np.isclose(ev_calibrated[i].pi_true, ev_calibrated[i].pi_hat)

    # Overconfident case (kappa = 2.0)
    ev_overconfident = engine.generate_evidence(task, calibration_kappas=np.full(6, 2.0))
    for i in range(6):
        pi = ev_overconfident[i].pi_true
        pi_hat = ev_overconfident[i].pi_hat
        if pi > 0.5:
            assert pi_hat >= pi, "Overconfidence should push beliefs > 0.5 closer to 1"
        elif pi < 0.5:
            assert pi_hat <= pi, "Overconfidence should push beliefs < 0.5 closer to 0"
