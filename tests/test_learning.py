"""Tests for agent strategic behavior, report shading, and dynamic costs."""

import numpy as np
import pytest

from src.agents import StrategicAgent
from src.types import Evidence, EvidencePayload, MechanismConfig


@pytest.fixture
def sample_evidence():
    payload = EvidencePayload(claimed_signal=0.5, is_tampered_ground_truth=False)
    return Evidence(agent_id=0, raw_signal=0.5, pi_true=0.7, pi_hat=0.7, payload=payload)


def test_aligned_agent_is_truthful(sample_evidence):
    agent = StrategicAgent(agent_id=0, base_ell=0.0)
    config = MechanismConfig(score_scale=1.0)
    report, is_tampered = agent.compute_report(sample_evidence, config, degree_m=2)
    assert np.isclose(report.r, sample_evidence.pi_hat)
    assert not is_tampered
    assert not sample_evidence.payload.is_tampered_ground_truth


def test_deflector_shades_down(sample_evidence):
    agent = StrategicAgent(agent_id=1, base_ell=2.0)  # Deflector
    config = MechanismConfig(score_scale=2.0)
    report, _ = agent.compute_report(sample_evidence, config, degree_m=2)
    assert report.r < sample_evidence.pi_hat, "Deflector must shade report downwards"


def test_acquirer_shades_up(sample_evidence):
    agent = StrategicAgent(agent_id=2, base_ell=-2.0)  # Acquirer
    config = MechanismConfig(score_scale=2.0)
    report, _ = agent.compute_report(sample_evidence, config, degree_m=2)
    assert report.r > sample_evidence.pi_hat, "Acquirer must shade report upwards"


def test_score_scale_disciplines_shading(sample_evidence):
    agent = StrategicAgent(agent_id=1, base_ell=2.0)
    config_weak = MechanismConfig(score_scale=0.5)
    config_strong = MechanismConfig(score_scale=10.0)

    report_weak, _ = agent.compute_report(sample_evidence, config_weak, degree_m=2)
    report_strong, _ = agent.compute_report(sample_evidence, config_strong, degree_m=2)

    # Stronger scoring forces report closer to truth
    assert abs(report_strong.r - sample_evidence.pi_hat) < abs(report_weak.r - sample_evidence.pi_hat)


def test_no_scoring_extreme_reports(sample_evidence):
    deflector = StrategicAgent(agent_id=1, base_ell=1.5)
    acquirer = StrategicAgent(agent_id=2, base_ell=-1.5)
    config_no_scoring = MechanismConfig(score_scale=0.0)

    rep_deflect, _ = deflector.compute_report(sample_evidence, config_no_scoring, degree_m=2)
    rep_acquire, _ = acquirer.compute_report(sample_evidence, config_no_scoring, degree_m=2)
    assert rep_deflect.r == 0.0
    assert rep_acquire.r == 1.0


def test_audit_penalty_deters_tampering(sample_evidence):
    agent = StrategicAgent(agent_id=1, base_ell=3.0)
    
    # Low penalty or zero audit -> tempts tampering
    config_lax = MechanismConfig(score_scale=0.0, audit_rate=0.0, audit_penalty=0.0)
    assert agent.decide_tampering(sample_evidence, config_lax, degree_m=3) is True

    # High audit rate and penalty -> deters tampering
    config_strict = MechanismConfig(score_scale=0.0, audit_rate=0.5, audit_penalty=20.0, audit_detection_delta=0.5)
    assert agent.decide_tampering(sample_evidence, config_strict, degree_m=3) is False


def test_missing_evidence_cannot_tamper():
    payload = EvidencePayload(claimed_signal=0.1, is_tampered_ground_truth=False)
    ev = Evidence(agent_id=0, raw_signal=None, pi_true=0.1, pi_hat=0.1, payload=payload)
    agent = StrategicAgent(agent_id=1, base_ell=3.0)
    config = MechanismConfig(score_scale=0.0, audit_rate=0.0, audit_penalty=0.0)
    assert agent.decide_tampering(ev, config, degree_m=3) is False


def test_tampering_alters_claimed_signal(sample_evidence):
    deflector = StrategicAgent(agent_id=1, base_ell=3.0)
    acquirer = StrategicAgent(agent_id=2, base_ell=-3.0)
    config_lax = MechanismConfig(score_scale=0.0, audit_rate=0.0, audit_penalty=0.0)

    rep_deflect, is_tampered_deflect = deflector.compute_report(sample_evidence, config_lax, degree_m=2)
    rep_acquire, is_tampered_acquire = acquirer.compute_report(sample_evidence, config_lax, degree_m=2)

    assert is_tampered_deflect is True
    assert sample_evidence.payload.is_tampered_ground_truth is True
    assert rep_deflect.payload.claimed_signal < 0.5
    assert is_tampered_acquire is True
    assert rep_acquire.payload.claimed_signal > 0.5


def test_online_learning_assignment_curve():
    agent = StrategicAgent(agent_id=0, base_ell=1.0, seed=42)
    # Before 8 observations, uses baseline slope 1.0
    assert agent.estimate_marginal_assignment_prob(0.5) == 1.0

    # Feed 40 rounds where reports >= 0.6 are assigned and < 0.6 are not
    for i in range(40):
        r = 0.8 if i % 2 == 0 else 0.2
        agent.update_assignment_model(r, was_assigned=(r >= 0.6))

    assert len(agent.assignment_history) == 40
    # Slope near decision boundary should be positive and bounded
    slope_mid = agent.estimate_marginal_assignment_prob(0.5)
    assert 0.1 <= slope_mid <= 2.5


def test_dynamic_workload_stays_bounded_and_tracks_credit():
    agent = StrategicAgent(agent_id=0, base_ell=-2.0, beta_load=0.5, load_decay=0.6, seed=42)
    for _ in range(200):
        agent.record_round_outcome(credit_earned=5.0, penalty_incurred=0.5, was_assigned=True, task_resolved=True)

    # Under decayed recent load with decay 0.6, steady-state cap is 1 / (1 - 0.6) = 2.5
    assert np.isclose(agent.recent_load, 2.5, atol=1e-4)
    assert np.isclose(agent.credit_ema, 5.0, atol=0.1)
    assert np.isclose(agent.total_penalties, 100.0)

    # When idle for several rounds, recent_load decays back toward 0
    for _ in range(20):
        agent.record_round_outcome(credit_earned=0.0, penalty_incurred=0.0, was_assigned=False, task_resolved=True)
    assert agent.recent_load < 0.01
