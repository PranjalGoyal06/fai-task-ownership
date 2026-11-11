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
    report = agent.compute_report(sample_evidence, config, degree_m=2)
    assert np.isclose(report.r, sample_evidence.pi_hat)
    assert not report.payload.is_tampered_ground_truth


def test_deflector_shades_down(sample_evidence):
    agent = StrategicAgent(agent_id=1, base_ell=2.0)  # Deflector
    config = MechanismConfig(score_scale=2.0)
    report = agent.compute_report(sample_evidence, config, degree_m=2)
    assert report.r < sample_evidence.pi_hat, "Deflector must shade report downwards"


def test_acquirer_shades_up(sample_evidence):
    agent = StrategicAgent(agent_id=2, base_ell=-2.0)  # Acquirer
    config = MechanismConfig(score_scale=2.0)
    report = agent.compute_report(sample_evidence, config, degree_m=2)
    assert report.r > sample_evidence.pi_hat, "Acquirer must shade report upwards"


def test_score_scale_disciplines_shading(sample_evidence):
    agent = StrategicAgent(agent_id=1, base_ell=2.0)
    config_weak = MechanismConfig(score_scale=0.5)
    config_strong = MechanismConfig(score_scale=10.0)

    report_weak = agent.compute_report(sample_evidence, config_weak, degree_m=2)
    report_strong = agent.compute_report(sample_evidence, config_strong, degree_m=2)

    # Stronger scoring forces report closer to truth
    assert abs(report_strong.r - sample_evidence.pi_hat) < abs(report_weak.r - sample_evidence.pi_hat)


def test_no_scoring_extreme_reports(sample_evidence):
    deflector = StrategicAgent(agent_id=1, base_ell=1.5)
    acquirer = StrategicAgent(agent_id=2, base_ell=-1.5)
    config_no_scoring = MechanismConfig(score_scale=0.0)

    assert deflector.compute_report(sample_evidence, config_no_scoring, degree_m=2).r == 0.0
    assert acquirer.compute_report(sample_evidence, config_no_scoring, degree_m=2).r == 1.0


def test_audit_penalty_deters_tampering(sample_evidence):
    agent = StrategicAgent(agent_id=1, base_ell=3.0)
    
    # Low penalty or zero audit -> tempts tampering
    config_lax = MechanismConfig(score_scale=0.0, audit_rate=0.0, audit_penalty=0.0)
    assert agent.decide_tampering(sample_evidence, config_lax, degree_m=3) is True

    # High audit rate and penalty -> deters tampering
    config_strict = MechanismConfig(score_scale=0.0, audit_rate=0.5, audit_penalty=20.0, audit_detection_delta=0.5)
    assert agent.decide_tampering(sample_evidence, config_strict, degree_m=3) is False
