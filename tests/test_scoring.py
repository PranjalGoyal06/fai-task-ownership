"""Tests for strictly proper Brier scoring rule."""

import numpy as np
import pytest

from src.mechanisms import AssignmentMechanism
from src.types import MechanismConfig


def test_brier_score_expected_value_strictly_proper():
    """Verify that E[s(r, z)] is strictly maximized at r = p for any p in (0, 1)."""
    p_true = 0.73
    reports = np.linspace(0.01, 0.99, 100)

    # E[s(r, z)] = p * s(r, 1) + (1 - p) * s(r, 0)
    expected_scores = [
        p_true * AssignmentMechanism.brier_score(r, 1) + (1 - p_true) * AssignmentMechanism.brier_score(r, 0)
        for r in reports
    ]

    best_idx = np.argmax(expected_scores)
    best_r = reports[best_idx]

    assert np.isclose(best_r, p_true, atol=0.02), f"Brier score maximum {best_r} must match true belief {p_true}"


def test_centered_brier_score_zero_expectation_at_prior():
    """Verify that reporting the base-rate prior yields an expected centered score of 0."""
    pi0 = 0.25
    p_true = pi0
    config = MechanismConfig()
    mech = AssignmentMechanism(config)

    # Expected centered score: E[s_tilde(pi0, z)]
    exp_centered = p_true * mech.centered_brier_score(pi0, 1, pi0) + (1 - p_true) * mech.centered_brier_score(pi0, 0, pi0)
    assert np.isclose(exp_centered, 0.0, atol=1e-7)
