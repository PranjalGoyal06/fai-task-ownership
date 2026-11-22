import numpy as np

from src.e1.dataset import owners
from src.e1.game import Condition, fit_strategy, information, population, select, submit
from src.e1.replay import settle
from src.mechanisms import AssignmentMechanism
from src.types import EvidencePayload, MechanismConfig, PublicReport, ResolutionRule, Task


def test_production_paths_and_cross_group_rename():
    patch = """diff --git a/django/core/x.py b/django/forms/x.py
diff --git a/tests/core/test_x.py b/tests/core/test_x.py
diff --git a/docs/index.txt b/docs/index.txt"""
    assert owners(patch) == [4, 5]


def test_equal_cost_selector_matches_shared_objective():
    rng = np.random.default_rng(12)
    mechanism = AssignmentMechanism(MechanismConfig())
    for p in rng.random((32, 9)):
        task = Task(1, np.array([]), 0, set(), np.zeros(9), 8., np.ones(9),
                    ResolutionRule.CONJUNCTIVE)
        reports = {i: PublicReport(i, x, EvidencePayload(x)) for i, x in enumerate(p)}
        chosen = mechanism.select_agents(task, reports)
        assert select(p, task.R) == sum(1 << i for i in chosen)


def test_fix_reveal_changes_settlement_only():
    p = np.tile([.9, .25, .01, .02, .03, .01, .03, .01, .01], (2, 1))
    c = Condition(ell=-1., phi=1/3, scale=2., audit=.25)
    training_y = np.zeros_like(p, dtype=int)
    training_y[:, 0] = 1
    degree = np.ones_like(p)
    prior = np.full(9, 1/9)
    policy, threshold, _ = fit_strategy(p, training_y, degree, prior, c, 4)
    types, checked, exposed = population(5, 2, c.phi, 9)
    _, effective, _ = submit(policy, p, information(p, threshold, types),
                              (checked, exposed), degree, prior, c)
    assigned = select(effective, c.reward)
    # Holding assignments fixed, omitting either historical owner is a miss.
    for target, expected in [([1, 0], 1), ([1, 1], 0)]:
        y = np.zeros((1, 9), dtype=int)
        y[0, :2] = target
        frame = settle(["fixture"], y, np.array([1]), np.array([3]), c.reward)
        assert frame.coverage.iloc[0] == expected
        assert frame.gap.iloc[0] == frame.miss_loss.iloc[0] + frame.excess_cost.iloc[0]
    assert assigned.shape == (2,)


def test_pure_report_shading_and_altered_payload_are_distinct():
    p = np.full((1, 9), .2)
    policy = np.full((9, 4), 21)
    c = Condition(audit=1., delta=1.)
    args = (p, np.zeros_like(p, dtype=int), (np.zeros_like(p), np.zeros_like(p)),
            np.ones_like(p), np.full(9, .1), c)
    report, used, caught = submit(policy, *args)
    assert np.all(report == 1) and np.all(used == 1) and not caught.any()
    report, used, caught = submit(policy + 22, *args)
    assert np.all(report == 1) and np.all(used == .1) and caught.all()
