"""Comparison baselines: B1 (Description Router), B2 (Polling Router), B3 (Voting), and B4 (Oracle)."""

from __future__ import annotations

import collections
import numpy as np
from scipy.special import expit, logit

from src.mechanisms import AssignmentMechanism
from src.metrics import MetricsEvaluator
from src.types import (
    EvidencePayload,
    MechanismConfig,
    PublicReport,
    Task,
)


class DescriptionRouterB1:
    """B1: Applies selection rule to stale description-based scores pi_tilde,
    where a fraction rho of agent capability descriptions is outdated.
    Does NOT observe private evidence e_i.
    """

    def __init__(
        self,
        env_profiles: np.ndarray,
        staleness_rho: float = 0.3,
        signal_sensitivity: float = 5.0,
        seed: int | None = 42,
    ):
        self.rho = staleness_rho
        self.signal_sensitivity = signal_sensitivity
        self.rng = np.random.default_rng(seed)
        self.num_agents, dim = env_profiles.shape

        # Copy written capability profiles and corrupt a fraction rho as stale
        self.stale_profiles = np.copy(env_profiles)
        num_corrupt = int(np.round(self.rho * self.num_agents))
        if num_corrupt > 0:
            corrupt_indices = self.rng.choice(self.num_agents, size=num_corrupt, replace=False)
            raw_stale = self.rng.normal(0, 1.0, size=(num_corrupt, dim))
            norms = np.linalg.norm(raw_stale, axis=1, keepdims=True)
            self.stale_profiles[corrupt_indices] = raw_stale / np.maximum(norms, 1e-9)

    def select(self, task: Task) -> set[int]:
        N = self.num_agents
        base_rate_pi0 = len(task.S_star) / max(1, N)
        base_logit = logit(np.clip(base_rate_pi0, 1e-4, 1.0 - 1e-4))

        # Score incoming task solely against stored (partially stale) descriptions
        stale_sims = self.stale_profiles @ task.t_embedding
        stale_scores = expit(base_logit + self.signal_sensitivity * stale_sims)

        # Wrap as pseudo-reports
        pseudo_reports = {
            i: PublicReport(i, float(stale_scores[i]), EvidencePayload(float(stale_scores[i])))
            for i in range(N)
        }

        # Apply mechanism selection rule
        mech = AssignmentMechanism(MechanismConfig(score_scale=0.0))
        return mech.select_agents(task, pseudo_reports)


class PollingRouterB2:
    """B2: Applies selection rule to reports with no incentive layer (M_{0,0})."""

    def __init__(self):
        self.mech = AssignmentMechanism(MechanismConfig(score_scale=0.0, audit_rate=0.0))

    def select(self, task: Task, reports: dict[int, PublicReport]) -> set[int]:
        return self.mech.select_agents(task, reports)


class VotingDeciderB3:
    """B3: Each agent votes for the agent it believes owns the task, plurality winner is assigned."""

    def __init__(self, seed: int | None = 42):
        self.rng = np.random.default_rng(seed)

    def select(self, task: Task, reports: dict[int, PublicReport]) -> set[int]:
        N = len(reports)
        votes: list[int] = []

        # Each agent votes for the highest reported candidate (or itself if report is high)
        r_arr = np.array([reports[i].r for i in range(N)])
        for i in range(N):
            # Vote for highest candidate with slight local noise
            noisy_r = r_arr + self.rng.normal(0, 0.05, size=N)
            vote = int(np.argmax(noisy_r))
            votes.append(vote)

        # Plurality winner
        counts = collections.Counter(votes)
        winner = counts.most_common(1)[0][0]
        return {winner}


class OracleBaselineB4:
    """B4: Omniscient oracle M^circ."""

    @staticmethod
    def select(task: Task) -> set[int]:
        A_circ, _, _, _ = MetricsEvaluator.compute_oracle(task)
        return A_circ
