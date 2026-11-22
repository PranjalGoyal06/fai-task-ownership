"""Comparison baselines: B1 (Description Router), B2 (Polling Router), B3 (Voting), and B4 (Oracle)."""

from __future__ import annotations

import collections
import numpy as np

from src.evidence import expit, logit
from src.mechanisms import AssignmentMechanism
from src.metrics import MetricsEvaluator
from src.types import (
    MechanismConfig,
    PublicPayload,
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
        desc_noise_sigma: float = 0.25,
        signal_sensitivity: float = 8.0,
        signal_midpoint: float = 0.25,
        prior_pi0: float | None = None,
        seed: int | None = 42,
    ):
        self.rho = staleness_rho
        self.desc_noise_sigma = desc_noise_sigma
        self.signal_sensitivity = signal_sensitivity
        self.signal_midpoint = signal_midpoint
        self.num_agents, self.dim = env_profiles.shape
        self.prior_pi0 = prior_pi0 if prior_pi0 is not None else (1.0 / max(1, self.num_agents))
        self.rng = np.random.default_rng(seed)
        self.mech = AssignmentMechanism(MechanismConfig(score_scale=0.0))

        # Stored capability profiles and initial stale profile sample
        self.base_profiles = np.copy(env_profiles)
        self.stale_profiles = self._sample_stale_profiles()

    def _sample_stale_profiles(self) -> np.ndarray:
        """Sample capability profiles where a random fraction rho is corrupted by staleness."""
        stale = np.copy(self.base_profiles)
        num_corrupt = int(np.round(self.rho * self.num_agents))
        if num_corrupt > 0:
            corrupt_indices = self.rng.choice(self.num_agents, size=num_corrupt, replace=False)
            raw_stale = self.rng.normal(0, 1.0, size=(num_corrupt, self.dim))
            norms = np.linalg.norm(raw_stale, axis=1, keepdims=True)
            stale[corrupt_indices] = raw_stale / np.maximum(norms, 1e-9)
        return stale

    def select(self, task: Task) -> set[int]:
        N = self.num_agents
        pi0 = self.prior_pi0
        base_logit = float(logit(pi0))

        # Re-sample stale profiles dynamically so corruption is randomly distributed across tasks
        self.stale_profiles = self._sample_stale_profiles()

        # Score incoming task against stored (partially stale, coarse) descriptions
        stale_sims = self.stale_profiles @ task.t_embedding
        if self.desc_noise_sigma > 0:
            stale_sims = stale_sims + self.rng.normal(0, self.desc_noise_sigma, size=N)
        stale_scores = expit(base_logit + self.signal_sensitivity * (stale_sims - self.signal_midpoint))

        # Wrap as pseudo-reports
        pseudo_reports = {
            i: PublicReport(i, float(stale_scores[i]), PublicPayload(float(stale_scores[i])))
            for i in range(N)
        }

        return self.mech.select_agents(task, pseudo_reports)


class PollingRouterB2:
    """B2: Applies selection rule to reports with no incentive layer (M_{0,0})."""

    def __init__(self):
        self.mech = AssignmentMechanism(MechanismConfig(score_scale=0.0, audit_rate=0.0))

    def select(
        self,
        task: Task,
        reports: dict[int, PublicReport],
        rng: np.random.Generator | None = None,
    ) -> set[int]:
        return self.mech.select_agents(task, reports, rng=rng)


class VotingDeciderB3:
    """B3: Each agent votes for the agent it believes owns the task, plurality winner is assigned.
    
    Each agent i observes ONLY its own report/belief r_i and coarse public
    capability similarity for peers j != i (never observing other agents' private reports r_j).
    """

    def __init__(
        self,
        env_profiles: np.ndarray | None = None,
        peer_noise_sigma: float = 0.25,
        signal_sensitivity: float = 8.0,
        signal_midpoint: float | None = None,
        prior_pi0: float | None = None,
        seed: int | None = 42,
    ):
        self.env_profiles = env_profiles
        self.peer_noise_sigma = peer_noise_sigma
        self.signal_sensitivity = signal_sensitivity
        self.signal_midpoint = signal_midpoint
        self.prior_pi0 = prior_pi0
        self.rng = np.random.default_rng(seed)

    def select(self, task: Task, reports: dict[int, PublicReport]) -> set[int]:
        N = len(reports)
        pi0 = self.prior_pi0 if self.prior_pi0 is not None else (1.0 / max(1, N))
        base_logit = float(logit(pi0))

        # Auto-compute signal midpoint if not explicitly configured
        if self.signal_midpoint is not None:
            midpoint = self.signal_midpoint
        else:
            estimated_k = max(1.0, round(pi0 * N))
            midpoint = float(0.23 * np.sqrt(2.0 / estimated_k))

        votes: list[int] = []

        # Public baseline similarity if profiles are provided
        if self.env_profiles is not None:
            public_sims = self.env_profiles @ task.t_embedding
        else:
            public_sims = np.zeros(N, dtype=float)

        for i in range(N):
            # Agent i estimates peer probabilities from coarse public capability + local noise
            noisy_sims = public_sims + self.rng.normal(0, self.peer_noise_sigma, size=N)
            perceived = expit(
                base_logit + self.signal_sensitivity * (noisy_sims - midpoint)
            )
            # For itself (j == i), agent i uses its own private report/claim
            perceived[i] = reports[i].r

            vote = int(np.argmax(perceived))
            votes.append(vote)

        # Plurality winner with randomized tie-breaking among top candidates
        counts = collections.Counter(votes)
        max_votes = max(counts.values())
        top_candidates = [cand for cand, cnt in counts.items() if cnt == max_votes]
        if len(top_candidates) > 1:
            winner = int(self.rng.choice(top_candidates))
        else:
            winner = top_candidates[0]
        return {winner}


class OracleBaselineB4:
    """B4: Omniscient oracle M^circ."""

    @staticmethod
    def select(task: Task) -> set[int]:
        A_circ, _, _, _ = MetricsEvaluator.compute_oracle(task)
        return A_circ
