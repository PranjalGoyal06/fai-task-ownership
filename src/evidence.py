"""Evidence generation, observation noise, Bayesian calibration, and miscalibration."""

from __future__ import annotations

import numpy as np

from src.environment import RepositoryEnvironment
from src.types import Evidence, EvidencePayload, Task


def expit(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def logit(p: float | np.ndarray) -> float | np.ndarray:
    """Log-odds transform for p in (0, 1)."""
    p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clipped / (1.0 - p_clipped))


class EvidenceEngine:
    """Generates private evidence observations and subjective agent beliefs."""

    def __init__(
        self,
        env: RepositoryEnvironment,
        obs_noise_sigma: float = 0.15,
        missing_evidence_prob: float = 0.0,  # gamma in [0, 1]
        signal_sensitivity: float = 8.0,     # Bayes log-likelihood slope beta_1
        signal_midpoint: float = 0.23,       # Bayes midpoint (mu_1 + mu_0) / 2
        seed: int | None = 42,
    ):
        self.env = env
        self.obs_noise_sigma = obs_noise_sigma
        self.missing_evidence_prob = missing_evidence_prob
        self.signal_sensitivity = signal_sensitivity
        self.signal_midpoint = signal_midpoint
        self.rng = np.random.default_rng(seed)

    def generate_evidence(
        self,
        task: Task,
        calibration_kappas: np.ndarray | None = None,
    ) -> dict[int, Evidence]:
        """Generates evidence for all N agents for a given task.
        
        Parameters:
        - task: the incoming Task instance
        - calibration_kappas: per-agent distortion exponent (kappa > 0;
          kappa=1 is calibrated, >1 overconfident, <1 underconfident)
        """
        N = self.env.num_agents
        base_rate_pi0 = len(task.S_star) / max(1, N)
        base_logit = float(logit(base_rate_pi0))

        if calibration_kappas is None:
            calibration_kappas = np.ones(N, dtype=float)

        evidence_dict = {}

        # Vectorized cosine similarity: w_i · t
        sims = self.env.profiles @ task.t_embedding

        for i in range(N):
            # 1. Missing evidence check (gamma)
            is_missing = self.rng.random() < self.missing_evidence_prob

            if is_missing:
                raw_signal = None
                pi_true = float(base_rate_pi0)
            else:
                noise = self.rng.normal(0, self.obs_noise_sigma)
                raw_signal = float(sims[i] + noise)
                # Bayesian calibrated log-odds centered at midpoint (mu_1 + mu_0) / 2
                calibrated_logit = base_logit + self.signal_sensitivity * (
                    raw_signal - self.signal_midpoint
                )
                pi_true = float(expit(calibrated_logit))

            # Ensure valid probability range
            pi_true = float(np.clip(pi_true, 1e-5, 1.0 - 1e-5))

            # 2. Miscalibration distortion: g(pi) = pi^kappa / (pi^kappa + (1-pi)^kappa)
            kappa = float(calibration_kappas[i])
            if abs(kappa - 1.0) < 1e-5:
                pi_hat = pi_true
            else:
                pi_pow = pi_true ** kappa
                comp_pow = (1.0 - pi_true) ** kappa
                pi_hat = float(pi_pow / max(1e-9, (pi_pow + comp_pow)))
                pi_hat = float(np.clip(pi_hat, 1e-5, 1.0 - 1e-5))

            # 3. Private payload artifact
            payload = EvidencePayload(
                claimed_signal=raw_signal if raw_signal is not None else 0.0,
                is_tampered_ground_truth=False,
            )

            evidence_dict[i] = Evidence(
                agent_id=i,
                raw_signal=raw_signal,
                pi_true=pi_true,
                pi_hat=pi_hat,
                payload=payload,
            )

        return evidence_dict
