"""Evidence generation, observation noise, Bayesian calibration, and miscalibration."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from src.environment import RepositoryEnvironment
from src.types import Evidence, EvidencePayload, Task


class EvidenceEngine:
    """Generates private evidence observations and subjective agent beliefs."""

    def __init__(
        self,
        env: RepositoryEnvironment,
        obs_noise_sigma: float = 0.15,
        missing_evidence_prob: float = 0.0,  # gamma in [0, 1]
        signal_sensitivity: float = 5.0,     # beta_1 in logistic link
        seed: int | None = 42,
    ):
        self.env = env
        self.obs_noise_sigma = obs_noise_sigma
        self.missing_evidence_prob = missing_evidence_prob
        self.signal_sensitivity = signal_sensitivity
        self.rng = np.random.default_rng(seed)

    def generate_evidence(
        self,
        task: Task,
        calibration_kappas: np.ndarray | None = None,
    ) -> dict[int, Evidence]:
        """Generates evidence for all N agents for a given task.
        
        Parameters:
        - task: the incoming Task instance
        - calibration_kappas: per-agent overconfidence exponent (kappa >= 1)
        """
        N = self.env.num_agents
        base_rate_pi0 = len(task.S_star) / max(1, N)
        base_logit = logit(np.clip(base_rate_pi0, 1e-4, 1.0 - 1e-4))

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
                # Logistic calibrated belief
                calibrated_logit = base_logit + self.signal_sensitivity * raw_signal
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
