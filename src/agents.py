"""Strategic agent modeling, report shading, dynamic costs, and learning."""

from __future__ import annotations

import numpy as np

from src.types import Evidence, EvidencePayload, MechanismConfig, PublicReport


class StrategicAgent:
    """Represents a candidate owner agent with strategic incentives and learning."""

    def __init__(
        self,
        agent_id: int,
        base_ell: float = 0.0,         # > 0 deflector, < 0 acquirer, 0 aligned
        beta_load: float = 0.5,        # sensitivity to workload backlog
        alpha_credit_ema: float = 0.1, # smoothing factor for expected credit
        seed: int | None = 42,
    ):
        self.agent_id = agent_id
        self.base_ell = base_ell
        self.beta_load = beta_load
        self.alpha_credit_ema = alpha_credit_ema
        self.rng = np.random.default_rng(seed)

        # Dynamic state
        self.active_workload: int = 0
        self.credit_ema: float = 0.0
        self.assignment_history: list[tuple[float, bool]] = []

        # Learned assignment curve parameters: X(r) = sigma(theta0 + theta1 * r)
        self.theta0: float = -2.0
        self.theta1: float = 4.0

    @property
    def current_ell(self) -> float:
        """Dynamic private cost: ell_i(t) = base_ell + beta_load * workload."""
        return float(self.base_ell + self.beta_load * self.active_workload)

    @property
    def effective_ell(self) -> float:
        """Effective incentive: ell_i^eff = ell_i(t) - bar{chi}_i."""
        return float(self.current_ell - self.credit_ema)

    def estimate_marginal_assignment_prob(self, r: float) -> float:
        """Estimates X_i'(r), the marginal increase in assignment probability."""
        # Minimum baseline slope when history is scarce
        if len(self.assignment_history) < 8:
            return 1.0

        # Marginal slope of logistic curve: theta1 * X(r) * (1 - X(r))
        z = self.theta0 + self.theta1 * r
        x = 1.0 / (1.0 + np.exp(-np.clip(z, -10.0, 10.0)))
        slope = self.theta1 * x * (1.0 - x)
        return float(np.clip(slope, 0.1, 2.5))

    def update_assignment_model(self, report_r: float, was_assigned: bool) -> None:
        """Updates internal model of assignment probability X_i(r) from observation."""
        self.assignment_history.append((report_r, was_assigned))

        # Perform online logistic regression update
        pred = 1.0 / (1.0 + np.exp(-np.clip(self.theta0 + self.theta1 * report_r, -10.0, 10.0)))
        error = (1.0 if was_assigned else 0.0) - pred
        lr = 0.05
        self.theta0 += lr * error
        self.theta1 += lr * error * report_r
        # Keep theta1 positive (higher report -> higher assignment chance)
        self.theta1 = max(0.5, self.theta1)

    def decide_tampering(
        self,
        evidence: Evidence,
        config: MechanismConfig,
        degree_m: int,
    ) -> bool:
        """Evaluates whether to tamper with the private evidence payload."""
        # Aligned agents never tamper
        if abs(self.effective_ell) < 1e-4:
            return False

        # If audits are off or penalty is 0, deflector or acquirer might tamper freely
        if config.audit_rate <= 1e-4 or config.audit_penalty <= 1e-4:
            return abs(self.effective_ell) > 1.0

        # Analytical contradiction probability: q_i = alpha * [1 - (1 - delta)^m_i]
        q_i = config.audit_rate * (1.0 - (1.0 - config.audit_detection_delta) ** degree_m)
        expected_penalty = q_i * config.audit_penalty

        # Expected benefit of complete evasion or false claiming
        evasion_benefit = abs(self.effective_ell) * 0.5

        # Tamper only if expected gain outweighs expected penalty
        return bool(evasion_benefit > expected_penalty)

    def compute_report(
        self,
        evidence: Evidence,
        config: MechanismConfig,
        degree_m: int,
    ) -> PublicReport:
        """Computes strategic report r_i and generates submitted public report."""
        eff_ell = self.effective_ell
        pi_hat = evidence.pi_hat

        # 1. Aligned agent: reports subjective belief honestly
        if abs(eff_ell) < 1e-4:
            r = pi_hat
        else:
            # 2. Strategic agent under scoring (lambda > 0)
            if config.score_scale > 1e-4:
                # Fixed-point approximation for r*: r* = clip(pi_hat - (eff_ell / (2 * lambda)) * X'(r*))
                r = pi_hat
                for _ in range(5):
                    slope = self.estimate_marginal_assignment_prob(r)
                    shading = (eff_ell / (2.0 * config.score_scale)) * slope
                    r = np.clip(pi_hat - shading, 0.0, 1.0)
            else:
                # 3. Without scoring (lambda = 0): extreme unmediated reporting
                r = 0.0 if eff_ell > 0 else 1.0

        # 4. Decide tampering
        should_tamper = self.decide_tampering(evidence, config, degree_m)

        # Build submitted payload (is_tampered_ground_truth is private to the artifact)
        submitted_payload = EvidencePayload(
            claimed_signal=evidence.payload.claimed_signal,
            is_tampered_ground_truth=should_tamper,
        )

        return PublicReport(
            agent_id=self.agent_id,
            r=float(r),
            payload=submitted_payload,
        )

    def record_round_outcome(
        self,
        credit_earned: float,
        penalty_incurred: float,
        was_assigned: bool,
        task_resolved: bool,
    ) -> None:
        """Updates internal dynamic state following round settlement."""
        if was_assigned:
            # Update EMA of received credit
            self.credit_ema = (
                (1.0 - self.alpha_credit_ema) * self.credit_ema
                + self.alpha_credit_ema * credit_earned
            )
            # Increase workload
            self.active_workload += 1

        # Simulate gradual resolution of active tasks
        if self.active_workload > 0 and self.rng.random() < 0.4:
            self.active_workload -= 1
