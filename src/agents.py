"""Strategic agent modeling, report shading, dynamic costs, and learning."""

from __future__ import annotations

import numpy as np

from src.types import Evidence, EvidencePayload, MechanismConfig, PublicPayload, PublicReport


class StrategicAgent:
    """Represents a candidate owner agent with strategic incentives and learning."""

    def __init__(
        self,
        agent_id: int,
        base_ell: float = 0.0,         # > 0 deflector, < 0 acquirer, 0 aligned
        beta_load: float = 0.5,        # sensitivity to recent work fatigue
        alpha_credit_ema: float = 0.1, # smoothing factor for expected credit
        load_decay: float = 0.6,       # decay factor per round for recent work load
        seed: int | None = 42,
        theta0: float = -2.0,          # initial assignment logistic intercept
        theta1: float = 4.0,           # initial assignment logistic slope
        online_lr: float = 0.05,       # learning rate for online logistic regression
        evasion_benefit_factor: float = 0.5, # factor relating ell to evasion benefit
        tamper_deflect_signal: float = -0.5, # forged signal for deflectors
        tamper_acquire_signal: float = 1.0,  # forged signal for acquirers
        fp_iterations: int = 8,        # fixed-point iteration rounds for report shading
    ):
        self.agent_id = agent_id
        self.base_ell = base_ell
        self.beta_load = beta_load
        self.alpha_credit_ema = alpha_credit_ema
        self.load_decay = load_decay
        self.rng = np.random.default_rng(seed)

        # Dynamic state
        self.recent_load: float = 0.0
        self.credit_ema: float = 0.0
        self.total_credits: float = 0.0
        self.total_penalties: float = 0.0
        self.assignment_history: list[tuple[float, bool]] = []

        # Learned assignment curve parameters: X(r) = sigma(theta0 + theta1 * r)
        self.theta0: float = theta0
        self.theta1: float = theta1
        self.online_lr: float = online_lr
        self.evasion_benefit_factor: float = evasion_benefit_factor
        self.tamper_deflect_signal: float = tamper_deflect_signal
        self.tamper_acquire_signal: float = tamper_acquire_signal
        self.fp_iterations: int = fp_iterations

    @property
    def current_ell(self) -> float:
        """Dynamic private cost: ell_i(t) = base_ell + beta_load * recent_load."""
        return float(self.base_ell + self.beta_load * self.recent_load)

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
        lr = self.online_lr
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
        # Cannot tamper if local evidence is missing
        if evidence.raw_signal is None:
            return False

        # Aligned agents never tamper
        if abs(self.effective_ell) < 1e-4:
            return False

        # If audits are off or penalty is 0, strong deflectors/acquirers tamper freely
        if config.audit_rate <= 1e-4 or config.audit_penalty <= 1e-4:
            return abs(self.effective_ell) > 1.0

        # Analytical contradiction probability: q_i = alpha * [1 - (1 - delta)^m_i]
        q_i = config.audit_rate * (1.0 - (1.0 - config.audit_detection_delta) ** degree_m)
        expected_penalty = q_i * config.audit_penalty

        # Expected benefit of fabricating payload
        evasion_benefit = abs(self.effective_ell) * self.evasion_benefit_factor

        # Tamper only if expected gain outweighs expected penalty
        return bool(evasion_benefit > expected_penalty)

    def compute_report(
        self,
        evidence: Evidence,
        config: MechanismConfig,
        degree_m: int,
    ) -> tuple[PublicReport, bool]:
        """Computes strategic report r_i and generates submitted public report.

        Returns:
        - report: PublicReport containing public payload (blind to ground-truth tampering)
        - is_tampered: bool indicating ground-truth tampering status for simulator audit oracle
        """
        eff_ell = self.effective_ell
        pi_hat = evidence.pi_hat

        # 1. Aligned agent: reports subjective belief honestly
        if abs(eff_ell) < 1e-4:
            r = pi_hat
        else:
            # 2. Strategic agent under scoring (lambda > 0)
            if config.score_scale > 1e-4:
                # Damped fixed-point iteration for r*:
                # r* = clip(pi_hat - (eff_ell / (2 * lambda)) * X'(r*), 0, 1)
                r = pi_hat
                for _ in range(self.fp_iterations):
                    slope = self.estimate_marginal_assignment_prob(r)
                    shading = (eff_ell / (2.0 * config.score_scale)) * slope
                    r_next = float(np.clip(pi_hat - shading, 0.0, 1.0))
                    r = 0.5 * r + 0.5 * r_next
            else:
                # 3. Without scoring (lambda = 0): extreme unmediated reporting
                r = 0.0 if eff_ell > 0 else 1.0

        # 4. Decide tampering and construct submitted payload
        should_tamper = self.decide_tampering(evidence, config, degree_m)
        claimed_signal = evidence.payload.claimed_signal
        if should_tamper:
            # Forge claimed signal to match deflection (low) or acquisition (high)
            claimed_signal = self.tamper_deflect_signal if eff_ell > 0 else self.tamper_acquire_signal

        # Update private evidence payload state
        evidence.payload.is_tampered_ground_truth = should_tamper

        # Construct public payload (containing only claimed_signal)
        submitted_payload = PublicPayload(claimed_signal=float(claimed_signal))

        report = PublicReport(
            agent_id=self.agent_id,
            r=float(r),
            payload=submitted_payload,
        )
        return report, should_tamper

    def record_round_outcome(
        self,
        credit_earned: float,
        penalty_incurred: float,
        was_assigned: bool,
        task_resolved: bool,
    ) -> None:
        """Updates internal dynamic state following round settlement."""
        self.total_credits += credit_earned
        self.total_penalties += penalty_incurred

        if was_assigned:
            # Update EMA of realized credit (0.0 if conjunctive task failed)
            realized_credit = credit_earned if task_resolved else 0.0
            self.credit_ema = (
                (1.0 - self.alpha_credit_ema) * self.credit_ema
                + self.alpha_credit_ema * realized_credit
            )

        # Task finishes before next issue; recent_load decays and increments if assigned
        self.recent_load = self.load_decay * self.recent_load + (1.0 if was_assigned else 0.0)
