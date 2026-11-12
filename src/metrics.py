"""Evaluation metrics: Social welfare, Oracle benchmarks, Regret decomposition, Calibration, and Incentive summaries."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.types import (
    Evidence,
    EvidencePayload,
    MechanismConfig,
    PublicReport,
    ResolutionRule,
    Task,
)


class MetricsEvaluator:
    """Computes social welfare, oracle/truthful benchmarks, regret decompositions, calibration, and incentive metrics."""

    @staticmethod
    def evaluate_assignment(
        task: Task,
        assigned_set: set[int],
    ) -> tuple[bool, float, float]:
        """Calculates resolution indicator V, social cost c(A), and gross social welfare W.

        Returns:
        - resolved: bool
        - social_cost: float
        - social_welfare: float (R * V - c(A))
        """
        if not task.S_star:
            resolved = False
        elif task.rule == ResolutionRule.CONJUNCTIVE:
            resolved = task.S_star.issubset(assigned_set)
        else:
            resolved = bool(task.S_star.intersection(assigned_set))

        social_cost = float(sum(task.costs[i] for i in assigned_set))
        welfare = float((task.R if resolved else 0.0) - social_cost)
        return resolved, social_cost, welfare

    @classmethod
    def compute_oracle(cls, task: Task) -> tuple[set[int], bool, float, float]:
        """Computes the omniscient ground-truth oracle benchmark M^circ.

        The oracle sees ground truth S* and chooses A^circ to maximize social welfare.

        Returns:
        - A_oracle: set[int]
        - oracle_resolved: bool
        - oracle_cost: float
        - oracle_welfare: float
        """
        if not task.S_star:
            return set(), False, 0.0, 0.0

        if task.rule == ResolutionRule.CONJUNCTIVE:
            # Conjunctive: Must assign all members of S* or abstain
            cost_all = float(sum(task.costs[s] for s in task.S_star))
            if task.R - cost_all > 0:
                return set(task.S_star), True, cost_all, float(task.R - cost_all)
            return set(), False, 0.0, 0.0

        # Disjunctive: Assign the single lowest-cost owner in S*
        best_owner = min(task.S_star, key=lambda s: task.costs[s])
        cost_best = float(task.costs[best_owner])
        if task.R - cost_best > 0:
            return {best_owner}, True, cost_best, float(task.R - cost_best)
        return set(), False, 0.0, 0.0

    @classmethod
    def compute_truthful_benchmark(
        cls,
        task: Task,
        evidence_dict: dict[int, Evidence],
        rng: np.random.Generator | None = None,
    ) -> tuple[set[int], bool, float, float]:
        """Computes the honest full-evidence benchmark M^honest using true posteriors pi_true.

        Separates irreducible private-evidence noise from strategic misreporting/miscalibration.
        """
        from src.mechanisms import AssignmentMechanism

        truthful_reports = {
            i: PublicReport(
                agent_id=i,
                r=float(ev.pi_true),
                payload=EvidencePayload(
                    claimed_signal=float(ev.pi_true),
                    is_tampered_ground_truth=False,
                ),
            )
            for i, ev in evidence_dict.items()
        }
        mech = AssignmentMechanism(MechanismConfig())
        A_honest = mech.select_agents(task, truthful_reports, rng=rng)
        resolved, cost, welfare = cls.evaluate_assignment(task, A_honest)
        return A_honest, resolved, cost, welfare

    @classmethod
    def compute_regret_decomposition(
        cls,
        task: Task,
        assigned_set: set[int],
        audit_cost: float = 0.0,
        evidence_dict: dict[int, Evidence] | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """Computes exact Regret = Miss Loss (Delta V) + Excess Cost (Delta C), plus optional gaps.

        When `evidence_dict` is provided, also decomposes total regret into:
        - information_gap = W(M^circ) - W(M^honest)
        - strategic_gap   = W(M^honest) - W(A)
        """
        resolved, cost, welfare = cls.evaluate_assignment(task, assigned_set)
        _, oracle_resolved, oracle_cost, oracle_welfare = cls.compute_oracle(task)

        V_circ = 1.0 if oracle_resolved else 0.0
        V = 1.0 if resolved else 0.0

        miss_loss = float(task.R * (V_circ - V))
        excess_cost = float(cost - oracle_cost)
        regret = float(oracle_welfare - welfare)
        net_welfare = float(welfare - audit_cost)

        if oracle_welfare > 0:
            efficiency_ratio = float(welfare / oracle_welfare)
        else:
            efficiency_ratio = 1.0 if np.isclose(welfare, 0.0) else 0.0

        result = {
            "resolved": V,
            "social_cost": cost,
            "social_welfare": welfare,
            "net_welfare": net_welfare,
            "oracle_welfare": oracle_welfare,
            "regret": regret,
            "miss_loss": miss_loss,
            "excess_cost": excess_cost,
            "efficiency_ratio": efficiency_ratio,
        }

        if evidence_dict is not None:
            _, _, _, honest_welfare = cls.compute_truthful_benchmark(task, evidence_dict, rng=rng)
            result["honest_welfare"] = honest_welfare
            result["information_gap"] = float(oracle_welfare - honest_welfare)
            result["strategic_gap"] = float(honest_welfare - welfare)

        return result

    @staticmethod
    def compute_ece(
        probs: Sequence[float] | np.ndarray,
        labels: Sequence[int | float] | np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Computes Expected Calibration Error (ECE) over predicted probabilities and binary labels."""
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=float)
        if len(p) == 0:
            return 0.0

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(p)
        for b in range(n_bins):
            if b == n_bins - 1:
                mask = (p >= bins[b]) & (p <= bins[b + 1])
            else:
                mask = (p >= bins[b]) & (p < bins[b + 1])
            if np.any(mask):
                ece += (float(np.sum(mask)) / n) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
        return float(ece)

    @staticmethod
    def compute_mean_brier(
        probs: Sequence[float] | np.ndarray,
        labels: Sequence[int | float] | np.ndarray,
    ) -> float:
        """Computes mean proper Brier score 1 - (z - r)^2 (higher is better, max 1.0)."""
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=float)
        if len(p) == 0:
            return 0.0
        return float(np.mean(1.0 - (y - p) ** 2))

    @staticmethod
    def compute_incentive_summary(
        task: Task,
        reports: dict[int, PublicReport],
        scores: dict[int, float],
        credits: dict[int, float],
        penalties: dict[int, float],
        audit_cost: float,
    ) -> dict[str, float]:
        """Summarizes mechanism transfers, audit cost, tampering rate, and owner credit share."""
        total_score = float(sum(scores.values()))
        total_credit = float(sum(credits.values()))
        total_penalty = float(sum(penalties.values()))
        audit_cost_f = float(audit_cost)

        num_agents = len(reports)
        tampered_count = sum(1 for rep in reports.values() if rep.payload.is_tampered_ground_truth)
        tampering_rate = float(tampered_count / num_agents) if num_agents > 0 else 0.0

        owner_credit = sum(credits.get(s, 0.0) for s in task.S_star)
        owner_credit_share = float(owner_credit / total_credit) if total_credit > 0 else 0.0

        return {
            "total_score_transfer": total_score,
            "total_credit_payout": total_credit,
            "total_penalties": total_penalty,
            "audit_cost": audit_cost_f,
            "net_principal_spend": float(total_score + total_credit - total_penalty + audit_cost_f),
            "tampering_rate": tampering_rate,
            "owner_credit_share": owner_credit_share,
        }
