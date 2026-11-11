"""Evaluation metrics: Social welfare, Oracle benchmark, Miss Loss, Excess Cost, and Regret."""

from __future__ import annotations

import numpy as np

from src.types import ResolutionRule, Task


class MetricsEvaluator:
    """Computes social welfare, oracle assignment, and regret decomposition."""

    @staticmethod
    def evaluate_assignment(
        task: Task,
        assigned_set: set[int],
    ) -> tuple[bool, float, float]:
        """Calculates resolution indicator V, social cost c(A), and social welfare W.
        
        Returns:
        - resolved: bool
        - social_cost: float
        - social_welfare: float
        """
        if task.rule == ResolutionRule.CONJUNCTIVE:
            resolved = task.S_star.issubset(assigned_set)
        else:
            resolved = bool(task.S_star.intersection(assigned_set))

        social_cost = float(sum(task.costs[i] for i in assigned_set))
        welfare = float((task.R if resolved else 0.0) - social_cost)
        return resolved, social_cost, welfare

    @classmethod
    def compute_oracle(cls, task: Task) -> tuple[set[int], bool, float, float]:
        """Computes the omniscient oracle benchmark M^circ.
        
        The oracle sees ground truth S* and chooses A^circ to maximize social welfare.
        
        Returns:
        - A_oracle: set[int]
        - oracle_resolved: bool
        - oracle_cost: float
        - oracle_welfare: float
        """
        if task.rule == ResolutionRule.CONJUNCTIVE:
            # Conjunctive: Must assign all members of S* or none
            cost_all = float(sum(task.costs[s] for s in task.S_star))
            if task.R - cost_all > 0:
                A_circ = set(task.S_star)
                return A_circ, True, cost_all, task.R - cost_all
            else:
                return set(), False, 0.0, 0.0

        else:
            # Disjunctive: Assign the single lowest-cost owner in S*
            if not task.S_star:
                return set(), False, 0.0, 0.0

            best_owner = min(task.S_star, key=lambda s: task.costs[s])
            cost_best = float(task.costs[best_owner])
            if task.R - cost_best > 0:
                return {best_owner}, True, cost_best, task.R - cost_best
            else:
                return set(), False, 0.0, 0.0

    @classmethod
    def compute_regret_decomposition(
        cls,
        task: Task,
        assigned_set: set[int],
    ) -> dict[str, float]:
        """Computes exact Regret = Delta V (Miss Loss) + Delta C (Excess Cost)."""
        resolved, cost, welfare = cls.evaluate_assignment(task, assigned_set)
        A_circ, oracle_resolved, oracle_cost, oracle_welfare = cls.compute_oracle(task)

        V_circ = 1.0 if oracle_resolved else 0.0
        V = 1.0 if resolved else 0.0

        miss_loss = float(task.R * (V_circ - V))
        excess_cost = float(cost - oracle_cost)
        regret = float(oracle_welfare - welfare)

        # Regret identically equals miss_loss + excess_cost
        assert np.isclose(regret, miss_loss + excess_cost, atol=1e-5)

        efficiency_ratio = float(welfare / oracle_welfare) if oracle_welfare > 0 else 0.0

        return {
            "resolved": V,
            "social_cost": cost,
            "social_welfare": welfare,
            "oracle_welfare": oracle_welfare,
            "regret": regret,
            "miss_loss": miss_loss,
            "excess_cost": excess_cost,
            "efficiency_ratio": efficiency_ratio,
        }
