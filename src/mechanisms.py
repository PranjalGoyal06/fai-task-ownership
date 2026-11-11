"""Mechanisms: Selection rules, centered Brier scoring, graph audits, and credit rules."""

from __future__ import annotations

import itertools
import math
import networkx as nx
import numpy as np

from src.types import (
    CreditRule,
    MechanismConfig,
    PublicReport,
    ResolutionRule,
    Task,
)


class AssignmentMechanism:
    """Implements assignment selection, scoring, audits, and credit settlement."""

    def __init__(self, config: MechanismConfig):
        self.config = config

    # -------------------------------------------------------------------------
    # 1. Selection Rules
    # -------------------------------------------------------------------------

    def select_agents(
        self,
        task: Task,
        reports: dict[int, PublicReport],
        graph: nx.Graph | None = None,
    ) -> set[int]:
        """Selects assigned set A subseteq [N] based on task resolution rule."""
        if task.rule == ResolutionRule.DISJUNCTIVE:
            return self._select_escalation(task, reports)
        else:
            return self._select_conjunctive(task, reports)

    def _select_escalation(
        self,
        task: Task,
        reports: dict[int, PublicReport],
    ) -> set[int]:
        """Escalation rule: order agents by r_i / c_i descending; assign until resolved."""
        N = len(reports)
        # Sort by efficiency ratio r_i / c_i descending
        r_over_c = []
        for i in range(N):
            c_i = max(task.costs[i], 1e-4)
            r_i = reports[i].r
            r_over_c.append((r_i / c_i, i))

        r_over_c.sort(key=lambda x: x[0], reverse=True)

        assigned: set[int] = set()
        for _, agent_id in r_over_c:
            assigned.add(agent_id)
            # In simulation, escalation stops as soon as a required owner is reached
            if agent_id in task.S_star:
                break

        return assigned

    def _select_conjunctive(
        self,
        task: Task,
        reports: dict[int, PublicReport],
    ) -> set[int]:
        """Conjunctive set selection (Equation 2 in proposal):
        
        A(r) in argmax_{A subseteq [N]} { R * prod_{j not in A} (1 - r_j) - sum_{i in A} c_i }
        """
        N = len(reports)
        r_arr = np.array([reports[i].r for i in range(N)])
        costs = task.costs
        R = task.R

        # For N <= 16, perform exact combinatorial search
        if N <= 16:
            best_A: set[int] = set()
            best_val = -float("inf")

            for k in range(N + 1):
                for subset in itertools.combinations(range(N), k):
                    A = set(subset)
                    # Compute prod_{j not in A} (1 - r_j)
                    prob_no_missed = 1.0
                    for j in range(N):
                        if j not in A:
                            prob_no_missed *= (1.0 - r_arr[j])

                    cost_A = sum(costs[i] for i in A)
                    obj = R * prob_no_missed - cost_A

                    if obj > best_val:
                        best_val = obj
                        best_A = A

            return best_A
        else:
            # Greedy marginal search for larger N
            current_A: set[int] = set()
            while True:
                best_candidate = None
                best_gain = 0.0

                current_prob = math.prod(1.0 - r_arr[j] for j in range(N) if j not in current_A)
                current_cost = sum(costs[i] for i in current_A)
                current_obj = R * current_prob - current_cost

                for cand in range(N):
                    if cand not in current_A:
                        cand_A = current_A | {cand}
                        new_prob = math.prod(1.0 - r_arr[j] for j in range(N) if j not in cand_A)
                        new_cost = current_cost + costs[cand]
                        new_obj = R * new_prob - new_cost

                        gain = new_obj - current_obj
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = cand

                if best_candidate is not None and best_gain > 1e-4:
                    current_A.add(best_candidate)
                else:
                    break

            return current_A

    # -------------------------------------------------------------------------
    # 2. Centered Proper Scoring Rule (Brier Score)
    # -------------------------------------------------------------------------

    @staticmethod
    def brier_score(r: float, z: int) -> float:
        """Strictly proper Brier score: s(r, z) = 1 - (z - r)^2."""
        return float(1.0 - (z - r) ** 2)

    def centered_brier_score(self, r: float, z: int, pi0: float) -> float:
        """Centered Brier score: s_tilde(r, z) = s(r, z) - s(pi0, z) = -(z - r)^2 + (z - pi0)^2."""
        return float(-(z - r) ** 2 + (z - pi0) ** 2)

    def compute_all_scores(
        self,
        reports: dict[int, PublicReport],
        z_vector: np.ndarray,
        base_rate_pi0: float,
    ) -> dict[int, float]:
        """Calculates scaled centered Brier scores: lambda * s_tilde_i for all agents."""
        scores = {}
        for i, rep in reports.items():
            s_tilde = self.centered_brier_score(rep.r, int(z_vector[i]), base_rate_pi0)
            scores[i] = float(self.config.score_scale * s_tilde)
        return scores

    # -------------------------------------------------------------------------
    # 3. Graph-Based Audits
    # -------------------------------------------------------------------------

    def execute_audits(
        self,
        reports: dict[int, PublicReport],
        graph: nx.Graph,
        rng: np.random.Generator,
    ) -> tuple[dict[int, float], list[int], list[int], float]:
        """Executes stochastic audits along dependency graph G.
        
        Returns:
        - penalties: dict of penalty amount per agent
        - audited_agents: list of audited agent IDs
        - contradicted_agents: list of caught agents
        - total_audit_cost: c_a * number_of_audited_agents
        """
        penalties = {i: 0.0 for i in reports}
        audited_agents: list[int] = []
        contradicted_agents: list[int] = []

        if self.config.audit_rate <= 1e-5:
            return penalties, audited_agents, contradicted_agents, 0.0

        for i, rep in reports.items():
            # Stochastic spot-check at audit rate alpha
            if rng.random() < self.config.audit_rate:
                audited_agents.append(i)

                # If the agent secretly tampered with its payload, inspect neighbors
                if rep.payload.is_tampered_ground_truth:
                    neighbors = list(graph.neighbors(i))
                    contradicted = False
                    for _ in neighbors:
                        if rng.random() < self.config.audit_detection_delta:
                            contradicted = True
                            break

                    if contradicted:
                        penalties[i] = float(self.config.audit_penalty)
                        contradicted_agents.append(i)

        total_audit_cost = float(len(audited_agents) * self.config.cost_per_audit)
        return penalties, audited_agents, contradicted_agents, total_audit_cost

    # -------------------------------------------------------------------------
    # 4. Credit Division for Coalitions
    # -------------------------------------------------------------------------

    def divide_credit(
        self,
        task: Task,
        assigned_set: set[int],
    ) -> dict[int, float]:
        """Computes credit payout chi_i to assigned agents under chosen credit rule."""
        N = len(task.costs)
        credits = {i: 0.0 for i in range(N)}

        if not assigned_set:
            return credits

        # Check resolution
        resolved = False
        if task.rule == ResolutionRule.CONJUNCTIVE:
            resolved = task.S_star.issubset(assigned_set)
        else:
            resolved = bool(task.S_star.intersection(assigned_set))

        if not resolved:
            return credits

        # Credit Rule 1: Equal Split
        if self.config.credit_rule == CreditRule.EQUAL_SPLIT:
            split_payout = float(task.R / len(assigned_set))
            for i in assigned_set:
                credits[i] = split_payout
            return credits

        # Credit Rule 2: Shapley Value Credit
        if self.config.credit_rule == CreditRule.SHAPLEY:
            return self._compute_shapley_credit(task, assigned_set)

        return credits

    def _compute_shapley_credit(
        self,
        task: Task,
        assigned_set: set[int],
    ) -> dict[int, float]:
        """Computes exact Shapley values psi_i(v_A) for cooperative game v_A(T) = R * V(T; S*)."""
        N = len(task.costs)
        credits = {i: 0.0 for i in range(N)}
        A = sorted(list(assigned_set))
        m = len(A)

        if m == 0:
            return credits

        def v(coalition: set[int]) -> float:
            if task.rule == ResolutionRule.CONJUNCTIVE:
                return task.R if task.S_star.issubset(coalition) else 0.0
            else:
                return task.R if bool(task.S_star.intersection(coalition)) else 0.0

        # Combinatorial Shapley computation over subsets of A
        fact = math.factorial
        for i in A:
            val = 0.0
            others = [j for j in A if j != i]
            for s in range(len(others) + 1):
                weight = (fact(s) * fact(m - s - 1)) / fact(m)
                for subset in itertools.combinations(others, s):
                    T = set(subset)
                    marginal = v(T | {i}) - v(T)
                    if marginal != 0.0:
                        val += weight * marginal
            credits[i] = float(val)

        return credits
