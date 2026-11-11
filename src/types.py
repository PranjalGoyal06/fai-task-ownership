"""Core data structures and types for the task ownership simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ResolutionRule(Enum):
    """Task resolution rules."""
    CONJUNCTIVE = "conjunctive"  # V_and: 1[S* subseteq A] (all required owners needed)
    DISJUNCTIVE = "disjunctive"  # V_or: 1[A cap S* != empty] (at least one owner suffices)


class CreditRule(Enum):
    """Credit division rules for coalitions."""
    EQUAL_SPLIT = "equal_split"  # chi_i = R * V / |A|
    SHAPLEY = "shapley"          # chi_i = psi_i(v_A) marginal contribution


@dataclass
class Task:
    """A task/issue arriving at the system."""
    task_id: int
    t_embedding: np.ndarray      # shape: (d,), unit norm issue vector
    root_module: int             # root-cause seed module s_0
    S_star: set[int]             # latent required set S* of size k
    z: np.ndarray                # binary indicator vector shape (N,)
    R: float                     # resolution reward R > 0
    costs: np.ndarray            # social cost vector c_i of shape (N,)
    rule: ResolutionRule         # conjunctive or disjunctive


@dataclass
class EvidencePayload:
    """Verifiable artifact payload attached to an evidence report.
    
    is_tampered_ground_truth is private to the simulator / agent state,
    and is NEVER exposed directly to the selection mechanism.
    """
    claimed_signal: float
    is_tampered_ground_truth: bool = False


@dataclass
class Evidence:
    """Private evidence observed by an individual agent."""
    agent_id: int
    raw_signal: float | None     # sim_i + noise; None if missing (with prob gamma)
    pi_true: float               # calibrated local belief P(z_i = 1 | e_i) in [0, 1]
    pi_hat: float                # subjective estimate with calibration distortion g(pi)
    payload: EvidencePayload     # private payload before submission


@dataclass
class PublicReport:
    """What the agent actually submits to the central mechanism."""
    agent_id: int
    r: float                     # reported probability in [0, 1]
    payload: EvidencePayload     # submitted payload (mechanism inspects only via audits)


@dataclass
class MechanismConfig:
    """Configuration parameters for the assignment mechanism M."""
    score_scale: float = 1.0           # lambda >= 0 (Brier score scale)
    audit_rate: float = 0.0            # alpha in [0, 1]
    audit_detection_delta: float = 0.4 # delta per neighbor
    audit_penalty: float = 5.0         # F >= 0
    credit_rule: CreditRule = CreditRule.SHAPLEY
    cost_per_audit: float = 0.1        # c_a: compute/log inspection cost per audited agent


@dataclass
class RoundResult:
    """Outcome and economics of one task assignment round."""
    task_id: int
    assigned_set: set[int]
    resolved: bool
    V: float                           # 1.0 if resolved else 0.0
    social_cost: float                 # sum_{i in A} c_i
    social_welfare: float              # R * V - social_cost
    oracle_assigned_set: set[int]
    oracle_resolved: bool
    oracle_social_welfare: float       # W(M^circ)
    regret: float                      # W(M^circ) - W(M)
    miss_loss: float                   # R * [V(A^circ) - V(A)]
    excess_cost: float                 # c(A) - c(A^circ)
    efficiency_ratio: float            # W(M) / W(M^circ)
    brier_scores: dict[int, float]     # centered Brier scores s_tilde_i
    credits: dict[int, float]          # credit chi_i paid
    penalties: dict[int, float]        # audit penalties F paid
    audited_agents: list[int]
    contradicted_agents: list[int]
    total_incentive_cost: float        # lambda * sum(s_tilde) + c_a * alpha * N
    agent_utilities: dict[int, float]  # U_i realized utility
