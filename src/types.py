"""Core data structures and types for the task ownership simulation."""

from __future__ import annotations

from dataclasses import dataclass
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
class PublicPayload:
    """Publicly visible evidence artifact payload attached to a public report.

    Contains ONLY the claimed signal / proof artifact, strictly maintaining
    the information boundary so mechanisms and baselines cannot peek at ground truth.
    """
    claimed_signal: float


@dataclass
class EvidencePayload:
    """Verifiable artifact payload attached to private agent evidence.
    
    is_tampered_ground_truth is strictly private to the simulator / agent state,
    and is NEVER exposed directly in public reports or to the selection mechanism.
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
    payload: PublicPayload       # public payload (contains only claimed_signal)


@dataclass
class MechanismConfig:
    """Configuration parameters for the assignment mechanism M."""
    score_scale: float = 1.0           # lambda >= 0 (Brier score scale)
    audit_rate: float = 0.0            # alpha in [0, 1]
    audit_detection_delta: float = 0.4 # delta per neighbor
    audit_penalty: float = 5.0         # F >= 0
    credit_rule: CreditRule = CreditRule.SHAPLEY
    cost_per_audit: float = 0.1        # c_a: compute/log inspection cost per audited agent
