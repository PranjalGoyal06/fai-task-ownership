"""Configurable simulation runner, parameter definitions, and data export for task ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable
import math
import numpy as np
import pandas as pd

from src.agents import StrategicAgent
from src.baselines import DescriptionRouterB1, PollingRouterB2, VotingDeciderB3
from src.environment import RepositoryEnvironment, TaskGenerator
from src.evidence import EvidenceEngine
from src.mechanisms import AssignmentMechanism
from src.metrics import MetricsEvaluator
from src.types import CreditRule, MechanismConfig, ResolutionRule, Task


class SimulationCancelled(RuntimeError):
    """Raised when a user requests cancellation between simulation runs."""


@dataclass
class SimulationConfig:
    """Complete parameter specification for the multiagent task ownership system."""

    # 1. General & Simulation
    num_rounds: int = 100
    num_agents: int = 12
    seed: int = 42

    # 2. Agent Population & Misalignment Regime
    agent_regime: str = "mixed"  # "mixed", "deflection", "acquisition", "custom"
    deflector_ratio: float = 0.3
    acquirer_ratio: float = 0.3
    neutral_ratio: float = 0.4
    deflector_ell: float = 2.0
    acquirer_ell: float = -2.0
    neutral_ell: float = 0.0

    # 3. Dynamic Learning Mode
    dynamic_mode: bool = False
    beta_load: float = 0.5
    load_decay: float = 0.6
    alpha_credit_ema: float = 0.1

    # 4. Task Structure & Resolution
    rule: ResolutionRule = ResolutionRule.CONJUNCTIVE
    coalition_size_k: int = 5
    use_coalition_range: bool = True
    coalition_min: int = 1
    coalition_max: int = 5
    reward_per_owner: float = 5.0
    resolution_reward: float = 10.0
    default_cost: float = 1.0
    cluster_prob: float = 0.7
    hop_decay: float = 0.5
    min_member_weight: float = 0.4
    task_noise_sigma: float = 0.2

    # 5. Repository Environment & Topology
    embedding_dim: int = 32
    graph_type: str = "scale_free"  # "scale_free", "erdos_renyi", "complete", "empty"
    avg_degree: int = 4

    # 6. Private Evidence & Calibration
    obs_noise_sigma: float = 0.15
    missing_evidence_prob: float = 0.0
    signal_sensitivity: float = 8.0
    use_auto_midpoint: bool = True
    signal_midpoint_override: float = 0.23
    calibration_kappa: float = 1.0

    # 7. Mechanism M Parameters
    score_scale: float = 2.5
    audit_rate: float = 0.2
    audit_penalty: float = 10.0
    audit_detection_delta: float = 0.4
    cost_per_audit: float = 0.1
    credit_rule: CreditRule = CreditRule.SHAPLEY

    # 8. Baseline Specifics
    b1_staleness_rho: float = 0.3
    b1_desc_noise_sigma: float = 0.25
    b3_peer_noise_sigma: float = 0.25

    # 9. Strategic Agent Internals (previously hardcoded constants)
    agent_theta0: float = -2.0
    agent_theta1: float = 4.0
    agent_online_lr: float = 0.05
    agent_evasion_benefit_factor: float = 0.5
    agent_tamper_deflect_signal: float = -0.5
    agent_tamper_acquire_signal: float = 1.0
    agent_fp_iterations: int = 8

    def __post_init__(self) -> None:
        """Fail fast when configuration values violate model domains."""
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 1 <= self.num_rounds:
            raise ValueError("num_rounds must be at least 1")
        if not 1 <= self.num_agents <= 100:
            raise ValueError("num_agents must be between 1 and 100")
        if not 0.0 <= self.deflector_ratio <= 1.0:
            raise ValueError("deflector_ratio must be between 0 and 1")
        if not 0.0 <= self.acquirer_ratio <= 1.0:
            raise ValueError("acquirer_ratio must be between 0 and 1")
        if self.deflector_ratio + self.acquirer_ratio > 1.0:
            raise ValueError("deflector_ratio + acquirer_ratio must not exceed 1")
        if self.deflector_ell < 0.0:
            raise ValueError("deflector_ell must be non-negative")
        if self.acquirer_ell > 0.0:
            raise ValueError("acquirer_ell must be non-positive")
        if self.reward_per_owner < 0.0 or self.resolution_reward < 0.0:
            raise ValueError("task rewards must be non-negative")
        if self.default_cost < 0.0:
            raise ValueError("default_cost must be non-negative")
        if self.task_noise_sigma < 0.0:
            raise ValueError("task_noise_sigma must be non-negative")
        if not 0.0 <= self.audit_rate <= 1.0:
            raise ValueError("audit_rate must be between 0 and 1")
        if not 0.0 <= self.audit_detection_delta <= 1.0:
            raise ValueError("audit_detection_delta must be between 0 and 1")
        if self.audit_penalty < 0.0:
            raise ValueError("audit_penalty must be non-negative")
        if self.cost_per_audit < 0.0:
            raise ValueError("cost_per_audit must be non-negative")
        if not 0.0 <= self.missing_evidence_prob <= 1.0:
            raise ValueError("missing_evidence_prob must be between 0 and 1")
        if self.obs_noise_sigma < 0.0:
            raise ValueError("obs_noise_sigma must be non-negative")
        if self.b1_desc_noise_sigma < 0.0 or self.b3_peer_noise_sigma < 0.0:
            raise ValueError("baseline noise values must be non-negative")
        if self.agent_online_lr < 0.0:
            raise ValueError("agent_online_lr must be non-negative")
        if not 0.0 <= self.score_scale:
            raise ValueError("score_scale must be non-negative")
        if not 0.0 <= self.cluster_prob <= 1.0:
            raise ValueError("cluster_prob must be between 0 and 1")
        if not 0.0 <= self.hop_decay <= 1.0:
            raise ValueError("hop_decay must be between 0 and 1")
        if not 0.0 < self.calibration_kappa:
            raise ValueError("calibration_kappa must be positive")
        if not 0.0 <= self.b1_staleness_rho <= 1.0:
            raise ValueError("b1_staleness_rho must be between 0 and 1")
        if self.use_coalition_range:
            if not 1 <= self.coalition_min <= self.coalition_max:
                raise ValueError("coalition bounds must satisfy 1 <= min <= max <= num_agents")
            self.coalition_min = min(self.coalition_min, self.num_agents)
            self.coalition_max = min(self.coalition_max, self.num_agents)
        elif not 1 <= self.coalition_size_k <= self.num_agents:
            raise ValueError("coalition_size_k must be between 1 and num_agents")

    @property
    def coalition_size_range(self) -> tuple[int, int] | None:
        if self.use_coalition_range:
            return (self.coalition_min, self.coalition_max)
        return None


# Dictionary of comprehensive metadata for every parameter
PARAMETER_METADATA: dict[str, dict[str, str]] = {
    # General
    "num_rounds": {
        "symbol": "T",
        "name": "Number of Rounds",
        "category": "General & Simulation",
        "desc": "Total number of sequential task simulation rounds to execute.",
    },
    "num_agents": {
        "symbol": "N",
        "name": "Number of Agents",
        "category": "General & Simulation",
        "desc": "Total number of candidate agents/modules in the repository system.",
    },
    "seed": {
        "symbol": "s",
        "name": "Random Seed",
        "category": "General & Simulation",
        "desc": "Master random seed ensuring deterministic reproducibility across runs.",
    },
    # Regime
    "agent_regime": {
        "symbol": "Regime",
        "name": "Misalignment Regime",
        "category": "Population & Misalignment",
        "desc": "Population distribution type: mixed, deflection-heavy, acquisition-heavy, or custom.",
    },
    "deflector_ratio": {
        "symbol": "ϕ_defl",
        "name": "Deflector Ratio",
        "category": "Population & Misalignment",
        "desc": "Fraction of population that avoids task ownership (net positive cost ℓ > 0).",
    },
    "acquirer_ratio": {
        "symbol": "ϕ_acq",
        "name": "Acquirer Ratio",
        "category": "Population & Misalignment",
        "desc": "Fraction of population that aggressively claims tasks (net negative cost ℓ < 0).",
    },
    "neutral_ratio": {
        "symbol": "ϕ_aligned",
        "name": "Aligned Ratio",
        "category": "Population & Misalignment",
        "desc": "Fraction of population aligned with the social welfare objective (ℓ = 0).",
    },
    "deflector_ell": {
        "symbol": "ℓ_defl",
        "name": "Deflector Bias Magnitude",
        "category": "Population & Misalignment",
        "desc": "Private net cost incurred when a deflector agent is assigned work.",
    },
    "acquirer_ell": {
        "symbol": "ℓ_acq",
        "name": "Acquirer Bias Magnitude",
        "category": "Population & Misalignment",
        "desc": "Private net benefit gained when an acquirer agent hoards task assignment.",
    },
    # Dynamic Mode
    "dynamic_mode": {
        "symbol": "Mode 2",
        "name": "Dynamic Workload & Learning",
        "category": "Dynamic Learning",
        "desc": "Enable online learning: agents accumulate fatigue and learn assignment curves over rounds.",
    },
    "beta_load": {
        "symbol": "β_load",
        "name": "Fatigue Sensitivity",
        "category": "Dynamic Learning",
        "desc": "Sensitivity of dynamic private cost ℓ_i(t) to recent workload accumulation.",
    },
    "load_decay": {
        "symbol": "γ_load",
        "name": "Workload Decay Factor",
        "category": "Dynamic Learning",
        "desc": "Per-round memory decay rate for accumulated task fatigue.",
    },
    "alpha_credit_ema": {
        "symbol": "α_ema",
        "name": "Credit EMA Smoothing",
        "category": "Dynamic Learning",
        "desc": "Exponential moving average factor for tracking expected historical credit payouts.",
    },
    # Task Structure
    "rule": {
        "symbol": "V(A; S*)",
        "name": "Resolution Rule",
        "category": "Task & Value Structure",
        "desc": "Resolution logic: Conjunctive requires ALL required owners; Disjunctive requires AT LEAST ONE.",
    },
    "coalition_size_k": {
        "symbol": "k",
        "name": "Base Coalition Size",
        "category": "Task & Value Structure",
        "desc": "Target size of latent required owner set S* when not using a stochastic range.",
    },
    "use_coalition_range": {
        "symbol": "k ∈ [k_min, k_max]",
        "name": "Stochastic Coalition Size",
        "category": "Task & Value Structure",
        "desc": "Whether required owner coalition size varies randomly across incoming tasks.",
    },
    "coalition_min": {
        "symbol": "k_min",
        "name": "Min Coalition Size",
        "category": "Task & Value Structure",
        "desc": "Lower bound of required owner set size per task.",
    },
    "coalition_max": {
        "symbol": "k_max",
        "name": "Max Coalition Size",
        "category": "Task & Value Structure",
        "desc": "Upper bound of required owner set size per task.",
    },
    "reward_per_owner": {
        "symbol": "R_k",
        "name": "Reward per Owner",
        "category": "Task & Value Structure",
        "desc": "Resolution reward scaled dynamically as R = R_k * |S*|.",
    },
    "resolution_reward": {
        "symbol": "R",
        "name": "Fixed Resolution Reward",
        "category": "Task & Value Structure",
        "desc": "Flat social reward paid when a task is successfully resolved (if R_k is disabled).",
    },
    "default_cost": {
        "symbol": "c_i",
        "name": "Agent Execution Cost",
        "category": "Task & Value Structure",
        "desc": "Social operating cost incurred per assigned agent.",
    },
    "cluster_prob": {
        "symbol": "P_cluster",
        "name": "Graph Clustering Prob",
        "category": "Task & Value Structure",
        "desc": "Probability that multi-owner task sets S* cluster along dependency graph neighbors.",
    },
    "hop_decay": {
        "symbol": "η_hop",
        "name": "Hop Influence Decay",
        "category": "Task & Value Structure",
        "desc": "Signal decay rate per graph hop distance from the root-cause seed module.",
    },
    "min_member_weight": {
        "symbol": "w_min",
        "name": "Min Co-Owner Weight",
        "category": "Task & Value Structure",
        "desc": "Floor weight for co-owners in issue embedding synthesis.",
    },
    "task_noise_sigma": {
        "symbol": "σ_task",
        "name": "Task Embedding Noise",
        "category": "Task & Value Structure",
        "desc": "Gaussian noise standard deviation added to synthesized task issue embeddings.",
    },
    # Environment
    "embedding_dim": {
        "symbol": "d",
        "name": "Embedding Dimension",
        "category": "Repository Environment",
        "desc": "Vector dimension for module capability profiles and issue text embeddings.",
    },
    "graph_type": {
        "symbol": "G_type",
        "name": "Graph Topology",
        "category": "Repository Environment",
        "desc": "Architecture topology of the software dependency graph: scale-free, Erdos-Renyi, complete, or empty.",
    },
    "avg_degree": {
        "symbol": "<m>",
        "name": "Average Node Degree",
        "category": "Repository Environment",
        "desc": "Target average node degree in the generated software dependency graph.",
    },
    # Evidence & Calibration
    "obs_noise_sigma": {
        "symbol": "σ_obs",
        "name": "Observation Noise",
        "category": "Evidence & Calibration",
        "desc": "Gaussian noise variance added to local private evidence observations.",
    },
    "missing_evidence_prob": {
        "symbol": "γ",
        "name": "Missing Evidence Rate",
        "category": "Evidence & Calibration",
        "desc": "Probability an agent receives no private signal and must rely on the prior base rate.",
    },
    "signal_sensitivity": {
        "symbol": "β_1",
        "name": "Signal Sensitivity",
        "category": "Evidence & Calibration",
        "desc": "Bayesian log-likelihood slope translating raw cosine similarity into calibrated log-odds.",
    },
    "signal_midpoint_override": {
        "symbol": "μ_mid",
        "name": "Signal Midpoint",
        "category": "Evidence & Calibration",
        "desc": "Similarity midpoint separating positive from negative owner evidence.",
    },
    "calibration_kappa": {
        "symbol": "κ",
        "name": "Calibration Exponent",
        "category": "Evidence & Calibration",
        "desc": "Cognitive distortion: κ=1 is calibrated, κ>1 is overconfident, κ<1 is underconfident.",
    },
    # Mechanism M
    "score_scale": {
        "symbol": "λ",
        "name": "Brier Score Scale",
        "category": "Mechanism M",
        "desc": "Ex-post proper scoring weight rewarding truthfulness and penalizing misreports.",
    },
    "audit_rate": {
        "symbol": "α",
        "name": "Audit Rate",
        "category": "Mechanism M",
        "desc": "Probability that the mechanism conducts an evidence cross-check audit on an agent.",
    },
    "audit_penalty": {
        "symbol": "F",
        "name": "Audit Penalty",
        "category": "Mechanism M",
        "desc": "Penalty imposed on agents caught submitting contradictory or tampered payloads.",
    },
    "audit_detection_delta": {
        "symbol": "δ",
        "name": "Per-Neighbor Detection Rate",
        "category": "Mechanism M",
        "desc": "Probability a single dependency neighbor detects an altered or fabricated payload.",
    },
    "cost_per_audit": {
        "symbol": "c_a",
        "name": "Cost per Audit",
        "category": "Mechanism M",
        "desc": "Administrative/computational cost incurred by the mechanism per audited agent.",
    },
    "credit_rule": {
        "symbol": "χ_i",
        "name": "Credit Division Rule",
        "category": "Mechanism M",
        "desc": "Credit allocation rule: Shapley value rewards marginal contribution; Equal Split divides evenly.",
    },
    # Baselines
    "b1_staleness_rho": {
        "symbol": "ρ",
        "name": "B1 Description Staleness",
        "category": "Baselines",
        "desc": "Fraction of agent capability profiles in central documentation that are outdated.",
    },
    "b1_desc_noise_sigma": {
        "symbol": "σ_desc",
        "name": "B1 Description Noise",
        "category": "Baselines",
        "desc": "Noise standard deviation in central description-based similarity evaluation.",
    },
    "b3_peer_noise_sigma": {
        "symbol": "σ_peer",
        "name": "B3 Voting Peer Noise",
        "category": "Baselines",
        "desc": "Perception noise when agents estimate peer capabilities during B3 voting.",
    },
    # Agent Internals
    "agent_theta0": {
        "symbol": "θ_0",
        "name": "Assignment Model Intercept",
        "category": "Agent Internals",
        "desc": "Initial intercept for the agent's internal logistic model of assignment chance X(r).",
    },
    "agent_theta1": {
        "symbol": "θ_1",
        "name": "Assignment Model Slope",
        "category": "Agent Internals",
        "desc": "Initial slope for the agent's internal logistic model of assignment chance X(r).",
    },
    "agent_online_lr": {
        "symbol": "lr",
        "name": "Assignment Learning Rate",
        "category": "Agent Internals",
        "desc": "Learning rate for the agent's online logistic regression tracking assignment outcomes.",
    },
    "agent_evasion_benefit_factor": {
        "symbol": "k_evasion",
        "name": "Evasion Benefit Factor",
        "category": "Agent Internals",
        "desc": "Scaling coefficient determining private gain from fabricating evidence payloads.",
    },
    "agent_tamper_deflect_signal": {
        "symbol": "s_defl",
        "name": "Forged Deflection Signal",
        "category": "Agent Internals",
        "desc": "Fabricated raw signal claimed by deflecting agents to disclaim ownership.",
    },
    "agent_tamper_acquire_signal": {
        "symbol": "s_acq",
        "name": "Forged Acquisition Signal",
        "category": "Agent Internals",
        "desc": "Fabricated raw signal claimed by acquiring agents to hoard ownership.",
    },
    "agent_fp_iterations": {
        "symbol": "n_fp",
        "name": "Fixed-Point Iterations",
        "category": "Agent Internals",
        "desc": "Number of damped fixed-point iterations used by strategic agents to compute shaded report r*.",
    },
}


def init_agent_population(
    config: SimulationConfig,
    beta_load: float,
    seed: int,
) -> dict[int, StrategicAgent]:
    """Instantiates a population of StrategicAgent instances according to config."""
    num_agents = config.num_agents

    if config.agent_regime == "deflection":
        n_deflect = num_agents // 2
        n_acquire = 0
        n_neutral = num_agents - n_deflect
        ells = [config.deflector_ell] * n_deflect + [config.neutral_ell] * n_neutral
    elif config.agent_regime == "acquisition":
        n_deflect = 0
        n_acquire = num_agents // 2
        n_neutral = num_agents - n_acquire
        ells = [config.acquirer_ell] * n_acquire + [config.neutral_ell] * n_neutral
    elif config.agent_regime == "custom":
        n_deflect = int(round(config.deflector_ratio * num_agents))
        n_acquire = int(round(config.acquirer_ratio * num_agents))
        n_neutral = max(0, num_agents - n_deflect - n_acquire)
        ells = (
            [config.deflector_ell] * n_deflect
            + [config.acquirer_ell] * n_acquire
            + [config.neutral_ell] * n_neutral
        )
        if len(ells) < num_agents:
            ells += [config.neutral_ell] * (num_agents - len(ells))
        ells = ells[:num_agents]
    else:  # "mixed" default
        n_deflect = int(round(0.3 * num_agents))
        n_acquire = int(round(0.3 * num_agents))
        n_neutral = max(0, num_agents - n_deflect - n_acquire)
        ells = (
            [config.deflector_ell] * n_deflect
            + [config.acquirer_ell] * n_acquire
            + [config.neutral_ell] * n_neutral
        )
        if len(ells) < num_agents:
            ells += [config.neutral_ell] * (num_agents - len(ells))
        ells = ells[:num_agents]

    np.random.default_rng(seed).shuffle(ells)

    agents: dict[int, StrategicAgent] = {}
    for i in range(num_agents):
        agents[i] = StrategicAgent(
            agent_id=i,
            base_ell=float(ells[i]),
            beta_load=beta_load,
            alpha_credit_ema=config.alpha_credit_ema,
            load_decay=config.load_decay,
            seed=seed + i,
            theta0=config.agent_theta0,
            theta1=config.agent_theta1,
            online_lr=config.agent_online_lr,
            evasion_benefit_factor=config.agent_evasion_benefit_factor,
            tamper_deflect_signal=config.agent_tamper_deflect_signal,
            tamper_acquire_signal=config.agent_tamper_acquire_signal,
            fp_iterations=config.agent_fp_iterations,
        )
    return agents


@dataclass
class SimulationResults:
    """Complete simulation outputs for analysis and visualization."""

    config: SimulationConfig
    summary_df: pd.DataFrame
    rounds_df: pd.DataFrame
    agents_df: pd.DataFrame
    raw_trackers: dict[str, Any]
    task_history: list[dict[str, Any]]
    environment: RepositoryEnvironment


def run_full_simulation(
    config: SimulationConfig,
    progress_callback: Callable[[float, str], None] | None = None,
) -> SimulationResults:
    """Executes the complete multi-round simulation and compiles rich metrics."""
    seed = config.seed
    rng = np.random.default_rng(seed)
    num_agents = config.num_agents
    num_rounds = config.num_rounds

    if progress_callback:
        progress_callback(0.05, "Initializing Repository Environment & Dependency Graph...")

    # 1. Environment & Graph
    env = RepositoryEnvironment(
        num_agents=num_agents,
        embedding_dim=config.embedding_dim,
        graph_type=config.graph_type,
        avg_degree=config.avg_degree,
        seed=seed,
    )

    # 2. Task Generator
    task_gen = TaskGenerator(
        env=env,
        coalition_size_k=config.coalition_size_k,
        coalition_size_range=config.coalition_size_range,
        cluster_prob=config.cluster_prob,
        hop_decay=config.hop_decay,
        min_member_weight=config.min_member_weight,
        task_noise_sigma=config.task_noise_sigma,
        resolution_reward=config.resolution_reward,
        reward_per_owner=config.reward_per_owner,
        default_cost=config.default_cost,
        rule=config.rule,
        seed=seed,
    )

    # 3. Evidence Engine
    expected_k = (
        0.5 * (config.coalition_min + min(config.coalition_max, num_agents))
        if config.use_coalition_range
        else float(config.coalition_size_k)
    )
    base_rate_pi0 = expected_k / num_agents

    midpoint = (
        float(0.23 * np.sqrt(2.0 / max(1.0, expected_k)))
        if config.use_auto_midpoint
        else config.signal_midpoint_override
    )

    evidence_eng = EvidenceEngine(
        env=env,
        obs_noise_sigma=config.obs_noise_sigma,
        missing_evidence_prob=config.missing_evidence_prob,
        signal_sensitivity=config.signal_sensitivity,
        signal_midpoint=None if config.use_auto_midpoint else config.signal_midpoint_override,
        prior_pi0=base_rate_pi0,
        seed=seed,
    )

    calibration_kappas = np.full(num_agents, config.calibration_kappa, dtype=float)

    # 4. Mechanism M Configuration
    mech_active = AssignmentMechanism(
        MechanismConfig(
            score_scale=config.score_scale,
            audit_rate=config.audit_rate,
            audit_detection_delta=config.audit_detection_delta,
            audit_penalty=config.audit_penalty,
            credit_rule=config.credit_rule,
            cost_per_audit=config.cost_per_audit,
        )
    )
    config_no_inc = MechanismConfig(score_scale=0.0, audit_rate=0.0)

    # 5. Baselines
    b1_router = DescriptionRouterB1(
        env_profiles=env.profiles,
        staleness_rho=config.b1_staleness_rho,
        desc_noise_sigma=config.b1_desc_noise_sigma,
        signal_sensitivity=config.signal_sensitivity,
        signal_midpoint=midpoint,
        prior_pi0=base_rate_pi0,
        seed=seed,
    )
    b2_polling = PollingRouterB2()
    b3_voting = VotingDeciderB3(
        env_profiles=env.profiles,
        peer_noise_sigma=config.b3_peer_noise_sigma,
        signal_sensitivity=config.signal_sensitivity,
        signal_midpoint=midpoint,
        prior_pi0=base_rate_pi0,
        seed=seed,
    )

    # 6. Agents
    beta_load = config.beta_load if config.dynamic_mode else 0.0
    agents_M = init_agent_population(config, beta_load=beta_load, seed=seed)
    agents_B2 = init_agent_population(config, beta_load=beta_load, seed=seed)

    system_names = [
        "Oracle (B4)",
        "Description (B1)",
        "Polling Router (B2)",
        "Voting (B3)",
        "Mechanism M",
    ]

    trackers: dict[str, dict[str, list[float]]] = {
        name: {
            "welfare": [],
            "regret": [],
            "miss_loss": [],
            "excess_cost": [],
            "resolved": [],
            "cost": [],
            "efficiency": [],
        }
        for name in system_names
    }
    trackers["Mechanism M"]["inc_cost"] = []
    trackers["Mechanism M"]["audit_cost"] = []
    trackers["Mechanism M"]["tampering_rate"] = []

    # Per-round history dataframe accumulators
    round_rows: list[dict[str, Any]] = []
    task_history: list[dict[str, Any]] = []

    # Agent level tracking
    agent_reports_M: dict[int, list[float]] = {i: [] for i in range(num_agents)}
    agent_reports_B2: dict[int, list[float]] = {i: [] for i in range(num_agents)}
    agent_pi_true: dict[int, list[float]] = {i: [] for i in range(num_agents)}
    agent_assigned_M: dict[int, int] = {i: 0 for i in range(num_agents)}
    agent_assigned_B2: dict[int, int] = {i: 0 for i in range(num_agents)}
    agent_tampering_count: dict[int, int] = {i: 0 for i in range(num_agents)}

    # -------------------------------------------------------------------------
    # Simulation Loop
    # -------------------------------------------------------------------------
    for round_id in range(num_rounds):
        if progress_callback and round_id % max(1, num_rounds // 20) == 0:
            frac = 0.1 + 0.85 * (round_id / num_rounds)
            progress_callback(frac, f"Simulating Round {round_id + 1} / {num_rounds}...")

        task = task_gen.generate_task(task_id=round_id)
        evidence_dict = evidence_eng.generate_evidence(task, calibration_kappas=calibration_kappas)

        # 1. Oracle benchmark
        A_O, o_res, o_cost, o_welfare = MetricsEvaluator.compute_oracle(task)
        trackers["Oracle (B4)"]["welfare"].append(o_welfare)
        trackers["Oracle (B4)"]["regret"].append(0.0)
        trackers["Oracle (B4)"]["miss_loss"].append(0.0)
        trackers["Oracle (B4)"]["excess_cost"].append(0.0)
        trackers["Oracle (B4)"]["resolved"].append(1.0 if o_res else 0.0)
        trackers["Oracle (B4)"]["cost"].append(o_cost)
        trackers["Oracle (B4)"]["efficiency"].append(1.0 if o_welfare > 0 else 0.0)

        # 2. Reports under Mechanism M (with incentives)
        reports_M = {}
        tampered_M = {}
        for i in range(num_agents):
            rep_m, is_tamp_m = agents_M[i].compute_report(
                evidence_dict[i],
                mech_active.config,
                degree_m=env.get_degree(i),
            )
            reports_M[i] = rep_m
            tampered_M[i] = is_tamp_m

        # 3. Reports under Polling Router B2 (no incentives)
        reports_B2 = {}
        tampered_B2 = {}
        for i in range(num_agents):
            rep_b2, is_tamp_b2 = agents_B2[i].compute_report(
                evidence_dict[i],
                config_no_inc,
                degree_m=env.get_degree(i),
            )
            reports_B2[i] = rep_b2
            tampered_B2[i] = is_tamp_b2

        # Record agent metrics
        for i in range(num_agents):
            agent_reports_M[i].append(reports_M[i].r)
            agent_reports_B2[i].append(reports_B2[i].r)
            agent_pi_true[i].append(evidence_dict[i].pi_true)
            if tampered_M[i]:
                agent_tampering_count[i] += 1

        # 4. Agent Selections across systems
        A_M = mech_active.select_agents(task, reports_M, rng=rng)
        A_B1 = b1_router.select(task)
        A_B2 = b2_polling.select(task, reports_B2)
        A_B3 = b3_voting.select(task, reports_B2)
        for i in A_M:
            agent_assigned_M[i] += 1
        for i in A_B2:
            agent_assigned_B2[i] += 1

        # 5. Regret Decompositions
        reg_B1 = MetricsEvaluator.compute_regret_decomposition(task, A_B1)
        reg_B2 = MetricsEvaluator.compute_regret_decomposition(task, A_B2)
        reg_B3 = MetricsEvaluator.compute_regret_decomposition(task, A_B3)
        reg_M = MetricsEvaluator.compute_regret_decomposition(task, A_M)

        # 6. Audits & Scoring for Mechanism M
        penalties, audited_agents, contradicted_agents, audit_cost = mech_active.execute_audits(
            reports_M, env.G, rng, tampered_flags=tampered_M
        )
        scores = mech_active.compute_all_scores(reports_M, task.z, base_rate_pi0)
        credits = mech_active.divide_credit(task, A_M)
        total_inc_cost = sum(scores.values()) + audit_cost
        inc_summary = MetricsEvaluator.compute_incentive_summary(
            task, reports_M, scores, credits, penalties, audit_cost, tampered_flags=tampered_M
        )

        trackers["Mechanism M"]["inc_cost"].append(total_inc_cost)
        trackers["Mechanism M"]["audit_cost"].append(audit_cost)
        trackers["Mechanism M"]["tampering_rate"].append(inc_summary["tampering_rate"])

        # 7. Update Agent dynamic states (record_round_outcome called unconditionally so credit_ema tracks expected credit in static and dynamic modes)
        for i in range(num_agents):
            agents_M[i].record_round_outcome(
                credit_earned=credits[i],
                penalty_incurred=penalties[i],
                was_assigned=(i in A_M),
                task_resolved=bool(reg_M["resolved"]),
            )
            agents_M[i].update_assignment_model(reports_M[i].r, was_assigned=(i in A_M))

        for i in range(num_agents):
            agents_B2[i].record_round_outcome(
                credit_earned=0.0,
                penalty_incurred=0.0,
                was_assigned=(i in A_B2),
                task_resolved=bool(reg_B2["resolved"]),
            )
            agents_B2[i].update_assignment_model(reports_B2[i].r, was_assigned=(i in A_B2))
            agents_B2[i].update_assignment_model(reports_B2[i].r, was_assigned=(i in A_B2))

        # Accumulate metrics
        eval_pairs = [
            ("Description (B1)", reg_B1, A_B1),
            ("Polling Router (B2)", reg_B2, A_B2),
            ("Voting (B3)", reg_B3, A_B3),
            ("Mechanism M", reg_M, A_M),
        ]
        for name, res, _ in eval_pairs:
            trackers[name]["welfare"].append(res["social_welfare"])
            trackers[name]["regret"].append(res["regret"])
            trackers[name]["miss_loss"].append(res["miss_loss"])
            trackers[name]["excess_cost"].append(res["excess_cost"])
            trackers[name]["resolved"].append(res["resolved"])
            trackers[name]["cost"].append(res["social_cost"])
            trackers[name]["efficiency"].append(res["efficiency_ratio"])

        # Accumulate per-round row for timeseries
        for name in system_names:
            round_rows.append(
                {
                    "round": round_id + 1,
                    "system": name,
                    "welfare": trackers[name]["welfare"][-1],
                    "regret": trackers[name]["regret"][-1],
                    "miss_loss": trackers[name]["miss_loss"][-1],
                    "excess_cost": trackers[name]["excess_cost"][-1],
                    "resolved": trackers[name]["resolved"][-1],
                    "cost": trackers[name]["cost"][-1],
                }
            )

        # Retain task snapshot for interactive inspection
        task_history.append(
            {
                "round_id": round_id,
                "task_id": task.task_id,
                "root_module": task.root_module,
                "S_star": sorted(list(task.S_star)),
                "k": len(task.S_star),
                "R": task.R,
                "t_embedding": task.t_embedding,
                "reports_M": {i: reports_M[i].r for i in range(num_agents)},
                "reports_B2": {i: reports_B2[i].r for i in range(num_agents)},
                "pi_true": {i: evidence_dict[i].pi_true for i in range(num_agents)},
                "raw_signals": {
                    i: (evidence_dict[i].raw_signal if evidence_dict[i].raw_signal is not None else -999.0)
                    for i in range(num_agents)
                },
                "A_Oracle": sorted(list(A_O)),
                "A_B1": sorted(list(A_B1)),
                "A_B2": sorted(list(A_B2)),
                "A_B3": sorted(list(A_B3)),
                "A_Mechanism": sorted(list(A_M)),
                "res_Mechanism": bool(reg_M["resolved"]),
                "welfare_Mechanism": reg_M["social_welfare"],
                "regret_Mechanism": reg_M["regret"],
                "audited": audited_agents,
                "contradicted": contradicted_agents,
            }
        )

    if progress_callback:
        progress_callback(0.95, "Compiling summary tables and time-series dataframes...")

    # Build Summary Table (exclude Voting B3 for conjunctive tasks where plurality single-choice is structurally inapplicable)
    display_systems = [
        name
        for name in system_names
        if not (config.rule == ResolutionRule.CONJUNCTIVE and name == "Voting (B3)")
    ]
    summary_data = []
    for name in display_systems:
        w_mean = float(np.mean(trackers[name]["welfare"]))
        reg_mean = float(np.mean(trackers[name]["regret"]))
        miss_mean = float(np.mean(trackers[name]["miss_loss"]))
        exc_mean = float(np.mean(trackers[name]["excess_cost"]))
        res_mean = float(np.mean(trackers[name]["resolved"]))
        cost_mean = float(np.mean(trackers[name]["cost"]))
        eff_mean = float(np.mean(trackers[name]["efficiency"]))

        inc_cost_mean = (
            float(np.mean(trackers["Mechanism M"]["inc_cost"])) if name == "Mechanism M" else 0.0
        )
        net_welfare = w_mean - inc_cost_mean

        summary_data.append(
            {
                "System": name,
                "Welfare (Gross)": round(w_mean, 2),
                "Regret": round(reg_mean, 2),
                "Miss Loss (ΔV)": round(miss_mean, 2),
                "Excess Cost (ΔC)": round(exc_mean, 2),
                "Resolution Rate": round(res_mean * 100, 1),
                "Avg Social Cost": round(cost_mean, 2),
                "Efficiency Ratio (η)": round(eff_mean, 3),
                "Incentive Overhead": round(inc_cost_mean, 2),
                "Net Welfare": round(net_welfare, 2),
            }
        )
    summary_df = pd.DataFrame(summary_data)

    rounds_df = pd.DataFrame(round_rows)
    rounds_df = rounds_df[rounds_df["system"].isin(display_systems)].copy()
    # Add cumulative metrics per system
    rounds_df["cum_welfare"] = rounds_df.groupby("system")["welfare"].cumsum()
    rounds_df["cum_regret"] = rounds_df.groupby("system")["regret"].cumsum()
    rounds_df["running_mean_welfare"] = (
        rounds_df["cum_welfare"] / rounds_df["round"]
    )
    rounds_df["running_mean_regret"] = (
        rounds_df["cum_regret"] / rounds_df["round"]
    )

    # Build Agent level dataframe
    agent_rows = []
    for i in range(num_agents):
        base_ell = agents_M[i].base_ell
        agent_type = (
            "Deflector (ℓ > 0)"
            if base_ell > 0.05
            else ("Acquirer (ℓ < 0)" if base_ell < -0.05 else "Aligned (ℓ = 0)")
        )
        mean_r_m = float(np.mean(agent_reports_M[i]))
        mean_r_b2 = float(np.mean(agent_reports_B2[i]))
        mean_pi = float(np.mean(agent_pi_true[i]))

        agent_rows.append(
            {
                "Agent ID": f"Agent {i}",
                "Base ℓ": round(base_ell, 2),
                "Regime Type": agent_type,
                "Mean True Belief (π_true)": round(mean_pi, 3),
                "Mean Report M (with λ)": round(mean_r_m, 3),
                "Mean Report B2 (no inc)": round(mean_r_b2, 3),
                "Shading M (r_M - π)": round(mean_r_m - mean_pi, 3),
                "Shading B2 (r_B2 - π)": round(mean_r_b2 - mean_pi, 3),
                "Assigned in M": agent_assigned_M[i],
                "Assigned in B2": agent_assigned_B2[i],
                "Total Credits M": round(agents_M[i].total_credits, 2),
                "Total Penalties M": round(agents_M[i].total_penalties, 2),
                "Tampering Count": agent_tampering_count[i],
                "Recent Fatigue Load": round(agents_M[i].recent_load, 2),
            }
        )
    agents_df = pd.DataFrame(agent_rows)

    if progress_callback:
        progress_callback(1.0, "Simulation complete!")

    return SimulationResults(
        config=config,
        summary_df=summary_df,
        rounds_df=rounds_df,
        agents_df=agents_df,
        raw_trackers=trackers,
        task_history=task_history,
        environment=env,
    )


def run_parameter_sweep(
    param_name: str,
    param_values: list[Any],
    base_config: SimulationConfig,
    progress_callback: Callable[[float, str], None] | None = None,
    num_replicates: int = 1,
) -> pd.DataFrame:
    """Executes a 1D sensitivity sweep, averaging independent seeded replicates.

    The same replicate seed is used across parameter values so each curve
    compares values under the same random draw, while different replicates
    estimate stochastic variability.
    """
    if not param_values:
        raise ValueError("param_values must contain at least one value")
    if num_replicates < 1:
        raise ValueError("num_replicates must be at least 1")

    sweep_rows = []
    total = len(param_values) * num_replicates
    completed = 0

    for idx, val in enumerate(param_values):
        replicate_rows: list[dict[str, Any]] = []
        for replicate in range(num_replicates):
            cfg_copy = SimulationConfig(**base_config.__dict__)
            setattr(cfg_copy, param_name, val)
            cfg_copy.seed = base_config.seed + replicate

            res = run_full_simulation(cfg_copy)
            summary_by_system = res.summary_df.set_index("System")
            research = _research_metrics(res)
            replicate_rows.append(
                {
                    "Mechanism Welfare": summary_by_system.at["Mechanism M", "Welfare (Gross)"],
                    "Mechanism Regret": summary_by_system.at["Mechanism M", "Regret"],
                    "Mechanism Net Welfare": summary_by_system.at["Mechanism M", "Net Welfare"],
                    "Mechanism Incentive Overhead": summary_by_system.at[
                        "Mechanism M", "Incentive Overhead"
                    ],
                    "Mechanism Miss Loss": summary_by_system.at["Mechanism M", "Miss Loss (ΔV)"],
                    "Mechanism Excess Cost": summary_by_system.at["Mechanism M", "Excess Cost (ΔC)"],
                    "Mechanism Resolution Rate": summary_by_system.at[
                        "Mechanism M", "Resolution Rate"
                    ],
                    "Oracle Welfare": summary_by_system.at["Oracle (B4)", "Welfare (Gross)"],
                    "B2 Polling Regret": summary_by_system.at["Polling Router (B2)", "Regret"],
                    "B1 Description Regret": summary_by_system.at["Description (B1)", "Regret"],
                    "Mechanism Mean Absolute Shading": research["Mean Absolute Shading M"],
                    "Mechanism Tampering Rate": research["Tampering Rate"],
                    "Mechanism Coalition Coverage": research["Mechanism Coalition Coverage"],
                    "Mechanism Coalition Padding": research["Mechanism Coalition Padding"],
                    "Mechanism ECE": research["Mechanism ECE"],
                }
            )
            completed += 1
            if progress_callback:
                progress_callback(
                    completed / total,
                    f"Sweeping {param_name} = {val}; replicate {replicate + 1}/{num_replicates} "
                    f"({completed}/{total})...",
                )

        sweep_row: dict[str, Any] = {param_name: val, "Runs": num_replicates}
        for metric_name in replicate_rows[0]:
            metric_values = [row[metric_name] for row in replicate_rows]
            sweep_row[metric_name] = float(np.mean(metric_values))
            sweep_row[f"{metric_name} Std"] = float(np.std(metric_values, ddof=1)) if num_replicates > 1 else 0.0
        sweep_rows.append(sweep_row)

    if progress_callback:
        progress_callback(1.0, f"Sweep for {param_name} complete!")

    return pd.DataFrame(sweep_rows)


def _coalition_metrics(
    task_history: list[dict[str, Any]],
    assignment_key: str,
) -> tuple[float, float]:
    """Returns mean required-owner coverage and non-owner padding for assignments."""
    if not task_history:
        return 0.0, 0.0

    coverage: list[float] = []
    padding: list[float] = []
    for task in task_history:
        required = set(task["S_star"])
        assigned = set(task[assignment_key])
        coverage.append(len(required.intersection(assigned)) / len(required) if required else 0.0)
        padding.append(float(len(assigned - required)))
    return float(np.mean(coverage)), float(np.mean(padding))


def _research_metrics(results: SimulationResults) -> dict[str, float]:
    """Extracts metrics aligned with RQ1-RQ3 from one simulation result."""
    summary = results.summary_df.set_index("System")
    b2 = summary.loc["Polling Router (B2)"]
    mechanism = summary.loc["Mechanism M"]
    reports_m: list[float] = []
    reports_b2: list[float] = []
    labels: list[int] = []
    for task in results.task_history:
        required = set(task["S_star"])
        for agent_id in task["pi_true"]:
            reports_m.append(float(task["reports_M"][agent_id]))
            reports_b2.append(float(task["reports_B2"][agent_id]))
            labels.append(int(agent_id in required))
    b2_coverage, b2_padding = _coalition_metrics(results.task_history, "A_B2")
    mechanism_coverage, mechanism_padding = _coalition_metrics(
        results.task_history, "A_Mechanism"
    )
    b2_regret = float(b2["Regret"])
    mechanism_regret = float(mechanism["Regret"])

    return {
        "B2 Welfare": float(b2["Welfare (Gross)"]),
        "B2 Regret": b2_regret,
        "B2 Miss Loss": float(b2["Miss Loss (ΔV)"]),
        "B2 Excess Cost": float(b2["Excess Cost (ΔC)"]),
        "B2 Resolution Rate": float(b2["Resolution Rate"]),
        "Mechanism Welfare": float(mechanism["Welfare (Gross)"]),
        "Mechanism Regret": mechanism_regret,
        "Mechanism Miss Loss": float(mechanism["Miss Loss (ΔV)"]),
        "Mechanism Excess Cost": float(mechanism["Excess Cost (ΔC)"]),
        "Mechanism Net Welfare": float(mechanism["Net Welfare"]),
        "Mechanism Efficiency": float(mechanism["Efficiency Ratio (η)"]),
        "Mechanism Resolution Rate": float(mechanism["Resolution Rate"]),
        "Incentive Overhead": float(mechanism["Incentive Overhead"]),
        "Regret Recovered": b2_regret - mechanism_regret,
        "Regret Recovery Fraction": (
            (b2_regret - mechanism_regret) / b2_regret if not np.isclose(b2_regret, 0.0) else 0.0
        ),
        "Mean Absolute Shading M": float(results.agents_df["Shading M (r_M - π)"].abs().mean()),
        "Mean Absolute Shading B2": float(results.agents_df["Shading B2 (r_B2 - π)"].abs().mean()),
        "Mechanism ECE": MetricsEvaluator.compute_ece(reports_m, labels),
        "B2 ECE": MetricsEvaluator.compute_ece(reports_b2, labels),
        "Tampering Rate": float(np.mean(results.raw_trackers["Mechanism M"]["tampering_rate"])),
        "B2 Coalition Coverage": b2_coverage,
        "B2 Coalition Padding": b2_padding,
        "Mechanism Coalition Coverage": mechanism_coverage,
        "Mechanism Coalition Padding": mechanism_padding,
    }


def run_parameter_grid_search(
    parameter_grid: dict[str, list[Any]],
    base_config: SimulationConfig,
    progress_callback: Callable[[float, str], None] | None = None,
    num_replicates: int = 1,
    cancel_check: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Runs every combination in a multi-parameter grid and averages replicates.

    Parameters not present in ``parameter_grid`` remain fixed at ``base_config``.
    Replicate seeds are shared across grid points to make comparisons less noisy.
    """
    if not parameter_grid:
        raise ValueError("parameter_grid must contain at least one parameter")
    if num_replicates < 1:
        raise ValueError("num_replicates must be at least 1")

    known_parameters = set(SimulationConfig.__dataclass_fields__)
    unknown_parameters = set(parameter_grid) - known_parameters
    if unknown_parameters:
        names = ", ".join(sorted(unknown_parameters))
        raise ValueError(f"Unknown simulation parameter(s): {names}")
    if any(not values for values in parameter_grid.values()):
        raise ValueError("Each grid parameter must contain at least one value")

    parameter_names = list(parameter_grid)
    grid_points = list(product(*(parameter_grid[name] for name in parameter_names)))
    total_runs = len(grid_points) * num_replicates
    completed = 0
    rows: list[dict[str, Any]] = []

    for point in grid_points:
        point_config = dict(zip(parameter_names, point))
        replicate_metrics: list[dict[str, float]] = []
        for replicate in range(num_replicates):
            if cancel_check and cancel_check():
                raise SimulationCancelled("Grid search cancelled by user.")

            config_values = dict(base_config.__dict__)
            config_values.update(point_config)
            config_values["seed"] = int(config_values["seed"]) + replicate
            config = SimulationConfig(**config_values)
            results = run_full_simulation(config)
            replicate_metrics.append(_research_metrics(results))

            completed += 1
            if progress_callback:
                progress_callback(
                    completed / total_runs,
                    f"Grid point {completed // num_replicates + (1 if completed % num_replicates else 0)}"
                    f"/{len(grid_points)}; replicate {replicate + 1}/{num_replicates}...",
                )

        row: dict[str, Any] = {**point_config, "Runs": num_replicates}
        for metric_name in replicate_metrics[0]:
            values = [metrics[metric_name] for metrics in replicate_metrics]
            row[metric_name] = float(np.mean(values))
            row[f"{metric_name} Std"] = (
                float(np.std(values, ddof=1)) if num_replicates > 1 else 0.0
            )
        rows.append(row)

    if progress_callback:
        progress_callback(1.0, f"Grid search complete: {len(grid_points)} point(s).")
    return pd.DataFrame(rows)

