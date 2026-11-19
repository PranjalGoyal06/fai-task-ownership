"""Demonstration and benchmark comparison runner for task ownership simulation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.agents import StrategicAgent
from src.baselines import DescriptionRouterB1, PollingRouterB2, VotingDeciderB3
from src.environment import RepositoryEnvironment, TaskGenerator
from src.evidence import EvidenceEngine
from src.mechanisms import AssignmentMechanism
from src.metrics import MetricsEvaluator
from src.simulation_runner import (
    SimulationConfig,
    SimulationResults,
    _research_metrics,
    run_full_simulation,
    run_parameter_sweep,
)
from src.types import (
    CreditRule,
    MechanismConfig,
    ResolutionRule,
)


@dataclass(frozen=True)
class ExperimentCard:
    """A named, reproducible simulation scenario for the interactive UI."""

    key: str
    title: str
    explanation: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ComparisonCard:
    """A one-parameter experiment with a small, interpretable value range."""

    key: str
    title: str
    parameter: str
    values: tuple[Any, ...]
    explanation: str
    base_overrides: dict[str, Any] = field(default_factory=dict)


EXPERIMENT_CARDS: tuple[ExperimentCard, ...] = (
    ExperimentCard(
        "no_incentives",
        "No incentive layer",
        "Turns off scoring and audits to approximate the polling-router condition and establish the mechanism's untreated baseline.",
        {"score_scale": 0.0, "audit_rate": 0.0},
    ),
    ExperimentCard(
        "small_population",
        "Small population",
        "Tests whether a small candidate pool makes selection easier or leaves less redundancy when agents misreport.",
        {"num_agents": 4, "coalition_min": 1, "coalition_max": 2},
    ),
    ExperimentCard(
        "large_population",
        "Large population",
        "Tests scalability and whether more candidates increase padding, competition, or opportunities for correct coverage.",
        {"num_agents": 24, "coalition_min": 1, "coalition_max": 5},
    ),
    ExperimentCard(
        "high_deflection",
        "High deflection cost",
        "Makes owning costly for deflectors, so under-reporting should increase miss loss and reduce resolution.",
        {"agent_regime": "deflection", "deflector_ell": 6.0},
    ),
    ExperimentCard(
        "high_acquisition",
        "High acquisition incentive",
        "Makes owning attractive, so over-reporting should increase assignment padding and excess cost.",
        {"agent_regime": "acquisition", "acquirer_ell": -6.0},
    ),
    ExperimentCard(
        "high_auditing",
        "High audit rate",
        "Increases the chance that altered evidence is inspected and detected, at the cost of more audits.",
        {"audit_rate": 0.8},
    ),
    ExperimentCard(
        "weak_audit_penalty",
        "Weak audit penalty",
        "Keeps audits active but reduces their deterrence, exposing whether tampering and shading persist.",
        {"audit_rate": 0.5, "audit_penalty": 1.0},
    ),
    ExperimentCard(
        "strong_scoring",
        "Strong proper scoring",
        "Tests whether a larger Brier-score scale contracts reports toward the agents' private beliefs.",
        {"score_scale": 6.0},
    ),
    ExperimentCard(
        "missing_evidence",
        "Missing evidence stress test",
        "Tests robustness when a substantial fraction of agents lack a usable private signal.",
        {"missing_evidence_prob": 0.4},
    ),
    ExperimentCard(
        "overconfident_evidence",
        "Overconfident evidence",
        "Makes subjective probabilities overconfident while keeping the underlying evidence process unchanged.",
        {"calibration_kappa": 2.0},
    ),
    ExperimentCard(
        "stale_descriptions",
        "Fully stale descriptions",
        "Tests the proposal's claim that description-based routing degrades when capability descriptions become outdated.",
        {"b1_staleness_rho": 1.0},
    ),
    ExperimentCard(
        "dense_dependency_graph",
        "Dense dependency graph",
        "Increases graph connectivity so audits have more neighboring payloads available for cross-checking.",
        {"graph_type": "complete"},
    ),
    ExperimentCard(
        "dynamic_workload",
        "Dynamic workload and fatigue",
        "Enables the extended workload model to test whether repeated assignments change fatigue and credit learning.",
        {"dynamic_mode": True},
    ),
    ExperimentCard(
        "disjunctive_routing",
        "Disjunctive specialist routing",
        "Switches to the one-owner-suffices rule used by specialist routing, making coalition padding easier to interpret.",
        {"rule": ResolutionRule.DISJUNCTIVE, "coalition_min": 1, "coalition_max": 3},
    ),
)


COMPARISON_CARDS: tuple[ComparisonCard, ...] = (
    ComparisonCard(
        "agents_curve",
        "Population size curve",
        "num_agents",
        (4, 8, 16, 24),
        "Varying population size isolates the effect of more candidate owners while keeping the mechanism fixed.",
    ),
    ComparisonCard(
        "deflection_curve",
        "Deflection strength curve",
        "deflector_ell",
        (0.0, 1.0, 3.0, 6.0),
        "Increasing positive ℓ tests whether costly ownership produces more under-reporting and miss loss.",
    ),
    ComparisonCard(
        "audit_curve",
        "Audit-rate curve",
        "audit_rate",
        (0.0, 0.2, 0.5, 1.0),
        "Increasing α tests the trade-off between tampering deterrence and audit overhead.",
    ),
    ComparisonCard(
        "penalty_curve",
        "Audit-penalty curve",
        "audit_penalty",
        (0.0, 2.0, 10.0, 20.0),
        "Increasing F tests whether stronger consequences reduce tampering and improve assignment outcomes.",
    ),
    ComparisonCard(
        "scoring_curve",
        "Scoring-strength curve",
        "score_scale",
        (0.0, 1.0, 2.5, 6.0),
        "Increasing λ tests whether proper scoring reduces strategic report shading.",
    ),
    ComparisonCard(
        "calibration_curve",
        "Calibration curve",
        "calibration_kappa",
        (0.5, 1.0, 1.5, 2.0),
        "Varying calibration distortion tests whether evidence quality changes ECE, regret, and welfare.",
    ),
    ComparisonCard(
        "missing_evidence_curve",
        "Missing-evidence curve",
        "missing_evidence_prob",
        (0.0, 0.1, 0.3, 0.6),
        "Increasing missing evidence tests the information limit separately from strategic misreporting.",
    ),
    ComparisonCard(
        "coalition_size_curve",
        "Coalition-size curve",
        "coalition_size_k",
        (1, 2, 3, 5),
        "Increasing required-owner count tests coalition coverage, padding, and the conjunctive resolution burden.",
        {"use_coalition_range": False},
    ),
    ComparisonCard(
        "detection_curve",
        "Audit-detection curve",
        "audit_detection_delta",
        (0.0, 0.2, 0.4, 0.8),
        "Increasing per-neighbor detection tests the graph-audit term in the proposed contradiction probability.",
    ),
)


def build_experiment_config(
    base_config: SimulationConfig,
    overrides: dict[str, Any],
) -> SimulationConfig:
    """Return a validated copy of ``base_config`` with scenario overrides."""
    known_fields = set(SimulationConfig.__dataclass_fields__)
    unknown = set(overrides) - known_fields
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown experiment parameter(s): {names}")
    return replace(base_config, **overrides)


def run_experiment_card(
    card: ExperimentCard,
    base_config: SimulationConfig,
) -> tuple[SimulationConfig, SimulationResults]:
    """Run one named scenario using a deterministic configuration."""
    config = build_experiment_config(base_config, card.overrides)
    return config, run_full_simulation(config)


def run_comparison_card(
    card: ComparisonCard,
    base_config: SimulationConfig,
    num_replicates: int = 1,
):
    """Run one one-dimensional comparison using common replicate seeds."""
    sweep_config = build_experiment_config(base_config, card.base_overrides)
    return run_parameter_sweep(
        card.parameter,
        list(card.values),
        sweep_config,
        num_replicates=num_replicates,
    )


def get_interpretation_metrics(results: SimulationResults) -> dict[str, float]:
    """Return research metrics used to explain an experiment-card result."""
    return _research_metrics(results)


def _init_agents(
    num_agents: int,
    agent_regime: str,
    beta_load: float,
    seed: int,
) -> dict[int, StrategicAgent]:
    """Creates an isolated population of agents for a single system."""
    if agent_regime == "deflection":
        n_deflect = num_agents // 2
        n_acquire = 0
    elif agent_regime == "acquisition":
        n_deflect = 0
        n_acquire = num_agents // 2
    else:
        # Target ~30% deflectors, ~30% acquirers, ~40% neutral
        n_deflect = round(0.3 * num_agents)
        n_acquire = round(0.3 * num_agents)

    n_neutral = max(0, num_agents - n_deflect - n_acquire)
    ells = [2.0] * n_deflect + [-2.0] * n_acquire + [0.0] * n_neutral
    np.random.default_rng(seed).shuffle(ells)

    agents: dict[int, StrategicAgent] = {}
    for i in range(num_agents):
        agents[i] = StrategicAgent(
            agent_id=i,
            base_ell=ells[i],
            beta_load=beta_load,
            seed=seed + i,
        )
    return agents


def run_experiment(
    num_rounds: int = 100,
    num_agents: int = 12,
    k_owners: int = 5,
    coalition_size_range: tuple[int, int] | None = (1, 5),
    reward_per_owner: float | None = 5.0,
    agent_regime: str = "mixed",  # "mixed", "deflection", "acquisition"
    lambda_scale: float = 2.5,
    audit_rate: float = 0.2,
    credit_rule: CreditRule = CreditRule.SHAPLEY,
    rule: ResolutionRule = ResolutionRule.CONJUNCTIVE,
    dynamic_mode: bool = False,
    seed: int = 42,
):
    """Runs a multi-round simulation comparing Oracle, Honest, Baselines, and the Mechanism."""
    rng = np.random.default_rng(seed)
    env = RepositoryEnvironment(num_agents=num_agents, embedding_dim=32, seed=seed)
    task_gen = TaskGenerator(
        env,
        coalition_size_k=k_owners,
        coalition_size_range=coalition_size_range,
        reward_per_owner=reward_per_owner,
        rule=rule,
        seed=seed,
    )
    evidence_eng = EvidenceEngine(env, seed=seed)

    beta_load = 0.5 if dynamic_mode else 0.0

    # Isolated agent populations so Mechanism M state never leaks into B2/B3
    agents_M = _init_agents(num_agents, agent_regime, beta_load=beta_load, seed=seed)
    agents_B2 = _init_agents(num_agents, agent_regime, beta_load=beta_load, seed=seed)

    # Mechanism configurations
    mech_active = AssignmentMechanism(
        MechanismConfig(
            score_scale=lambda_scale,
            audit_rate=audit_rate,
            audit_penalty=10.0,
            credit_rule=credit_rule,
        )
    )
    config_no_inc = MechanismConfig(score_scale=0.0, audit_rate=0.0)
    expected_k = (
        0.5 * (coalition_size_range[0] + min(coalition_size_range[1], num_agents))
        if coalition_size_range is not None
        else float(k_owners)
    )
    base_rate_pi0 = expected_k / num_agents
    b1_midpoint = float(0.23 * np.sqrt(2.0 / max(1.0, expected_k)))
    b1_router = DescriptionRouterB1(
        env.profiles,
        staleness_rho=0.3,
        signal_midpoint=b1_midpoint,
        prior_pi0=base_rate_pi0,
        seed=seed,
    )
    b2_polling = PollingRouterB2()
    b3_voting = VotingDeciderB3(env.profiles, seed=seed)

    # Performance tracking accumulators
    trackers = {
        "Oracle (B4)": {"welfare": [], "cost": [], "resolved": []},
        "Honest (pi_true)": {"welfare": [], "regret": [], "miss_loss": [], "excess_cost": []},
        "Description (B1)": {"welfare": [], "regret": [], "miss_loss": [], "excess_cost": []},
        "Polling Router (B2)": {"welfare": [], "regret": [], "miss_loss": [], "excess_cost": []},
        "Voting (B3)": {"welfare": [], "regret": [], "miss_loss": [], "excess_cost": []},
        "Mechanism M": {"welfare": [], "regret": [], "miss_loss": [], "excess_cost": [], "inc_cost": []},
    }

    for round_id in range(num_rounds):
        task = task_gen.generate_task(task_id=round_id)
        evidence_dict = evidence_eng.generate_evidence(task)

        # 1. Oracle benchmark
        _, o_res, o_cost, o_welfare = MetricsEvaluator.compute_oracle(task)
        trackers["Oracle (B4)"]["welfare"].append(o_welfare)
        trackers["Oracle (B4)"]["cost"].append(o_cost)
        trackers["Oracle (B4)"]["resolved"].append(1.0 if o_res else 0.0)

        # 2. Reports under Mechanism M (with incentives)
        reports_M = {
            i: agents_M[i].compute_report(
                evidence_dict[i],
                mech_active.config,
                degree_m=env.get_degree(i),
            )
            for i in range(num_agents)
        }

        # 3. Reports under unmediated Polling Router B2 (no incentives, isolated population)
        reports_B2 = {
            i: agents_B2[i].compute_report(
                evidence_dict[i],
                config_no_inc,
                degree_m=env.get_degree(i),
            )
            for i in range(num_agents)
        }

        # Selection for each system
        A_M = mech_active.select_agents(task, reports_M, env.G)
        A_B1 = b1_router.select(task)
        A_B2 = b2_polling.select(task, reports_B2)
        A_B3 = b3_voting.select(task, reports_B2)

        # Evaluate Regret and Welfare
        A_H, _, _, _ = MetricsEvaluator.compute_truthful_benchmark(task, evidence_dict)
        reg_H = MetricsEvaluator.compute_regret_decomposition(task, A_H)
        reg_M = MetricsEvaluator.compute_regret_decomposition(task, A_M)
        reg_B1 = MetricsEvaluator.compute_regret_decomposition(task, A_B1)
        reg_B2 = MetricsEvaluator.compute_regret_decomposition(task, A_B2)
        reg_B3 = MetricsEvaluator.compute_regret_decomposition(task, A_B3)

        # Audits & scoring for Mechanism M
        penalties, _, _, audit_cost = mech_active.execute_audits(reports_M, env.G, rng)
        scores = mech_active.compute_all_scores(reports_M, task.z, base_rate_pi0)
        credits = mech_active.divide_credit(task, A_M)
        total_inc_cost = sum(scores.values()) + audit_cost

        # Update Mechanism M agents
        for i in range(num_agents):
            if dynamic_mode:
                agents_M[i].record_round_outcome(
                    credit_earned=credits[i],
                    penalty_incurred=penalties[i],
                    was_assigned=(i in A_M),
                    task_resolved=bool(reg_M["resolved"]),
                )
            agents_M[i].update_assignment_model(reports_M[i].r, was_assigned=(i in A_M))

        # Update isolated B2 agents (only workload if dynamic, 0 credit, 0 penalties)
        for i in range(num_agents):
            if dynamic_mode:
                agents_B2[i].record_round_outcome(
                    credit_earned=0.0,
                    penalty_incurred=0.0,
                    was_assigned=(i in A_B2),
                    task_resolved=bool(reg_B2["resolved"]),
                )
            agents_B2[i].update_assignment_model(reports_B2[i].r, was_assigned=(i in A_B2))

        # Accumulate metrics
        for name, res in [
            ("Honest (pi_true)", reg_H),
            ("Description (B1)", reg_B1),
            ("Polling Router (B2)", reg_B2),
            ("Voting (B3)", reg_B3),
            ("Mechanism M", reg_M),
        ]:
            trackers[name]["welfare"].append(res["social_welfare"])
            trackers[name]["regret"].append(res["regret"])
            trackers[name]["miss_loss"].append(res["miss_loss"])
            trackers[name]["excess_cost"].append(res["excess_cost"])

        trackers["Mechanism M"]["inc_cost"].append(total_inc_cost)

    return trackers


def print_comparison_table(trackers: dict, title: str):
    print(f"\n{'='*75}")
    print(f"EXPERIMENT: {title}")
    print(f"{'='*75}")
    print(f"{'System':<22} | {'Welfare':<9} | {'Regret':<9} | {'Miss Loss':<9} | {'Excess Cost':<11}")
    print(f"{'-'*75}")

    o_welfare = np.mean(trackers["Oracle (B4)"]["welfare"])
    print(f"{'Oracle (B4)':<22} | {o_welfare:9.2f} | {0.0:9.2f} | {0.0:9.2f} | {0.0:11.2f}")

    for name in ["Honest (pi_true)", "Description (B1)", "Polling Router (B2)", "Voting (B3)", "Mechanism M"]:
        w = np.mean(trackers[name]["welfare"])
        reg = np.mean(trackers[name]["regret"])
        miss = np.mean(trackers[name]["miss_loss"])
        exc = np.mean(trackers[name]["excess_cost"])
        print(f"{name:<22} | {w:9.2f} | {reg:9.2f} | {miss:9.2f} | {exc:11.2f}")

    if "inc_cost" in trackers["Mechanism M"]:
        avg_inc = np.mean(trackers["Mechanism M"]["inc_cost"])
        print(f"{'-'*75}")
        print(f"Mechanism Incentive Overhead (Score + Audits): {avg_inc:.2f}")


if __name__ == "__main__":
    # Mode 1: Static Benchmark - Mixed Regime (4 Defl / 4 Acq / 4 Neut, k in 1..5, R = 5*k)
    trackers_mixed = run_experiment(
        num_rounds=100,
        num_agents=12,
        coalition_size_range=(1, 5),
        reward_per_owner=5.0,
        agent_regime="mixed",
        lambda_scale=2.5,
        audit_rate=0.2,
        dynamic_mode=False,
        seed=42,
    )
    print_comparison_table(
        trackers_mixed,
        "Mode 1 (Static): Mixed Regime (N=12, k in 1..5, R=5*k, lam=2.5, Conjunctive)",
    )

    trackers_mixed_hi = run_experiment(
        num_rounds=100,
        num_agents=12,
        coalition_size_range=(1, 5),
        reward_per_owner=5.0,
        agent_regime="mixed",
        lambda_scale=5.0,
        audit_rate=0.2,
        dynamic_mode=False,
        seed=42,
    )
    print_comparison_table(
        trackers_mixed_hi,
        "Mode 1 (Static): Mixed Regime (N=12, k in 1..5, R=5*k, lam=5.0, Conjunctive)",
    )
