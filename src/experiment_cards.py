"""Interpretability and scenario cards for the interactive Streamlit experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from src.simulation_runner import (
    SimulationConfig,
    SimulationResults,
    _research_metrics,
    run_full_simulation,
    run_parameter_grid_search,
    run_parameter_sweep,
)
from src.types import CreditRule, ResolutionRule


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


@dataclass(frozen=True)
class ResearchQuestionCard:
    """A reproducible experiment card for one of the three proposal research questions."""

    key: str
    rq: str
    title: str
    question: str
    explanation: str
    parameter_grid: dict[str, tuple[Any, ...]]
    primary_parameters: tuple[str, ...] = ()
    nuisance_parameters: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    default_rounds: int = 60
    default_replicates: int = 5


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


RESEARCH_QUESTION_CARDS: tuple[ResearchQuestionCard, ...] = (
    ResearchQuestionCard(
        "rq1_misalignment",
        "RQ1",
        "Cost of Misalignment",
        "How do regret, miss loss, and excess cost of unmediated polling scale with the "
        "misaligned fraction ϕ and cost magnitude ℓ_i?",
        "Runs a two-factor grid over deflector fraction and deflector cost. Each grid point "
        "uses shared replicate seeds, with mean and standard deviation reported for B2 regret "
        "decomposition and resolution outcomes.",
        {
            "deflector_ratio": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            "deflector_ell": (0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0),
            "num_agents": (6, 12, 24),
            "num_rounds": (30, 60, 120),
            "missing_evidence_prob": (0.0, 0.3, 0.6),
        },
        primary_parameters=("deflector_ratio", "deflector_ell"),
        nuisance_parameters={
            "num_agents": (6, 12, 24),
            "num_rounds": (30, 60, 120),
            "missing_evidence_prob": (0.0, 0.3, 0.6),
        },
        default_rounds=60,
        default_replicates=5,
    ),
    ResearchQuestionCard(
        "rq2_incentive_recovery",
        "RQ2",
        "Incentive Recovery",
        "For a given budget B, how much regret is recovered by scoring scale λ and audit "
        "rate α, and what balance is optimal?",
        "Runs a λ × α grid and evaluates regret recovery fraction, mechanism regret, incentive "
        "overhead, report shading, and tampering. The heatmap identifies the strongest recovery "
        "subject to the configured audit and scoring budget.",
        {
            "score_scale": (0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 6.5, 8.0, 10.0),
            "audit_rate": (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0),
            "num_agents": (6, 12, 24),
            "num_rounds": (30, 60, 120),
            "audit_penalty": (2.5, 10.0, 25.0),
            "audit_detection_delta": (0.2, 0.4, 0.8),
        },
        primary_parameters=("score_scale", "audit_rate"),
        nuisance_parameters={
            "num_agents": (6, 12, 24),
            "num_rounds": (30, 60, 120),
            "audit_penalty": (2.5, 10.0, 25.0),
            "audit_detection_delta": (0.2, 0.4, 0.8),
        },
        default_rounds=60,
        default_replicates=5,
    ),
    ResearchQuestionCard(
        "rq3_coalition_credit",
        "RQ3",
        "Coalition Credit Sharing",
        "How do Shapley and Equal-Split credit division rules affect free-riding, team padding, "
        "and equilibrium reports across conjunctive versus disjunctive tasks?",
        "Crosses credit rule, resolution rule, and coalition size. Reports coalition coverage, "
        "padding, mean report shading, required-owner under-reporting, non-required-owner "
        "over-reporting, and ECE with between-run uncertainty.",
        {
            "credit_rule": (CreditRule.SHAPLEY, CreditRule.EQUAL_SPLIT),
            "rule": (ResolutionRule.CONJUNCTIVE, ResolutionRule.DISJUNCTIVE),
            "coalition_size_k": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            "num_agents": (12, 18, 24),
            "num_rounds": (30, 60, 120),
            "missing_evidence_prob": (0.0, 0.3, 0.6),
        },
        primary_parameters=("coalition_size_k",),
        nuisance_parameters={
            "num_agents": (12, 18, 24),
            "num_rounds": (30, 60, 120),
            "missing_evidence_prob": (0.0, 0.3, 0.6),
        },
        default_rounds=50,
        default_replicates=5,
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


def run_research_question_card(
    card: ResearchQuestionCard,
    base_config: SimulationConfig,
    num_rounds: int | None = None,
    num_replicates: int | None = None,
    num_agents: int | None = None,
    seed: int | None = None,
    parameter_grid: dict[str, list[Any]] | None = None,
    progress_callback=None,
    cancel_check=None,
):
    """Run a research-question grid with fixed seed offsets and averaged replicates."""
    overrides: dict[str, Any] = {
        "num_rounds": num_rounds if num_rounds is not None else card.default_rounds,
    }
    if num_agents is not None:
        overrides["num_agents"] = num_agents
    if seed is not None:
        overrides["seed"] = seed
    if card.key == "rq1_misalignment":
        overrides["agent_regime"] = "custom"
        overrides["acquirer_ratio"] = 0.0
        overrides["neutral_ratio"] = 0.0
    if "coalition_size_k" in card.parameter_grid:
        overrides["use_coalition_range"] = False
        overrides["coalition_size_k"] = 1
    config = replace(base_config, **overrides)
    grid = (
        {name: list(values) for name, values in parameter_grid.items()}
        if parameter_grid is not None
        else {name: list(values) for name, values in card.parameter_grid.items()}
    )
    return run_parameter_grid_search(
        grid,
        config,
        progress_callback=progress_callback,
        num_replicates=num_replicates if num_replicates is not None else card.default_replicates,
        cancel_check=cancel_check,
    )


def get_interpretation_metrics(results: SimulationResults) -> dict[str, float]:
    """Return research metrics used to explain an experiment-card result."""
    return _research_metrics(results)
