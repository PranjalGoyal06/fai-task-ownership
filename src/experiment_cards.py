"""Interpretability and scenario cards for the interactive Streamlit experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from src.simulation_runner import (
    SimulationConfig,
    SimulationResults,
    SimulationCancelled,
    _benchmark_worker_count,
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
        "Runs two targeted parameter sweeps over deflector fraction ϕ (with fixed ℓ=3.0) and "
        "deflector cost ℓ (with fixed ϕ=0.5). Each point uses shared replicate seeds, with "
        "mean and standard deviation reported for both Polling (B2) and Mechanism M regret decomposition "
        "(Regret = Miss Loss + Excess Cost).",
        {
            "deflector_ratio": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            "deflector_ell": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        },
        primary_parameters=("deflector_ratio", "deflector_ell"),
        nuisance_parameters={},
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
            "num_agents": (12,),
            "num_rounds": (60,),
            "audit_penalty": (10.0,),
            "audit_detection_delta": (0.4,),
            "seed": (42,),
            "agent_regime": ("mixed",),
            "deflector_ratio": (0.3,),
            "acquirer_ratio": (0.3,),
            "neutral_ratio": (0.4,),
            "deflector_ell": (2.0,),
            "acquirer_ell": (-2.0,),
            "neutral_ell": (0.0,),
            "dynamic_mode": (False,),
            "beta_load": (0.5,),
            "load_decay": (0.6,),
            "alpha_credit_ema": (0.1,),
            "rule": (ResolutionRule.CONJUNCTIVE,),
            "coalition_size_k": (5,),
            "use_coalition_range": (True,),
            "coalition_min": (1,),
            "coalition_max": (5,),
            "reward_per_owner": (5.0,),
            "resolution_reward": (10.0,),
            "default_cost": (1.0,),
            "cluster_prob": (0.7,),
            "hop_decay": (0.5,),
            "min_member_weight": (0.4,),
            "task_noise_sigma": (0.2,),
            "embedding_dim": (32,),
            "graph_type": ("scale_free",),
            "avg_degree": (4,),
            "obs_noise_sigma": (0.15,),
            "missing_evidence_prob": (0.0,),
            "signal_sensitivity": (8.0,),
            "use_auto_midpoint": (True,),
            "signal_midpoint_override": (0.23,),
            "calibration_kappa": (1.0,),
            "cost_per_audit": (0.1,),
            "credit_rule": (CreditRule.SHAPLEY,),
            "b1_staleness_rho": (0.3,),
            "b1_desc_noise_sigma": (0.25,),
            "b3_peer_noise_sigma": (0.25,),
            "agent_theta0": (-2.0,),
            "agent_theta1": (4.0,),
            "agent_online_lr": (0.05,),
            "agent_evasion_benefit_factor": (0.5,),
            "agent_tamper_deflect_signal": (-0.5,),
            "agent_tamper_acquire_signal": (1.0,),
            "agent_fp_iterations": (8,),
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
            "num_agents": (12,),
            "num_rounds": (50,),
            "missing_evidence_prob": (0.0,),
            "seed": (42,),
            "agent_regime": ("mixed",),
            "deflector_ratio": (0.3,),
            "acquirer_ratio": (0.3,),
            "neutral_ratio": (0.4,),
            "deflector_ell": (2.0,),
            "acquirer_ell": (-2.0,),
            "neutral_ell": (0.0,),
            "dynamic_mode": (False,),
            "beta_load": (0.5,),
            "load_decay": (0.6,),
            "alpha_credit_ema": (0.1,),
            "use_coalition_range": (False,),
            "coalition_min": (1,),
            "coalition_max": (5,),
            "reward_per_owner": (5.0,),
            "resolution_reward": (10.0,),
            "default_cost": (1.0,),
            "cluster_prob": (0.7,),
            "hop_decay": (0.5,),
            "min_member_weight": (0.4,),
            "task_noise_sigma": (0.2,),
            "embedding_dim": (32,),
            "graph_type": ("scale_free",),
            "avg_degree": (4,),
            "obs_noise_sigma": (0.15,),
            "signal_sensitivity": (8.0,),
            "use_auto_midpoint": (True,),
            "signal_midpoint_override": (0.23,),
            "calibration_kappa": (1.0,),
            "score_scale": (2.5,),
            "audit_rate": (0.2,),
            "audit_penalty": (10.0,),
            "audit_detection_delta": (0.4,),
            "cost_per_audit": (0.1,),
            "b1_staleness_rho": (0.3,),
            "b1_desc_noise_sigma": (0.25,),
            "b3_peer_noise_sigma": (0.25,),
            "agent_theta0": (-2.0,),
            "agent_theta1": (4.0,),
            "agent_online_lr": (0.05,),
            "agent_evasion_benefit_factor": (0.5,),
            "agent_tamper_deflect_signal": (-0.5,),
            "agent_tamper_acquire_signal": (1.0,),
            "agent_fp_iterations": (8,),
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


def run_rq1_experiment(
    base_config: SimulationConfig,
    base_seed: int = 42,
    num_replicates: int = 5,
    num_rounds: int | None = None,
    num_agents: int | None = None,
    phi_levels: list[float] | None = None,
    ell_levels: list[float] | None = None,
    fixed_ell: float | None = None,
    fixed_phi: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Run two targeted sweeps for RQ1 with multi-seed averaging and progress tracking.

    Sweep 1: Vary misaligned deflector fraction phi in phi_levels (fixed ell).
    Sweep 2: Vary deflector cost magnitude ell in ell_levels (fixed phi).
    For each point, runs num_replicates seeds (base_seed + r) and computes mean and std
    for Regret, Miss Loss, Excess Cost, and other metrics for both B2 and Mechanism M.
    """
    if phi_levels is None:
        phi_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if ell_levels is None:
        ell_levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    phi_levels = sorted(list(dict.fromkeys(float(v) for v in phi_levels)))
    ell_levels = sorted(list(dict.fromkeys(float(v) for v in ell_levels)))

    if fixed_ell is None:
        fixed_ell = 3.0 if 3.0 in ell_levels else float(ell_levels[len(ell_levels) // 2])
    if fixed_phi is None:
        fixed_phi = 0.5 if 0.5 in phi_levels else float(phi_levels[len(phi_levels) // 2])

    rounds = num_rounds if num_rounds is not None else base_config.num_rounds
    agents = num_agents if num_agents is not None else base_config.num_agents

    work_items: list[tuple[str, int, float, dict[str, Any]]] = []

    for idx, phi_val in enumerate(phi_levels):
        cfg_dict = {
            "agent_regime": "custom",
            "deflector_ratio": phi_val,
            "acquirer_ratio": 0.0,
            "neutral_ratio": max(0.0, 1.0 - phi_val),
            "deflector_ell": float(fixed_ell),
            "neutral_ell": 0.0,
            "acquirer_ell": 0.0,
            "num_rounds": rounds,
            "num_agents": agents,
        }
        work_items.append(("phi", idx, phi_val, cfg_dict))

    for idx, ell_val in enumerate(ell_levels):
        cfg_dict = {
            "agent_regime": "custom",
            "deflector_ratio": float(fixed_phi),
            "acquirer_ratio": 0.0,
            "neutral_ratio": max(0.0, 1.0 - float(fixed_phi)),
            "deflector_ell": ell_val,
            "neutral_ell": 0.0,
            "acquirer_ell": 0.0,
            "num_rounds": rounds,
            "num_agents": agents,
        }
        work_items.append(("ell", idx, ell_val, cfg_dict))

    total_runs = len(work_items) * num_replicates
    completed = 0
    replicate_metrics: dict[tuple[str, int, int], dict[str, float]] = {}

    def run_replicate(
        sweep_name: str,
        pt_idx: int,
        cfg_dict: dict[str, Any],
        replicate: int,
    ) -> tuple[str, int, int, dict[str, float]]:
        if cancel_check and cancel_check():
            raise SimulationCancelled("RQ1 experiment cancelled by user.")
        config_values = dict(base_config.__dict__)
        config_values.update(cfg_dict)
        config_values["seed"] = int(base_seed) + replicate
        sim_config = SimulationConfig(**config_values)
        results = run_full_simulation(sim_config)
        return sweep_name, pt_idx, replicate, _research_metrics(results)

    with ThreadPoolExecutor(
        max_workers=_benchmark_worker_count(),
        thread_name_prefix="rq1-sweep",
    ) as executor:
        futures = {
            executor.submit(
                run_replicate,
                sweep_name,
                pt_idx,
                cfg_dict,
                rep,
            ): (sweep_name, pt_idx, rep)
            for sweep_name, pt_idx, _, cfg_dict in work_items
            for rep in range(num_replicates)
        }
        for future in as_completed(futures):
            sweep_name, pt_idx, rep, metrics = future.result()
            replicate_metrics[(sweep_name, pt_idx, rep)] = metrics
            completed += 1
            if progress_callback:
                progress_callback(
                    completed / total_runs,
                    f"Completed RQ1 run {completed}/{total_runs} (Sweep: {sweep_name})",
                )

    rows: list[dict[str, Any]] = []
    for sweep_name, pt_idx, val, cfg_dict in work_items:
        point_metrics = [
            replicate_metrics[(sweep_name, pt_idx, rep)]
            for rep in range(num_replicates)
        ]
        row: dict[str, Any] = {
            "sweep": sweep_name,
            "deflector_ratio": cfg_dict["deflector_ratio"],
            "deflector_ell": cfg_dict["deflector_ell"],
            "Runs": num_replicates,
        }
        for metric_name in point_metrics[0]:
            vals = [m[metric_name] for m in point_metrics]
            row[metric_name] = float(np.mean(vals))
            row[f"{metric_name} Std"] = (
                float(np.std(vals, ddof=1)) if num_replicates > 1 else 0.0
            )
        rows.append(row)

    if progress_callback:
        progress_callback(1.0, f"RQ1 experiment complete: {len(work_items)} parameter points.")

    return pd.DataFrame(rows)


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
    """Run a research-question grid or focused sweeps with fixed seed offsets and averaged replicates."""
    if card.key == "rq1_misalignment":
        phi_levels = None
        ell_levels = None
        if parameter_grid is not None:
            if "deflector_ratio" in parameter_grid:
                phi_levels = [float(v) for v in parameter_grid["deflector_ratio"]]
            if "deflector_ell" in parameter_grid:
                ell_levels = [float(v) for v in parameter_grid["deflector_ell"]]
        return run_rq1_experiment(
            base_config=base_config,
            base_seed=seed if seed is not None else base_config.seed,
            num_replicates=num_replicates if num_replicates is not None else card.default_replicates,
            num_rounds=num_rounds if num_rounds is not None else card.default_rounds,
            num_agents=num_agents if num_agents is not None else base_config.num_agents,
            phi_levels=phi_levels,
            ell_levels=ell_levels,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    overrides: dict[str, Any] = {
        "num_rounds": num_rounds if num_rounds is not None else card.default_rounds,
    }
    if num_agents is not None:
        overrides["num_agents"] = num_agents
    if seed is not None:
        overrides["seed"] = seed
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
