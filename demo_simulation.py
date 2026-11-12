"""Demonstration and benchmark comparison runner for task ownership simulation."""

from __future__ import annotations

import numpy as np

from src.agents import StrategicAgent
from src.baselines import DescriptionRouterB1, PollingRouterB2, VotingDeciderB3
from src.environment import RepositoryEnvironment, TaskGenerator
from src.evidence import EvidenceEngine
from src.mechanisms import AssignmentMechanism
from src.metrics import MetricsEvaluator
from src.types import (
    CreditRule,
    MechanismConfig,
    ResolutionRule,
)


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
    task_gen = TaskGenerator(env, coalition_size_k=k_owners, rule=rule, seed=seed)
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
    base_rate_pi0 = k_owners / num_agents
    b1_router = DescriptionRouterB1(
        env.profiles, staleness_rho=0.3, prior_pi0=base_rate_pi0, seed=seed
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
    # Mode 1: Static Benchmark - Mixed Regime (4 Deflectors, 4 Acquirers, 4 Neutral, Conjunctive)
    trackers_mixed = run_experiment(
        num_rounds=100,
        num_agents=12,
        k_owners=5,
        agent_regime="mixed",
        lambda_scale=2.5,
        audit_rate=0.2,
        dynamic_mode=False,
        seed=42,
    )
    print_comparison_table(
        trackers_mixed,
        "Mode 1 (Static): Mixed Regime (N=12, k=5, 4 Defl / 4 Acq / 4 Neut, Conjunctive)",
    )
