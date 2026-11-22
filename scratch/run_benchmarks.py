"""Script to run benchmark presets reproducibly and output raw metrics and LaTeX table."""

import sys
import os
import json
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation_runner import run_full_simulation, SimulationConfig
from src.types import ResolutionRule, CreditRule

BENCHMARK_PRESETS = {
    "Multi-Service Incident (Conjunctive Baseline)": {
        "num_rounds": 100,
        "num_agents": 10,
        "agent_regime": "mixed",
        "score_scale": 2.5,
        "audit_rate": 0.2,
        "audit_penalty": 10.0,
        "rule": ResolutionRule.CONJUNCTIVE,
        "credit_rule": CreditRule.SHAPLEY,
        "use_coalition_range": True,
        "coalition_min": 1,
        "coalition_max": 5,
        "reward_per_owner": 5.0,
        "dynamic_mode": False,
        "b1_staleness_rho": 0.3,
        "seed": 42,
    },
    "Specialist & Reviewer Dispatch (Disjunctive Baseline)": {
        "num_rounds": 100,
        "num_agents": 8,
        "agent_regime": "mixed",
        "score_scale": 2.5,
        "audit_rate": 0.2,
        "audit_penalty": 10.0,
        "rule": ResolutionRule.DISJUNCTIVE,
        "credit_rule": CreditRule.SHAPLEY,
        "use_coalition_range": False,
        "coalition_size_k": 1,
        "reward_per_owner": 5.0,
        "dynamic_mode": False,
        "b1_staleness_rho": 0.3,
        "seed": 42,
    },
    "Specialist Deflection (Disjunctive)": {
        "num_rounds": 100,
        "num_agents": 8,
        "agent_regime": "deflection",
        "deflector_ell": 3.0,
        "score_scale": 2.5,
        "audit_rate": 0.2,
        "audit_penalty": 10.0,
        "rule": ResolutionRule.DISJUNCTIVE,
        "credit_rule": CreditRule.SHAPLEY,
        "use_coalition_range": False,
        "coalition_size_k": 1,
        "reward_per_owner": 5.0,
        "dynamic_mode": False,
        "b1_staleness_rho": 0.3,
        "seed": 42,
    },
    "Reviewer Empire-Building (Disjunctive)": {
        "num_rounds": 100,
        "num_agents": 8,
        "agent_regime": "acquisition",
        "acquirer_ell": -3.0,
        "score_scale": 2.5,
        "audit_rate": 0.2,
        "audit_penalty": 10.0,
        "rule": ResolutionRule.DISJUNCTIVE,
        "credit_rule": CreditRule.SHAPLEY,
        "use_coalition_range": False,
        "coalition_size_k": 1,
        "reward_per_owner": 5.0,
        "dynamic_mode": False,
        "b1_staleness_rho": 0.3,
        "seed": 42,
    },
    "Deflection Crisis (Conjunctive)": {
        "num_rounds": 100,
        "num_agents": 10,
        "agent_regime": "deflection",
        "deflector_ell": 3.0,
        "score_scale": 2.5,
        "audit_rate": 0.2,
        "audit_penalty": 10.0,
        "rule": ResolutionRule.CONJUNCTIVE,
        "credit_rule": CreditRule.SHAPLEY,
        "use_coalition_range": True,
        "coalition_min": 1,
        "coalition_max": 5,
        "reward_per_owner": 5.0,
        "dynamic_mode": False,
        "b1_staleness_rho": 0.3,
        "seed": 42,
    },
    "Acquisition Hoarding (Conjunctive)": {
        "num_rounds": 100,
        "num_agents": 10,
        "agent_regime": "acquisition",
        "acquirer_ell": -3.0,
        "score_scale": 2.5,
        "audit_rate": 0.2,
        "audit_penalty": 10.0,
        "rule": ResolutionRule.CONJUNCTIVE,
        "credit_rule": CreditRule.SHAPLEY,
        "use_coalition_range": True,
        "coalition_min": 1,
        "coalition_max": 5,
        "reward_per_owner": 5.0,
        "dynamic_mode": False,
        "b1_staleness_rho": 0.3,
        "seed": 42,
    },
    "Legacy Catalog Staleness (rho = 0.70)": {
        "num_rounds": 100,
        "num_agents": 10,
        "agent_regime": "mixed",
        "score_scale": 2.5,
        "audit_rate": 0.2,
        "audit_penalty": 10.0,
        "rule": ResolutionRule.CONJUNCTIVE,
        "credit_rule": CreditRule.SHAPLEY,
        "use_coalition_range": True,
        "coalition_min": 1,
        "coalition_max": 5,
        "reward_per_owner": 5.0,
        "dynamic_mode": False,
        "b1_staleness_rho": 0.70,
        "seed": 42,
    },
    "Fully Aligned Baseline (ell = 0)": {
        "num_rounds": 100,
        "num_agents": 10,
        "agent_regime": "aligned",
        "score_scale": 2.5,
        "audit_rate": 0.2,
        "audit_penalty": 10.0,
        "rule": ResolutionRule.CONJUNCTIVE,
        "credit_rule": CreditRule.SHAPLEY,
        "use_coalition_range": True,
        "coalition_min": 1,
        "coalition_max": 5,
        "reward_per_owner": 5.0,
        "dynamic_mode": False,
        "b1_staleness_rho": 0.3,
        "seed": 42,
    },
}

def run_all():
    all_results = {}
    print("=" * 80)
    print("RUNNING REPRODUCIBLE BENCHMARK EXPERIMENTS (SEED = 42)")
    print("=" * 80)

    for preset_name, params in BENCHMARK_PRESETS.items():
        print(f"\n>>> Running: {preset_name} ...")
        # Handle regime ratios
        p = params.copy()
        regime = p.get("agent_regime", "mixed")
        if regime == "deflection":
            p["deflector_ratio"] = 0.5
            p["acquirer_ratio"] = 0.0
            p["neutral_ratio"] = 0.5
        elif regime == "acquisition":
            p["deflector_ratio"] = 0.0
            p["acquirer_ratio"] = 0.5
            p["neutral_ratio"] = 0.5
        elif regime == "aligned":
            p["deflector_ratio"] = 0.0
            p["acquirer_ratio"] = 0.0
            p["neutral_ratio"] = 1.0
        elif regime == "mixed":
            p["deflector_ratio"] = 0.3
            p["acquirer_ratio"] = 0.3
            p["neutral_ratio"] = 0.4

        cfg = SimulationConfig(**p)
        results = run_full_simulation(cfg)
        summary_df = results.summary_df

        res_dict = summary_df.set_index("System").to_dict(orient="index")
        all_results[preset_name] = res_dict

        print(summary_df[["System", "Welfare (Gross)", "Regret", "Efficiency Ratio (η)", "Miss Loss (ΔV)", "Excess Cost (ΔC)"]])

    # Save to JSON for perfect reproducibility
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved all raw results to {out_path}")

if __name__ == "__main__":
    run_all()
