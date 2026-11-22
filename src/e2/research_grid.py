"""Parameter sweep and research question analysis over direct LLM or lexical replay presets."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.e1.dataset import MODULES
from src.e1.game import Condition
from src.lab.preset import ReplayPreset, load_prepared
from src.lab.replay import Settings, replay, Run


def run_condition_metrics(preset: ReplayPreset, settings: Settings) -> dict:
    """Run a single condition and extract summary metrics."""
    run = replay(preset, settings)
    issues = run.issues
    
    count = len(preset.manifest["modules"])
    y = preset.outcomes
    sizes = np.array([i.bit_count() for i in range(1 << count)])
    targets = y @ (1 << np.arange(count))
    
    # Calculate True Oracle (B4) for this reward:
    # Under full information, Oracle assigns exactly the target modules S* if reward >= |S*|, else 0.
    oracle_assignments = np.where(settings.reward >= sizes[targets], targets, 0)
    oracle_covered = ((oracle_assignments & targets) == targets).astype(float)
    oracle_welfare = float((settings.reward * oracle_covered - sizes[oracle_assignments]).mean())
    
    welfare = float(issues["welfare"].mean())
    coverage = float(issues["coverage"].mean())
    
    # Recalculate true miss loss and excess cost against true oracle
    chosen = issues["assigned"].to_numpy()
    covered = (chosen & targets) == targets
    chosen_sizes = sizes[chosen]
    extra = sizes[chosen & ~targets]
    miss = float((settings.reward * (oracle_covered - covered.astype(float))).mean())
    excess = float((chosen_sizes - sizes[oracle_assignments]).mean())
    regret = miss + excess
    eff_ratio = float(welfare / oracle_welfare) if oracle_welfare > 0 else 0.0
    
    # Incentive and audit costs
    score_payout = float(issues["score"].mean())
    audit_cost = float((settings.condition(count).audit_cost * settings.audit * count))
    fines = float(issues["fines"].mean())
    net_overhead = score_payout + audit_cost - fines
    net_welfare = welfare - net_overhead
    
    # Pruning / Padding
    padding = float(extra.mean())
    any_padding = float((extra > 0).mean())
    
    # Audit detection
    contradicted = int(run.agents["contradicted"].sum())
    audited = int(run.agents["audited"].sum())
    catch_rate = (contradicted / audited) if audited > 0 else 0.0

    return {
        "welfare": welfare,
        "net_welfare": net_welfare,
        "efficiency": eff_ratio,
        "coverage": coverage,
        "miss_loss": miss,
        "excess_cost": excess,
        "regret": regret,
        "padding": padding,
        "any_padding": any_padding,
        "score_overhead": score_payout,
        "net_overhead": net_overhead,
        "audited_count": audited,
        "contradicted_count": contradicted,
        "catch_rate": catch_rate,
        "iterations": run.diagnostic.get("iterations", 0),
        "converged": run.diagnostic.get("status") == "converged",
    }


def sweep_rq1(preset: ReplayPreset, seed: int = 331) -> pd.DataFrame:
    """RQ1: Cost of Misalignment.
    Vary private cost ell (acquisition < 0, deflection > 0) and fraction of strategic agents phi.
    Scoring scale lambda = 0 (unmediated polling).
    """
    count = len(preset.manifest["modules"])
    rows = []
    
    # ell < 0 (acquisition), ell = 0 (honest), ell > 0 (deflection)
    cost_levels = [-2.0, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0]
    phi_levels = [0.0, 1/3, 2/3, 1.0]
    
    for ell in cost_levels:
        for phi in phi_levels:
            k = int(round(count * phi))
            settings = Settings(
                private_cost=ell,
                strategic_agents=k,
                scoring=0.0,
                audit=0.0,
                credit="off",
                seed=seed,
            )
            metrics = run_condition_metrics(preset, settings)
            regime = "Honest" if ell == 0 or k == 0 else ("Acquisition" if ell < 0 else "Deflection")
            rows.append({
                "regime": regime,
                "private_cost_ell": ell,
                "strategic_fraction_phi": phi,
                "strategic_count": k,
                **metrics,
            })
            
    return pd.DataFrame(rows)


def sweep_rq2(preset: ReplayPreset, seed: int = 331) -> pd.DataFrame:
    """RQ2: Incentive Recovery and the Audit Boundary.
    Examine scoring scale lambda and audit rate alpha across acquisition and deflection regimes.
    """
    count = len(preset.manifest["modules"])
    rows = []
    
    regimes = [
        ("Acquisition (phi=2/3)", -1.0, int(round(count * 2 / 3))),
        ("Deflection (phi=2/3)", 2.0, int(round(count * 2 / 3))),
    ]
    
    scoring_scales = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    audit_rates = [0.0, 0.10, 0.25, 0.50]
    
    for regime_name, ell, k in regimes:
        for scale in scoring_scales:
            for audit in audit_rates:
                settings = Settings(
                    private_cost=ell,
                    strategic_agents=k,
                    scoring=scale,
                    audit=audit,
                    credit="off",
                    seed=seed,
                )
                metrics = run_condition_metrics(preset, settings)
                rows.append({
                    "regime": regime_name,
                    "scoring_lambda": scale,
                    "audit_rate_alpha": audit,
                    **metrics,
                })
                
    return pd.DataFrame(rows)


def sweep_rq3(preset: ReplayPreset, seed: int = 331) -> pd.DataFrame:
    """RQ3: Coalition Credit Sharing & Multi-Module Trade-offs.
    Compare CreditRule (off, equal, shapley) across reward levels R and scoring levels.
    """
    count = len(preset.manifest["modules"])
    rows = []
    
    rules = ["off", "equal", "shapley"]
    rewards = [5, 8, 12]
    scoring_levels = [0.0, 8.0]
    
    # Focus on acquisition where credit competition is fiercest
    ell = -1.0
    k = int(round(count * 2 / 3))
    
    for credit in rules:
        for r in rewards:
            for scale in scoring_levels:
                settings = Settings(
                    private_cost=ell,
                    strategic_agents=k,
                    scoring=scale,
                    audit=0.25 if scale > 0 else 0.0,
                    credit=credit,
                    reward=r,
                    seed=seed,
                )
                metrics = run_condition_metrics(preset, settings)
                rows.append({
                    "credit_rule": credit,
                    "reward_R": r,
                    "scoring_lambda": scale,
                    **metrics,
                })
                
    return pd.DataFrame(rows)


def generate_methods_benchmark(preset: ReplayPreset, seed: int = 331) -> pd.DataFrame:
    """Generate the standardized Table 4 methods comparison on E2."""
    count = len(preset.manifest["modules"])
    k_strategic = int(round(count * 2 / 3))
    
    # Standard policies
    configs = [
        ("Honest Decentralized Polling (ell=0)", Settings(private_cost=0.0, strategic_agents=0, scoring=0.0, audit=0.0, seed=seed)),
        ("Strategic Polling (B2, phi=2/3)", Settings(private_cost=-1.0, strategic_agents=k_strategic, scoring=0.0, audit=0.0, seed=seed)),
        ("Mechanism M (Proper Scoring lambda=8)", Settings(private_cost=-1.0, strategic_agents=k_strategic, scoring=8.0, audit=0.0, seed=seed)),
        ("Mechanism M + Audit (lambda=8, alpha=0.25)", Settings(private_cost=-1.0, strategic_agents=k_strategic, scoring=8.0, audit=0.25, seed=seed)),
        ("Mechanism M + Shapley (lambda=8, alpha=0.25)", Settings(private_cost=-1.0, strategic_agents=k_strategic, scoring=8.0, audit=0.25, credit="shapley", seed=seed)),
    ]
    
    rows = []
    for name, s in configs:
        m = run_condition_metrics(preset, s)
        rows.append({
            "Policy / Mechanism": name,
            "Welfare (W)": m["welfare"],
            "Eff. Ratio (eta)": m["efficiency"],
            "Miss Loss (dV)": m["miss_loss"],
            "Excess Cost (dC)": m["excess_cost"],
            "Coverage (%)": m["coverage"] * 100,
            "Padding": m["padding"],
            "Net Overhead": m["net_overhead"],
            "Converged": m["converged"],
        })
    return pd.DataFrame(rows)


def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Helper to format a DataFrame into a clean LaTeX booktabs table snippet."""
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" + "c" * (len(df.columns) - 1) + "}",
        r"\toprule",
        " & ".join(df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        formatted_vals = []
        for v in row:
            if isinstance(v, float):
                formatted_vals.append(f"{v:.2f}")
            elif isinstance(v, bool):
                formatted_vals.append("Yes" if v else "No")
            else:
                formatted_vals.append(str(v))
        lines.append(" & ".join(formatted_vals) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def export_full_research_suite(preset: ReplayPreset, output_dir: Path | str = "reports/e2") -> dict[str, Path]:
    """Execute all sweeps, export CSVs, and generate LaTeX snippets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("[E2 Grid] Running RQ1: Misalignment Sweep ...")
    df_rq1 = sweep_rq1(preset)
    p_rq1 = out_path / "rq1_misalignment.csv"
    df_rq1.to_csv(p_rq1, index=False)
    
    print("[E2 Grid] Running RQ2: Scoring & Audit Sweep ...")
    df_rq2 = sweep_rq2(preset)
    p_rq2 = out_path / "rq2_scoring_audit.csv"
    df_rq2.to_csv(p_rq2, index=False)
    
    print("[E2 Grid] Running RQ3: Coalition Credit Sweep ...")
    df_rq3 = sweep_rq3(preset)
    p_rq3 = out_path / "rq3_credit_coalitions.csv"
    df_rq3.to_csv(p_rq3, index=False)
    
    print("[E2 Grid] Generating Table 4 Methods Benchmark ...")
    df_methods = generate_methods_benchmark(preset)
    p_methods = out_path / "methods_benchmark.csv"
    df_methods.to_csv(p_methods, index=False)
    
    # Generate LaTeX snippets
    latex_methods = format_latex_table(
        df_methods,
        caption="Empirical Benchmark Performance under LLM Probabilities (E2, qwen2.5-coder:3b on Django).",
        label="tab:e2_methods_benchmarks"
    )
    (out_path / "table4_e2.tex").write_text(latex_methods)
    
    print(f"[E2 Grid] All research outputs successfully written to {out_path.resolve()}")
    return {
        "rq1": p_rq1,
        "rq2": p_rq2,
        "rq3": p_rq3,
        "methods": p_methods,
        "tex": out_path / "table4_e2.tex",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run research question sweeps over prepared replay preset.")
    parser.add_argument("preset_path", type=Path, help="Path to prepared inputs.json.gz")
    parser.add_argument("--output", "-o", type=Path, default=Path("reports/e2"), help="Output directory")
    args = parser.parse_args()
    
    preset = load_prepared(args.preset_path)
    export_full_research_suite(preset, args.output)
