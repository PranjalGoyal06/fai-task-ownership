"""Streamlit Web Application: Deciding Task Ownership from Private Evidence under Misaligned Incentives.

Multi-Agent Systems Research Project | IIT Gandhinagar
Authors: Pranjal Goyal, Ujas Shah, Harshal Kundanlal Rana, Satyakam Mishra
"""

from __future__ import annotations

import math
import threading
from typing import Any
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.simulation_runner import (
    PARAMETER_METADATA,
    SimulationConfig,
    SimulationResults,
    SimulationCancelled,
    run_full_simulation,
    run_parameter_grid_search,
    run_parameter_sweep,
)
from src.types import CreditRule, ResolutionRule
from src.experiment_cards import (
    COMPARISON_CARDS,
    EXPERIMENT_CARDS,
    RESEARCH_QUESTION_CARDS,
    get_interpretation_metrics,
    run_comparison_card,
    run_experiment_card,
    run_research_question_card,
    run_rq1_experiment,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FAI Multi-Agent Task Routing Simulator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .delta-pos { color: #10B981; }
    .delta-neg { color: #EF4444; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 10px 16px;
        border-radius: 6px;
    }
    .param-box {
        background-color: #F1F5F9;
        border-left: 3px solid #3B82F6;
        padding: 6px 10px;
        font-size: 0.8rem;
        color: #475569;
        border-radius: 0 4px 4px 0;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header():
    st.markdown(
        """
        <div class="main-header">⚖️ Deciding Task Ownership from Private Evidence under Misaligned Incentives</div>
        <div class="sub-header">
            <b>Multi-Agent Mechanism Simulator & Benchmark Suite</b> | IIT Gandhinagar FAI Project<br>
            <i>Pranjal Goyal, Ujas Shah, Harshal Kundanlal Rana, Satyakam Mishra</i>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Sidebar: Presets and Parameter Configuration
# -----------------------------------------------------------------------------
def configure_parameters() -> SimulationConfig:
    st.sidebar.title("🎛️ Simulation Controls")

    # Benchmark Preset Selector
    presets = {
        "Mode 1: Standard Static Benchmark (Demo Default)": {
            "num_rounds": 100,
            "num_agents": 12,
            "agent_regime": "mixed",
            "score_scale": 2.5,
            "audit_rate": 0.2,
            "audit_penalty": 10.0,
            "rule": ResolutionRule.CONJUNCTIVE,
            "use_coalition_range": True,
            "coalition_min": 1,
            "coalition_max": 5,
            "reward_per_owner": 5.0,
            "dynamic_mode": False,
            "b1_staleness_rho": 0.3,
            "seed": 42,
        },
        "Mode 1: High Scoring Incentive (λ = 5.0)": {
            "num_rounds": 100,
            "num_agents": 12,
            "agent_regime": "mixed",
            "score_scale": 5.0,
            "audit_rate": 0.2,
            "audit_penalty": 10.0,
            "rule": ResolutionRule.CONJUNCTIVE,
            "use_coalition_range": True,
            "coalition_min": 1,
            "coalition_max": 5,
            "reward_per_owner": 5.0,
            "dynamic_mode": False,
            "b1_staleness_rho": 0.3,
            "seed": 42,
        },
        "Mode 2: Dynamic Learning & Workload Fatigue": {
            "num_rounds": 100,
            "num_agents": 12,
            "agent_regime": "mixed",
            "score_scale": 2.5,
            "audit_rate": 0.2,
            "audit_penalty": 10.0,
            "rule": ResolutionRule.CONJUNCTIVE,
            "use_coalition_range": True,
            "coalition_min": 1,
            "coalition_max": 5,
            "reward_per_owner": 5.0,
            "dynamic_mode": True,
            "beta_load": 0.5,
            "load_decay": 0.6,
            "alpha_credit_ema": 0.1,
            "b1_staleness_rho": 0.3,
            "seed": 42,
        },
        "Deflection Crisis (All Agents Deflect ℓ = +2.0)": {
            "num_rounds": 100,
            "num_agents": 12,
            "agent_regime": "deflection",
            "score_scale": 2.5,
            "audit_rate": 0.2,
            "audit_penalty": 10.0,
            "rule": ResolutionRule.CONJUNCTIVE,
            "use_coalition_range": True,
            "coalition_min": 1,
            "coalition_max": 5,
            "reward_per_owner": 5.0,
            "dynamic_mode": False,
            "b1_staleness_rho": 0.3,
            "seed": 42,
        },
        "Acquisition Hoarding (All Agents Acquire ℓ = -2.0)": {
            "num_rounds": 100,
            "num_agents": 12,
            "agent_regime": "acquisition",
            "score_scale": 2.5,
            "audit_rate": 0.2,
            "audit_penalty": 10.0,
            "rule": ResolutionRule.CONJUNCTIVE,
            "use_coalition_range": True,
            "coalition_min": 1,
            "coalition_max": 5,
            "reward_per_owner": 5.0,
            "dynamic_mode": False,
            "b1_staleness_rho": 0.3,
            "seed": 42,
        },
        "Stale Description Stress Test (ρ = 0.7)": {
            "num_rounds": 100,
            "num_agents": 12,
            "agent_regime": "mixed",
            "score_scale": 2.5,
            "audit_rate": 0.2,
            "audit_penalty": 10.0,
            "rule": ResolutionRule.CONJUNCTIVE,
            "use_coalition_range": True,
            "coalition_min": 1,
            "coalition_max": 5,
            "reward_per_owner": 5.0,
            "dynamic_mode": False,
            "b1_staleness_rho": 0.7,
            "seed": 42,
        },
        "Custom Configuration": {},
    }

    selected_preset = st.sidebar.selectbox(
        "⚡ Choose Benchmark Preset",
        list(presets.keys()),
        index=0,
        help="Quickly load predefined experiment regimes or select Custom to adjust all parameters.",
    )
    p_dict = presets[selected_preset]

    st.sidebar.markdown("---")

    # Group 1: General & Population Setup
    with st.sidebar.expander("⚙️ 1. General & Population Regime", expanded=True):
        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['num_rounds']['symbol']}</b>: {PARAMETER_METADATA['num_rounds']['desc']}</div>",
            unsafe_allow_html=True,
        )
        num_rounds = st.slider(
            "Rounds (T)",
            min_value=10,
            max_value=1000,
            value=p_dict.get("num_rounds", 100),
            step=10,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['num_agents']['symbol']}</b>: {PARAMETER_METADATA['num_agents']['desc']}</div>",
            unsafe_allow_html=True,
        )
        num_agents = st.slider(
            "Agents (N)",
            min_value=4,
            max_value=100,
            value=p_dict.get("num_agents", 12),
            step=1,
            help="Above 14 agents, assignment search uses the greedy marginal method instead of exact combinatorial search.",
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['agent_regime']['symbol']}</b>: {PARAMETER_METADATA['agent_regime']['desc']}</div>",
            unsafe_allow_html=True,
        )
        agent_regime = st.selectbox(
            "Agent Regime",
            ["mixed", "deflection", "acquisition", "custom"],
            index=["mixed", "deflection", "acquisition", "custom"].index(
                p_dict.get("agent_regime", "mixed")
            ),
            help="Mixed: 30% deflectors, 30% acquirers, 40% neutral. Deflection: 50% deflectors, 50% neutral. Acquisition: 50% acquirers, 50% neutral.",
        )
        if agent_regime == "mixed":
            st.caption("30% deflectors | 30% acquirers | 40% neutral")
        elif agent_regime == "deflection":
            st.caption("50% deflectors | 50% neutral; no acquirers")
        elif agent_regime == "acquisition":
            st.caption("50% acquirers | 50% neutral; no deflectors")

        deflector_ratio = 0.3
        acquirer_ratio = 0.3
        neutral_ratio = 0.4
        if agent_regime == "custom":
            st.markdown("Custom population proportions (ϕ_defl + ϕ_acq ≤ 1):")
            deflector_ratio = st.slider("Deflector Fraction (ϕ_defl)", 0.0, 1.0, 0.33, 0.05)
            acquirer_ratio = st.slider(
                "Acquirer Fraction (ϕ_acq)",
                0.0,
                max(0.0, 1.0 - deflector_ratio),
                min(0.33, max(0.0, 1.0 - deflector_ratio)),
                0.05,
            )
            neutral_ratio = max(0.0, 1.0 - deflector_ratio - acquirer_ratio)
            st.caption(f"Aligned Fraction (ϕ_neutral): {neutral_ratio:.2f}")

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['deflector_ell']['symbol']}</b>: {PARAMETER_METADATA['deflector_ell']['desc']}</div>",
            unsafe_allow_html=True,
        )
        deflector_ell = st.number_input(
            "Deflector Cost (ℓ_defl ≥ 0)",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['acquirer_ell']['symbol']}</b>: {PARAMETER_METADATA['acquirer_ell']['desc']}</div>",
            unsafe_allow_html=True,
        )
        acquirer_ell = st.number_input(
            "Acquirer Benefit (ℓ_acq ≤ 0)",
            max_value=0.0,
            value=-2.0,
            step=0.5,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['seed']['symbol']}</b>: {PARAMETER_METADATA['seed']['desc']}</div>",
            unsafe_allow_html=True,
        )
        seed = st.number_input(
            "Random Seed (s)",
            min_value=0,
            value=p_dict.get("seed", 42),
            step=1,
        )

    # Group 2: Mechanism M Incentives
    with st.sidebar.expander("⚖️ 2. Mechanism M (Scoring, Audits, Credit)", expanded=True):
        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['score_scale']['symbol']}</b>: {PARAMETER_METADATA['score_scale']['desc']}</div>",
            unsafe_allow_html=True,
        )
        score_scale = st.slider(
            "Brier Score Scale (λ)",
            min_value=0.0,
            max_value=10.0,
            value=float(p_dict.get("score_scale", 2.5)),
            step=0.5,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['audit_rate']['symbol']}</b>: {PARAMETER_METADATA['audit_rate']['desc']}</div>",
            unsafe_allow_html=True,
        )
        audit_rate = st.slider(
            "Audit Rate (α)",
            min_value=0.0,
            max_value=1.0,
            value=float(p_dict.get("audit_rate", 0.2)),
            step=0.05,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['audit_penalty']['symbol']}</b>: {PARAMETER_METADATA['audit_penalty']['desc']}</div>",
            unsafe_allow_html=True,
        )
        audit_penalty = st.number_input(
            "Audit Penalty (F)",
            min_value=0.0,
            max_value=50.0,
            value=float(p_dict.get("audit_penalty", 10.0)),
            step=1.0,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['audit_detection_delta']['symbol']}</b>: {PARAMETER_METADATA['audit_detection_delta']['desc']}</div>",
            unsafe_allow_html=True,
        )
        audit_detection_delta = st.slider(
            "Neighbor Detection Rate (δ)",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['cost_per_audit']['symbol']}</b>: {PARAMETER_METADATA['cost_per_audit']['desc']}</div>",
            unsafe_allow_html=True,
        )
        cost_per_audit = st.number_input(
            "Cost per Audit (c_a)",
            min_value=0.0,
            value=0.1,
            step=0.05,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['credit_rule']['symbol']}</b>: {PARAMETER_METADATA['credit_rule']['desc']}</div>",
            unsafe_allow_html=True,
        )
        credit_rule_str = st.selectbox("Credit Rule (χ)", ["SHAPLEY", "EQUAL_SPLIT"], index=0)
        credit_rule = (
            CreditRule.SHAPLEY if credit_rule_str == "SHAPLEY" else CreditRule.EQUAL_SPLIT
        )

    # Group 3: Task & Value Structure
    with st.sidebar.expander("📦 3. Task Structure & Coalition Resolution", expanded=False):
        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['rule']['symbol']}</b>: {PARAMETER_METADATA['rule']['desc']}</div>",
            unsafe_allow_html=True,
        )
        rule_str = st.selectbox(
            "Resolution Rule",
            ["CONJUNCTIVE", "DISJUNCTIVE"],
            index=0 if p_dict.get("rule", ResolutionRule.CONJUNCTIVE) == ResolutionRule.CONJUNCTIVE else 1,
        )
        rule = (
            ResolutionRule.CONJUNCTIVE
            if rule_str == "CONJUNCTIVE"
            else ResolutionRule.DISJUNCTIVE
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['use_coalition_range']['symbol']}</b>: {PARAMETER_METADATA['use_coalition_range']['desc']}</div>",
            unsafe_allow_html=True,
        )
        use_coalition_range = st.checkbox(
            "Use Stochastic Coalition Range",
            value=p_dict.get("use_coalition_range", True),
        )

        if use_coalition_range:
            col_min, col_max = st.columns(2)
            with col_min:
                coalition_min = st.number_input("Min k", min_value=1, max_value=num_agents, value=1)
            with col_max:
                coalition_max = st.number_input(
                    "Max k",
                    min_value=coalition_min,
                    max_value=num_agents,
                    value=min(5, num_agents),
                )
            coalition_size_k = coalition_min
        else:
            coalition_size_k = st.slider("Fixed Coalition Size (k)", 1, num_agents, 2)
            coalition_min = coalition_size_k
            coalition_max = coalition_size_k

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['reward_per_owner']['symbol']}</b>: {PARAMETER_METADATA['reward_per_owner']['desc']}</div>",
            unsafe_allow_html=True,
        )
        reward_per_owner = st.number_input(
            "Reward per Owner (R_k)",
            min_value=0.0,
            value=float(p_dict.get("reward_per_owner", 5.0)),
            step=1.0,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['default_cost']['symbol']}</b>: {PARAMETER_METADATA['default_cost']['desc']}</div>",
            unsafe_allow_html=True,
        )
        default_cost = st.number_input(
            "Social Cost per Agent (c_i)",
            min_value=0.0,
            value=1.0,
            step=0.25,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['cluster_prob']['symbol']}</b>: {PARAMETER_METADATA['cluster_prob']['desc']}</div>",
            unsafe_allow_html=True,
        )
        cluster_prob = st.slider("Graph Cluster Prob", 0.0, 1.0, 0.7, 0.05)

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['hop_decay']['symbol']}</b>: {PARAMETER_METADATA['hop_decay']['desc']}</div>",
            unsafe_allow_html=True,
        )
        hop_decay = st.slider("Hop Signal Decay", 0.1, 0.9, 0.5, 0.05)

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['task_noise_sigma']['symbol']}</b>: {PARAMETER_METADATA['task_noise_sigma']['desc']}</div>",
            unsafe_allow_html=True,
        )
        task_noise_sigma = st.number_input(
            "Task Embedding Noise (σ_task)",
            min_value=0.0,
            value=0.2,
            step=0.05,
        )

    # Group 4: Repository Environment & Dependency Graph
    with st.sidebar.expander("🕸️ 4. Environment & Topology Graph", expanded=False):
        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['graph_type']['symbol']}</b>: {PARAMETER_METADATA['graph_type']['desc']}</div>",
            unsafe_allow_html=True,
        )
        graph_type = st.selectbox(
            "Graph Topology (G)",
            ["scale_free", "erdos_renyi", "complete", "empty"],
            index=0,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['avg_degree']['symbol']}</b>: {PARAMETER_METADATA['avg_degree']['desc']}</div>",
            unsafe_allow_html=True,
        )
        avg_degree = st.slider("Average Node Degree (<m>)", 1, 8, 4)

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['embedding_dim']['symbol']}</b>: {PARAMETER_METADATA['embedding_dim']['desc']}</div>",
            unsafe_allow_html=True,
        )
        embedding_dim = st.number_input("Embedding Dim (d)", min_value=8, max_value=128, value=32, step=8)

    # Group 5: Private Evidence & Calibration
    with st.sidebar.expander("🔍 5. Private Evidence & Calibration", expanded=False):
        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['obs_noise_sigma']['symbol']}</b>: {PARAMETER_METADATA['obs_noise_sigma']['desc']}</div>",
            unsafe_allow_html=True,
        )
        obs_noise_sigma = st.slider("Observation Noise (σ_obs)", 0.0, 0.5, 0.15, 0.02)

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['missing_evidence_prob']['symbol']}</b>: {PARAMETER_METADATA['missing_evidence_prob']['desc']}</div>",
            unsafe_allow_html=True,
        )
        missing_evidence_prob = st.slider("Missing Evidence Rate (γ)", 0.0, 0.5, 0.0, 0.05)

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['signal_sensitivity']['symbol']}</b>: {PARAMETER_METADATA['signal_sensitivity']['desc']}</div>",
            unsafe_allow_html=True,
        )
        signal_sensitivity = st.number_input(
            "Signal Sensitivity (β_1)",
            min_value=0.0,
            value=8.0,
            step=1.0,
        )

        use_auto_midpoint = st.checkbox("Auto-compute midpoint (0.23 * sqrt(2/k))", value=True)
        signal_midpoint_override = 0.23
        if not use_auto_midpoint:
            signal_midpoint_override = st.number_input(
                "Manual Midpoint",
                min_value=0.0,
                max_value=1.0,
                value=0.23,
                step=0.05,
            )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['calibration_kappa']['symbol']}</b>: {PARAMETER_METADATA['calibration_kappa']['desc']}</div>",
            unsafe_allow_html=True,
        )
        calibration_kappa = st.slider(
            "Calibration Exponent (κ)",
            min_value=0.2,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="1.0: Calibrated. >1: Overconfident. <1: Underconfident.",
        )

    # Group 6: Strategic Agents & Dynamic Learning
    with st.sidebar.expander("🤖 6. Strategic Agents & Dynamic Learning", expanded=False):
        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['dynamic_mode']['symbol']}</b>: {PARAMETER_METADATA['dynamic_mode']['desc']}</div>",
            unsafe_allow_html=True,
        )
        dynamic_mode = st.checkbox(
            "Enable Mode 2 Dynamic Learning",
            value=p_dict.get("dynamic_mode", False),
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['beta_load']['symbol']}</b>: {PARAMETER_METADATA['beta_load']['desc']}</div>",
            unsafe_allow_html=True,
        )
        beta_load = st.slider("Fatigue Sensitivity (β_load)", 0.0, 2.0, 0.5, 0.1)

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['load_decay']['symbol']}</b>: {PARAMETER_METADATA['load_decay']['desc']}</div>",
            unsafe_allow_html=True,
        )
        load_decay = st.slider("Workload Decay (γ_load)", 0.1, 0.95, 0.6, 0.05)

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['alpha_credit_ema']['symbol']}</b>: {PARAMETER_METADATA['alpha_credit_ema']['desc']}</div>",
            unsafe_allow_html=True,
        )
        alpha_credit_ema = st.slider("Credit EMA Smoothing (α_ema)", 0.01, 0.5, 0.1, 0.05)

        # Agent internals
        st.markdown("<b>Agent Internal Heuristics:</b>", unsafe_allow_html=True)
        agent_theta0 = st.number_input("Assignment Intercept (θ_0)", value=-2.0, step=0.5)
        agent_theta1 = st.number_input("Assignment Slope (θ_1)", value=4.0, step=0.5)
        agent_online_lr = st.number_input(
            "Assignment Learning Rate (lr)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
        )
        agent_evasion_benefit_factor = st.slider("Evasion Benefit Factor", 0.1, 1.0, 0.5, 0.1)
        agent_tamper_deflect_signal = st.number_input(
            "Forged Deflect Signal",
            max_value=0.0,
            value=-0.5,
            step=0.1,
        )
        agent_tamper_acquire_signal = st.number_input(
            "Forged Acquire Signal",
            min_value=0.0,
            value=1.0,
            step=0.1,
        )
        agent_fp_iterations = st.slider("Fixed-Point Iterations (n_fp)", 2, 16, 8, 1)

    # Group 7: Baselines Specifics
    with st.sidebar.expander("🧭 7. Baselines Tuning (B1 & B3)", expanded=False):
        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['b1_staleness_rho']['symbol']}</b>: {PARAMETER_METADATA['b1_staleness_rho']['desc']}</div>",
            unsafe_allow_html=True,
        )
        b1_staleness_rho = st.slider(
            "B1 Description Staleness (ρ)",
            0.0,
            1.0,
            float(p_dict.get("b1_staleness_rho", 0.3)),
            0.05,
        )

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['b1_desc_noise_sigma']['symbol']}</b>: {PARAMETER_METADATA['b1_desc_noise_sigma']['desc']}</div>",
            unsafe_allow_html=True,
        )
        b1_desc_noise_sigma = st.slider("B1 Description Noise (σ_desc)", 0.0, 0.5, 0.25, 0.05)

        st.markdown(
            f"<div class='param-box'><b>{PARAMETER_METADATA['b3_peer_noise_sigma']['symbol']}</b>: {PARAMETER_METADATA['b3_peer_noise_sigma']['desc']}</div>",
            unsafe_allow_html=True,
        )
        b3_disabled = (rule == ResolutionRule.CONJUNCTIVE)
        b3_help = "Single-choice plurality voting only applies to Disjunctive tasks (excluded under Conjunctive mode)."
        b3_peer_noise_sigma = st.slider(
            "B3 Voting Peer Noise (σ_peer)",
            0.0,
            0.5,
            0.25,
            0.05,
            disabled=b3_disabled,
            help=b3_help,
        )
        if b3_disabled:
            st.caption("ℹ️ *B3 Voting is excluded under Conjunctive mode because single-choice plurality cannot form multi-agent coalitions.*")

    return SimulationConfig(
        num_rounds=num_rounds,
        num_agents=num_agents,
        seed=int(seed),
        agent_regime=agent_regime,
        deflector_ratio=deflector_ratio,
        acquirer_ratio=acquirer_ratio,
        neutral_ratio=neutral_ratio,
        deflector_ell=deflector_ell,
        acquirer_ell=acquirer_ell,
        dynamic_mode=dynamic_mode,
        beta_load=beta_load,
        load_decay=load_decay,
        alpha_credit_ema=alpha_credit_ema,
        rule=rule,
        coalition_size_k=coalition_size_k,
        use_coalition_range=use_coalition_range,
        coalition_min=int(coalition_min),
        coalition_max=int(coalition_max),
        reward_per_owner=reward_per_owner,
        default_cost=default_cost,
        cluster_prob=cluster_prob,
        hop_decay=hop_decay,
        task_noise_sigma=task_noise_sigma,
        embedding_dim=int(embedding_dim),
        graph_type=graph_type,
        avg_degree=int(avg_degree),
        obs_noise_sigma=obs_noise_sigma,
        missing_evidence_prob=missing_evidence_prob,
        signal_sensitivity=signal_sensitivity,
        use_auto_midpoint=use_auto_midpoint,
        signal_midpoint_override=signal_midpoint_override,
        calibration_kappa=calibration_kappa,
        score_scale=score_scale,
        audit_rate=audit_rate,
        audit_penalty=audit_penalty,
        audit_detection_delta=audit_detection_delta,
        cost_per_audit=cost_per_audit,
        credit_rule=credit_rule,
        b1_staleness_rho=b1_staleness_rho,
        b1_desc_noise_sigma=b1_desc_noise_sigma,
        b3_peer_noise_sigma=b3_peer_noise_sigma,
        agent_theta0=agent_theta0,
        agent_theta1=agent_theta1,
        agent_online_lr=agent_online_lr,
        agent_evasion_benefit_factor=agent_evasion_benefit_factor,
        agent_tamper_deflect_signal=agent_tamper_deflect_signal,
        agent_tamper_acquire_signal=agent_tamper_acquire_signal,
        agent_fp_iterations=agent_fp_iterations,
    )


# -----------------------------------------------------------------------------
# Main Application Content Tabs
# -----------------------------------------------------------------------------
def render_kpi_cards(results: SimulationResults):
    """Renders top-level KPI scorecards."""
    summary_df = results.summary_df
    m_row = summary_df[summary_df["System"] == "Mechanism M"].iloc[0]
    o_row = summary_df[summary_df["System"] == "Oracle (B4)"].iloc[0]
    b2_row = summary_df[summary_df["System"] == "Polling Router (B2)"].iloc[0]
    b1_row = summary_df[summary_df["System"] == "Description (B1)"].iloc[0]

    cols = st.columns(6)
    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Oracle Welfare</div>
                <div class="metric-value">{o_row['Welfare (Gross)']:.2f}</div>
                <div class="metric-delta delta-pos">Optimal Upper Bound</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        diff_b2 = m_row["Welfare (Gross)"] - b2_row["Welfare (Gross)"]
        sign = "+" if diff_b2 > 0 else ""
        delta_cls = "delta-pos" if diff_b2 > 0 else "delta-neg"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Mechanism M Welfare</div>
                <div class="metric-value">{m_row['Welfare (Gross)']:.2f}</div>
                <div class="metric-delta {delta_cls}">{sign}{diff_b2:.2f} vs Polling B2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        reg_diff = b2_row["Regret"] - m_row["Regret"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Mechanism M Regret</div>
                <div class="metric-value">{m_row['Regret']:.2f}</div>
                <div class="metric-delta delta-pos">Saved {reg_diff:.2f} regret vs B2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[3]:
        eff = m_row["Efficiency Ratio (η)"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Efficiency Ratio (η)</div>
                <div class="metric-value">{eff:.2f}</div>
                <div class="metric-delta">{eff * 100:.1f}% of Oracle</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[4]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Miss Loss (ΔV)</div>
                <div class="metric-value">{m_row['Miss Loss (ΔV)']:.2f}</div>
                <div class="metric-delta">Unresolved penalties</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[5]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Incentive Overhead</div>
                <div class="metric-value">{m_row['Incentive Overhead']:.2f}</div>
                <div class="metric-delta">Net Welfare: {m_row['Net Welfare']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_benchmark_tab(results: SimulationResults):
    """Tab 1: Benchmark Summary Table and Regret Decomposition."""
    st.subheader("📊 Comparative Benchmark Results")
    st.caption("Comprehensive comparative benchmark results with full regret decomposition and incentive economics.")

    summary_df = results.summary_df.copy()

    # Style dataframe for presentation
    def highlight_systems(row):
        if row["System"] == "Mechanism M":
            return ["background-color: #EFF6FF; font-weight: bold"] * len(row)
        elif row["System"] == "Oracle (B4)":
            return ["background-color: #F0FDF4; font-weight: bold"] * len(row)
        elif row["System"] == "Polling Router (B2)":
            return ["background-color: #FEF2F2"] * len(row)
        return [""] * len(row)

    styled_df = summary_df.style.apply(highlight_systems, axis=1).format(
        {
            "Welfare (Gross)": "{:.2f}",
            "Regret": "{:.2f}",
            "Miss Loss (ΔV)": "{:.2f}",
            "Excess Cost (ΔC)": "{:.2f}",
            "Resolution Rate": "{:.1f}%",
            "Avg Social Cost": "{:.2f}",
            "Efficiency Ratio (η)": "{:.3f}",
            "Incentive Overhead": "{:.2f}",
            "Net Welfare": "{:.2f}",
        }
    )
    st.dataframe(styled_df, use_container_width=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Social Welfare vs Regret Comparison")
        fig_bars = go.Figure()
        fig_bars.add_trace(
            go.Bar(
                x=summary_df["System"],
                y=summary_df["Welfare (Gross)"],
                name="Gross Social Welfare W",
                marker_color="#3B82F6",
            )
        )
        fig_bars.add_trace(
            go.Bar(
                x=summary_df["System"],
                y=summary_df["Regret"],
                name="Regret Reg(M)",
                marker_color="#EF4444",
            )
        )
        fig_bars.update_layout(
            barmode="group",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_tickangle=-25,
            yaxis_title="Value (Units)",
            height=380,
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    with col2:
        st.markdown("#### 🧩 Regret Decomposition: Miss Loss (ΔV) vs Excess Cost (ΔC)")
        fig_decomp = go.Figure()
        fig_decomp.add_trace(
            go.Bar(
                x=summary_df["System"],
                y=summary_df["Miss Loss (ΔV)"],
                name="Miss Loss ΔV (Failed Resolution)",
                marker_color="#F59E0B",
            )
        )
        fig_decomp.add_trace(
            go.Bar(
                x=summary_df["System"],
                y=summary_df["Excess Cost (ΔC)"],
                name="Excess Cost ΔC (Over-assignment)",
                marker_color="#8B5CF6",
            )
        )
        fig_decomp.update_layout(
            barmode="stack",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_tickangle=-25,
            yaxis_title="Regret Components",
            height=380,
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

    # Key Takeaways Box
    m_row = summary_df[summary_df["System"] == "Mechanism M"].iloc[0]
    b2_row = summary_df[summary_df["System"] == "Polling Router (B2)"].iloc[0]
    b1_row = summary_df[summary_df["System"] == "Description (B1)"].iloc[0]

    st.info(
        f"""
        **Key Mechanism Insights:**
        - **Incentive Recovery:** Unmediated Polling Router (B2) suffers **{b2_row['Regret']:.2f}** regret due to deflectors disclaiming responsibility ($r_i=0$) and acquirers hoarding tasks ($r_i=1$). Mechanism M reduces regret to **{m_row['Regret']:.2f}** (recovering **{b2_row['Regret'] - m_row['Regret']:.2f}** points).
        - **Staleness Defeat:** The central Description Router (B1) loses efficiency due to outdated documentation (Regret: **{b1_row['Regret']:.2f}**), whereas Mechanism M uses fresh private evidence disciplined by scoring and audits.
        - **Net Welfare Surplus:** Even after paying total incentive transfers of **{m_row['Incentive Overhead']:.2f}**, Mechanism M achieves a Net Social Welfare of **{m_row['Net Welfare']:.2f}**, vastly outperforming unmediated baselines.
        """
    )


def render_dynamics_tab(results: SimulationResults):
    """Tab 2: Round Convergence Dynamics and Time Series."""
    st.subheader("📈 Convergence & Multi-Round Dynamics")
    st.caption("Track cumulative social welfare, running average regret, and workload stabilization over rounds.")

    rounds_df = results.rounds_df

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cumulative Social Welfare Over Rounds")
        fig_cum = px.line(
            rounds_df,
            x="round",
            y="cum_welfare",
            color="system",
            labels={"cum_welfare": "Cumulative Social Welfare", "round": "Simulation Round"},
            color_discrete_map={
                "Oracle (B4)": "#10B981",
                "Description (B1)": "#F59E0B",
                "Polling Router (B2)": "#EF4444",
                "Voting (B3)": "#8B5CF6",
                "Mechanism M": "#2563EB",
            },
        )
        fig_cum.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=380)
        st.plotly_chart(fig_cum, use_container_width=True)

    with col2:
        st.markdown("#### Running Average Regret Stabilization")
        fig_reg = px.line(
            rounds_df,
            x="round",
            y="running_mean_regret",
            color="system",
            labels={"running_mean_regret": "Running Mean Regret", "round": "Simulation Round"},
            color_discrete_map={
                "Oracle (B4)": "#10B981",
                "Description (B1)": "#F59E0B",
                "Polling Router (B2)": "#EF4444",
                "Voting (B3)": "#8B5CF6",
                "Mechanism M": "#2563EB",
            },
        )
        fig_reg.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=380)
        st.plotly_chart(fig_reg, use_container_width=True)

    if results.config.dynamic_mode:
        st.markdown("#### 🔄 Mode 2: Agent Dynamic Fatigue & Credit Accumulation")
        agents_df = results.agents_df
        col_dyn1, col_dyn2 = st.columns(2)
        with col_dyn1:
            fig_fatigue = px.bar(
                agents_df,
                x="Agent ID",
                y="Recent Fatigue Load",
                color="Regime Type",
                title="Recent Accumulated Fatigue Load (β_load = " + str(results.config.beta_load) + ")",
            )
            st.plotly_chart(fig_fatigue, use_container_width=True)
        with col_dyn2:
            fig_credit = px.bar(
                agents_df,
                x="Agent ID",
                y="Total Credits M",
                color="Regime Type",
                title="Total Shapley Credits Earned Under Mechanism M",
            )
            st.plotly_chart(fig_credit, use_container_width=True)


def render_shading_tab(results: SimulationResults):
    """Tab 3: Strategic Report Shading & Audit Analysis."""
    st.subheader("🔬 Strategic Report Shading & Mechanism Discipline")
    st.caption("Observe how proper Brier scoring (λ) forces misaligned reports back towards truthful beliefs.")

    # Flatten reports across rounds for plotting
    task_hist = results.task_history
    data_points = []
    for t in task_hist:
        for i in range(results.config.num_agents):
            data_points.append(
                {
                    "agent_id": f"Agent {i}",
                    "pi_true": t["pi_true"][i],
                    "report_M": t["reports_M"][i],
                    "report_B2": t["reports_B2"][i],
                    "shading_M": t["reports_M"][i] - t["pi_true"][i],
                    "shading_B2": t["reports_B2"][i] - t["pi_true"][i],
                }
            )
    df_pts = pd.DataFrame(data_points)

    # Merge agent regime type
    agent_types = dict(zip(results.agents_df["Agent ID"], results.agents_df["Regime Type"]))
    df_pts["regime"] = df_pts["agent_id"].map(agent_types)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Mechanism M: Disciplined Reports (λ = " + str(results.config.score_scale) + ")")
        fig_m = px.scatter(
            df_pts.sample(min(800, len(df_pts)), random_state=42),
            x="pi_true",
            y="report_M",
            color="regime",
            labels={"pi_true": "True Belief π_true", "report_M": "Reported Probability r_i"},
            color_discrete_map={
                "Deflector (ℓ > 0)": "#EF4444",
                "Acquirer (ℓ < 0)": "#3B82F6",
                "Aligned (ℓ = 0)": "#10B981",
            },
            opacity=0.6,
        )
        # 45 degree truth line
        fig_m.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Truthful (r = π)",
                line=dict(color="black", dash="dash"),
            )
        )
        fig_m.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        st.markdown("#### Polling Router B2: Unmediated Reports (No Incentives)")
        fig_b2 = px.scatter(
            df_pts.sample(min(800, len(df_pts)), random_state=42),
            x="pi_true",
            y="report_B2",
            color="regime",
            labels={"pi_true": "True Belief π_true", "report_B2": "Reported Probability r_i"},
            color_discrete_map={
                "Deflector (ℓ > 0)": "#EF4444",
                "Acquirer (ℓ < 0)": "#3B82F6",
                "Aligned (ℓ = 0)": "#10B981",
            },
            opacity=0.6,
        )
        fig_b2.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Truthful (r = π)",
                line=dict(color="black", dash="dash"),
            )
        )
        fig_b2.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_b2, use_container_width=True)

    # Agent Level Metrics Table
    st.markdown("#### 👥 Agent Level Statistics & Shading Summary")
    st.dataframe(results.agents_df, use_container_width=True)

    # Audits & Fraud Deterrence
    m_tampering = sum(results.agents_df["Tampering Count"])
    total_audits = sum(len(t["audited"]) for t in task_hist)
    total_caught = sum(len(t["contradicted"]) for t in task_hist)

    st.markdown("#### 🕵️ Graph-Based Audits & Fraud Deterrence")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Tampering Attempts", m_tampering)
    with col_b:
        st.metric("Total Spot-Check Audits (α)", total_audits)
    with col_c:
        st.metric("Caught Contradictions (Penalties Paid)", total_caught)


def render_topology_tab(results: SimulationResults):
    """Tab 4: Repository Dependency Graph and Step-Through Round Inspector."""
    st.subheader("🕸️ Repository Topology & Task Inspection")
    st.caption("Inspect the module call graph and step through individual task assignment rounds.")

    col_graph, col_task = st.columns([1, 1.2])

    with col_graph:
        st.markdown("#### Module Dependency Graph G")
        G = results.environment.G

        pos = nx.spring_layout(G, seed=42)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.2, color="#94A3B8"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            deg = G.degree[node]
            node_text.append(f"Module {node}<br>Degree: {deg}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[f"M{i}" for i in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                size=22,
                color="#3B82F6",
                line=dict(width=2, color="#1E293B"),
            ),
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                margin=dict(b=10, l=10, r=10, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=420,
            ),
        )
        st.plotly_chart(fig_net, use_container_width=True)

    with col_task:
        st.markdown("#### 🔍 Step-Through Round Inspector")
        num_rounds = len(results.task_history)
        selected_round = st.slider("Select Round to Inspect", 0, num_rounds - 1, 0)
        t_data = results.task_history[selected_round]

        res_badge = "✅ Resolved" if t_data["res_Mechanism"] else "❌ Failed (Miss Loss)"
        st.markdown(
            fr"""
            **Task ID #{t_data['task_id']}** &nbsp;|&nbsp; Status: **{res_badge}**<br>
            - **Root Cause Seed Module ($s_0$):** `Module {t_data['root_module']}`
            - **Latent Required Coalition ($S^\star$):** `{t_data['S_star']}` (Coalition Size: {t_data['k']})
            - **Resolution Reward ($R$):** `{t_data['R']:.2f}`
            - **Audited Modules:** `{t_data['audited'] if t_data['audited'] else 'None'}`
            - **Contradicted & Penalized:** `{t_data['contradicted'] if t_data['contradicted'] else 'None'}`
            """,
            unsafe_allow_html=True,
        )

        # Assignment sets across routers
        assign_table = [
            {"System": "Oracle (B4)", "Assigned Set (A)": str(t_data["A_Oracle"]), "Status": "Optimal"},
            {"System": "Description (B1)", "Assigned Set (A)": str(t_data["A_B1"]), "Status": "Stale Match"},
            {"System": "Polling Router (B2)", "Assigned Set (A)": str(t_data["A_B2"]), "Status": "Unmediated"},
        ]
        if results.config.rule != ResolutionRule.CONJUNCTIVE:
            assign_table.append(
                {"System": "Voting Decider (B3)", "Assigned Set (A)": str(t_data["A_B3"]), "Status": "Plurality"}
            )
        assign_table.append(
            {"System": "Mechanism M", "Assigned Set (A)": str(t_data["A_Mechanism"]), "Status": "Incentivized"}
        )
        st.table(pd.DataFrame(assign_table))

        # Show module reports in selected round
        mod_reports = pd.DataFrame(
            {
                "Module": [f"M{i}" for i in range(results.config.num_agents)],
                "True In S*": [1 if i in t_data["S_star"] else 0 for i in range(results.config.num_agents)],
                "π_true": [round(t_data["pi_true"][i], 3) for i in range(results.config.num_agents)],
                "Report M (r)": [round(t_data["reports_M"][i], 3) for i in range(results.config.num_agents)],
                "Report B2 (r)": [round(t_data["reports_B2"][i], 3) for i in range(results.config.num_agents)],
            }
        )
        st.dataframe(mod_reports, height=180, use_container_width=True)


def render_sweep_tab(base_config: SimulationConfig):
    """Tab 5: Sensitivity Parameter Sweep Tool."""
    st.subheader("🎛️ Sensitivity Sweep Analysis (Research Experimentation)")
    st.caption(
        "Explore average assignment efficiency and regret across repeated runs as a single "
        "parameter is swept."
    )

    sweep_options = {
        "score_scale": ("Brier Score Scale (λ)", [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]),
        "audit_rate": ("Audit Rate (α)", [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]),
        "b1_staleness_rho": ("B1 Description Staleness (ρ)", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        "audit_penalty": ("Audit Penalty (F)", [0.0, 2.5, 5.0, 10.0, 15.0, 25.0]),
        "coalition_size_k": ("Base Coalition Size (k)", [1, 2, 3, 4, 6]),
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        chosen_param = st.selectbox(
            "Select Parameter to Sweep",
            list(sweep_options.keys()),
            format_func=lambda k: sweep_options[k][0],
        )
        param_label, default_vals = sweep_options[chosen_param]
        vals_str = st.text_input(
            "Sweep Values (comma-separated)",
            value=",".join(str(v) for v in default_vals),
        )
        try:
            sweep_vals = [float(x.strip()) if "." in x else int(x.strip()) for x in vals_str.split(",")]
        except Exception:
            sweep_vals = default_vals

        sweep_rounds = st.slider("Rounds per Sweep Step", 20, 100, 50, 10)
        sweep_replicates = st.number_input(
            "Independent Runs per Value",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            help="Average each parameter value across independently seeded runs.",
        )
        start_sweep = st.button("🚀 Run Sensitivity Sweep", type="primary")

    with col2:
        if start_sweep:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def sweep_cb(frac: float, msg: str):
                progress_bar.progress(frac)
                status_text.text(msg)

            cfg_sweep = SimulationConfig(**base_config.__dict__)
            cfg_sweep.num_rounds = sweep_rounds
            if chosen_param == "coalition_size_k":
                cfg_sweep.use_coalition_range = False

            df_sweep = run_parameter_sweep(
                chosen_param,
                sweep_vals,
                cfg_sweep,
                progress_callback=sweep_cb,
                num_replicates=int(sweep_replicates),
            )

            status_text.success("Sweep completed!")
            st.caption(
                f"Each value is averaged over {int(sweep_replicates)} independent run(s). "
                "Std columns show between-run variability."
            )
            st.dataframe(df_sweep, use_container_width=True)

            # Plot sweep results
            fig_sw = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Regret vs " + param_label, "Social Welfare vs " + param_label),
            )
            fig_sw.add_trace(
                go.Scatter(
                    x=df_sweep[chosen_param],
                    y=df_sweep["Mechanism Regret"],
                    name="Mechanism Regret",
                    line=dict(color="#2563EB", width=3),
                    error_y=dict(
                        type="data",
                        array=df_sweep["Mechanism Regret Std"],
                        visible=True,
                    ),
                ),
                row=1,
                col=1,
            )
            fig_sw.add_trace(
                go.Scatter(
                    x=df_sweep[chosen_param],
                    y=df_sweep["B2 Polling Regret"],
                    name="B2 Polling Regret",
                    line=dict(color="#EF4444", dash="dash"),
                ),
                row=1,
                col=1,
            )
            fig_sw.add_trace(
                go.Scatter(
                    x=df_sweep[chosen_param],
                    y=df_sweep["Mechanism Welfare"],
                    name="Mechanism Gross Welfare",
                    line=dict(color="#2563EB", width=3),
                    error_y=dict(
                        type="data",
                        array=df_sweep["Mechanism Welfare Std"],
                        visible=True,
                    ),
                ),
                row=1,
                col=2,
            )
            fig_sw.add_trace(
                go.Scatter(
                    x=df_sweep[chosen_param],
                    y=df_sweep["Mechanism Net Welfare"],
                    name="Mechanism Net Welfare",
                    line=dict(color="#10B981", width=2),
                ),
                row=1,
                col=2,
            )
            fig_sw.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_sw, use_container_width=True)
        else:
            st.info("Configure the sweep parameters on the left and click 'Run Sensitivity Sweep' to analyze response curves.")

    st.divider()
    st.subheader("🧮 Multi-Parameter Grid Search")
    st.caption(
        "Select any number of numeric parameters to vary. All unselected parameters stay fixed "
        "at their current sidebar values."
    )

    grid_options = {
        "deflector_ratio": ("Deflector Ratio (ϕ_defl)", False),
        "acquirer_ratio": ("Acquirer Ratio (ϕ_acq)", False),
        "score_scale": ("Brier Score Scale (λ)", False),
        "audit_rate": ("Audit Rate (α)", False),
        "audit_penalty": ("Audit Penalty (F)", False),
        "b1_staleness_rho": ("B1 Description Staleness (ρ)", False),
        "calibration_kappa": ("Calibration Exponent (κ)", False),
        "missing_evidence_prob": ("Missing Evidence Rate (γ)", False),
        "beta_load": ("Fatigue Sensitivity (β_load)", False),
        "load_decay": ("Workload Decay (γ_load)", False),
        "coalition_size_k": ("Coalition Size (k)", True),
    }
    selected_grid_params = st.multiselect(
        "Parameters to vary in the grid",
        list(grid_options),
        default=["score_scale", "audit_rate"],
        format_func=lambda key: grid_options[key][0],
    )

    grid_spec: dict[str, list[float | int]] = {}
    if selected_grid_params:
        grid_columns = st.columns(2)
        for index, parameter_name in enumerate(selected_grid_params):
            label, is_integer = grid_options[parameter_name]
            current_value = getattr(base_config, parameter_name)
            with grid_columns[index % 2]:
                st.markdown(f"**{label}**")
                input_columns = st.columns(3)
                with input_columns[0]:
                    grid_start = st.number_input(
                        "Start",
                        value=float(current_value),
                        step=1.0 if is_integer else 0.1,
                        key=f"grid_start_{parameter_name}",
                    )
                with input_columns[1]:
                    grid_stop = st.number_input(
                        "Stop",
                        value=float(current_value) + (2.0 if is_integer else 0.5),
                        step=1.0 if is_integer else 0.1,
                        key=f"grid_stop_{parameter_name}",
                    )
                with input_columns[2]:
                    grid_step = st.number_input(
                        "Step",
                        min_value=0.0001,
                        value=1.0 if is_integer else 0.1,
                        step=1.0 if is_integer else 0.1,
                        key=f"grid_step_{parameter_name}",
                    )
            if grid_step <= 0 or grid_stop < grid_start:
                st.error(f"{label}: stop must be at least start and step must be positive.")
                continue
            values = np.arange(grid_start, grid_stop + grid_step * 0.5, grid_step)
            if len(values) > 100:
                st.error(f"{label}: choose a range with at most 100 values.")
                continue
            if is_integer:
                grid_spec[parameter_name] = [int(round(value)) for value in values]
            else:
                grid_spec[parameter_name] = [round(float(value), 10) for value in values]

        grid_replicates = st.number_input(
            "Independent Runs per Grid Point",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            help="Averages each parameter combination across independently seeded runs.",
        )
        grid_size = int(np.prod([len(values) for values in grid_spec.values()])) if grid_spec else 0
        st.caption(f"Grid size: {grid_size} combination(s) × {int(grid_replicates)} run(s).")
        grid_run_column, grid_stop_column = st.columns(2)
        with grid_run_column:
            run_grid = st.button("🚀 Run Multi-Parameter Grid Search", type="primary")
        with grid_stop_column:
            stop_grid = st.button("⏹ Stop Grid Search")
        if "grid_cancel_requested" not in st.session_state:
            st.session_state.grid_cancel_requested = False
        if stop_grid:
            st.session_state.grid_cancel_requested = True
            st.warning("Grid search cancellation requested.")

        if run_grid:
            if grid_size == 0:
                st.error("Provide a valid range for every selected parameter.")
            else:
                st.session_state.grid_cancel_requested = False
                grid_progress = st.progress(0.0)
                grid_status = st.empty()

                def grid_cb(frac: float, msg: str):
                    grid_progress.progress(frac)
                    grid_status.text(msg)

                try:
                    grid_results = run_parameter_grid_search(
                        grid_spec,
                        base_config,
                        progress_callback=grid_cb,
                        num_replicates=int(grid_replicates),
                        cancel_check=lambda: st.session_state.grid_cancel_requested,
                    )
                except SimulationCancelled:
                    grid_status.warning("Grid search stopped.")
                else:
                    grid_status.success("Grid search completed!")
                    st.caption(
                        "Means are reported for each combination; columns ending in 'Std' show "
                        "between-run variability."
                    )
                    st.dataframe(grid_results, use_container_width=True)

                    st.markdown("#### Research-question metrics")
                    st.caption(
                        "RQ1: B2 regret decomposition · RQ2: recovery and incentive cost · "
                        "RQ3: coalition coverage and padding."
                    )
                    research_columns = [
                        *selected_grid_params,
                        "B2 Regret",
                        "B2 Miss Loss",
                        "B2 Excess Cost",
                        "Mechanism Regret",
                        "Regret Recovered",
                        "Regret Recovery Fraction",
                        "Incentive Overhead",
                        "Mean Absolute Shading M",
                        "Mechanism ECE",
                        "Tampering Rate",
                        "B2 Coalition Coverage",
                        "B2 Coalition Padding",
                        "Mechanism Coalition Coverage",
                        "Mechanism Coalition Padding",
                        "Mechanism Net Welfare",
                    ]
                    st.dataframe(
                        grid_results[[column for column in research_columns if column in grid_results]],
                        use_container_width=True,
                    )

                    if len(selected_grid_params) == 1:
                        parameter_name = selected_grid_params[0]
                        grid_plot = px.line(
                            grid_results,
                            x=parameter_name,
                            y=["Mechanism Net Welfare", "Mechanism Regret", "Regret Recovery Fraction"],
                            markers=True,
                            title=f"Research metrics across {grid_options[parameter_name][0]}",
                        )
                    else:
                        grid_plot = px.scatter(
                            grid_results,
                            x=selected_grid_params[0],
                            y=selected_grid_params[1],
                            color="Mechanism Net Welfare",
                            hover_data=[
                                "Mechanism Regret",
                                "Regret Recovery Fraction",
                                "Mechanism Coalition Coverage",
                            ],
                            title="Grid combinations colored by Mechanism Net Welfare",
                        )
                    st.plotly_chart(grid_plot, use_container_width=True)
    else:
        st.info("Select one or more parameters to configure a multi-parameter grid.")


def render_theory_tab():
    """Tab 6: Formal Mathematical Foundations and Reference."""
    st.subheader("📖 FAI Theoretical Foundation & Formula Guide")

    st.markdown(
        r"""
        ### 1. Task Resolution Rules
        Let $A \subseteq [N]$ be the assigned candidate owners and $S^\star \subseteq [N]$ be the latent ground-truth owners:
        - **Conjunctive Rule ($V_\wedge$):**
          $$V_\wedge(A; S^\star) = \mathbf{1}[S^\star \subseteq A]$$
          Every required owner must be assigned for task resolution.
        - **Disjunctive Rule ($V_\vee$):**
          $$V_\vee(A; S^\star) = \mathbf{1}[A \cap S^\star \neq \emptyset]$$
          At least one required owner suffices to resolve the task.

        ---

        ### 2. Centered Strictly Proper Brier Scoring Rule
        To elicit honest posterior beliefs $\pi_i$ from candidate owners, the mechanism scores reported probability $r_i \in [0, 1]$ against revealed outcome $z_i = \mathbf{1}[i \in S^\star]$:
        - **Brier Score:**
          $$s(r_i, z_i) = 1 - (z_i - r_i)^2$$
        - **Centered Brier Score against Base Rate $\pi_i^0$:**
          $$\tilde{s}_i(r_i, z_i) = s(r_i, z_i) - s(\pi_i^0, z_i) = -(z_i - r_i)^2 + (z_i - \pi_i^0)^2$$
          Centering guarantees zero expected payout to uninformed agents reporting the prior $\pi_i^0$.

        ---

        ### 3. Graph-Based Cross-Check Audits
        When an agent manipulates its verifiable evidence payload, the mechanism performs a spot-check with probability $\alpha$. Neighbor telemetry along dependency graph $G = ([N], E)$ uncovers fraud with contradiction probability:
        $$q_i = \alpha \left[1 - (1 - \delta)^{m_i}\right]$$
        where $m_i = |\mathcal{N}(i)|$ is agent $i$'s degree and $\delta$ is the per-neighbor detection rate. Caught agents pay penalty $F \ge 0$.

        ---

        ### 4. Coalition Credit Allocation (Shapley Value)
        For value function $v_A(T) = R \cdot V(T; S^\star)$, credit is allocated via the exact Shapley value:
        $$\psi_i(v_A) = \sum_{T \subseteq A \setminus \{i\}} \frac{|T|! (|A| - |T| - 1)!}{|A|!} \left[ v_A(T \cup \{i\}) - v_A(T) \right]$$
        - Under **Conjunctive Rule:** If $S^\star \subseteq A$, each critical owner in $S^\star$ receives $\frac{R}{|S^\star|}$; non-critical dummy players in $A \setminus S^\star$ receive $0$.
        - Under **Disjunctive Rule:** Critical owners in $A \cap S^\star$ receive $\frac{R}{|A \cap S^\star|}$.

        ---

        ### 5. Regret Decomposition Theorem
        $$\text{Reg}(M) = W(M^\circ) - W(M) = \underbrace{R \cdot \mathbb{E}\left[V(A^\circ) - V(A)\right]}_{\Delta V \text{ (Miss Loss)}} + \underbrace{\mathbb{E}\left[c(A) - c(A^\circ)\right]}_{\Delta C \text{ (Excess Cost)}}$$
        - **Deflectors ($\ell_i > 0$)** under-report ($r_i < \pi_i$), leading to **Miss Loss ($\Delta V$)**.
        - **Acquirers ($\ell_i < 0$)** over-report ($r_i > \pi_i$), leading to **Excess Cost ($\Delta C$)**.
        - **Efficiency Ratio:** $\eta(M) = \frac{W(M)}{W(M^\circ)}$.
        """
    )


def render_experiment_cards_tab(base_config: SimulationConfig):
    """Tab 7: Small reproducible scenarios and one-parameter comparisons."""
    st.subheader("🧪 Interpretable Experiment Cards")
    st.caption(
        "Each card runs only when clicked. The fixed seed makes repeated runs deterministic; "
        "expand a result to inspect the exact parameters used."
    )

    if "experiment_card_results" not in st.session_state:
        st.session_state.experiment_card_results = {}
    if "comparison_card_results" not in st.session_state:
        st.session_state.comparison_card_results = {}

    st.markdown("### Scenario cards")
    st.markdown(
        "These scenarios change one research condition at a time relative to the current "
        "sidebar configuration."
    )
    run_all_scenarios = st.container()
    with run_all_scenarios:
        if st.button(
            "▶▶ Run all scenario cards",
            key="run_all_scenarios",
            use_container_width=True,
        ):
            progress = st.progress(0.0)
            for index, card in enumerate(EXPERIMENT_CARDS, start=1):
                config, results = run_experiment_card(card, base_config)
                st.session_state.experiment_card_results[card.key] = (config, results)
                progress.progress(index / len(EXPERIMENT_CARDS))
            progress.empty()
            st.success(f"Completed {len(EXPERIMENT_CARDS)} scenario cards.")

    scenario_columns = st.columns(2)
    for index, card in enumerate(EXPERIMENT_CARDS):
        with scenario_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {card.title}")
                st.write(card.explanation)
                st.caption(
                    "Overrides: "
                    + ", ".join(f"`{key}={value}`" for key, value in card.overrides.items())
                )
                if st.button(
                    f"▶ Run {card.title}",
                    key=f"run_scenario_{card.key}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Running {card.title}..."):
                        config, results = run_experiment_card(card, base_config)
                    st.session_state.experiment_card_results[card.key] = (config, results)

                if card.key in st.session_state.experiment_card_results:
                    config, results = st.session_state.experiment_card_results[card.key]
                    interpretation = get_interpretation_metrics(results)
                    summary = results.summary_df[
                        results.summary_df["System"].isin(
                            ["Oracle (B4)", "Polling Router (B2)", "Mechanism M"]
                        )
                    ][
                        [
                            "System",
                            "Welfare (Gross)",
                            "Regret",
                            "Resolution Rate",
                            "Incentive Overhead",
                        ]
                    ]
                    st.dataframe(summary, hide_index=True, use_container_width=True)
                    metric_columns = st.columns(4)
                    metric_columns[0].metric(
                        "Report shading",
                        f"{interpretation['Mean Absolute Shading M']:.3f}",
                    )
                    metric_columns[1].metric(
                        "Tampering rate",
                        f"{interpretation['Tampering Rate']:.3f}",
                    )
                    metric_columns[2].metric(
                        "Coalition coverage",
                        f"{interpretation['Mechanism Coalition Coverage']:.3f}",
                    )
                    metric_columns[3].metric(
                        "Coalition padding",
                        f"{interpretation['Mechanism Coalition Padding']:.3f}",
                    )
                    with st.expander("Parameters and interpretation"):
                        parameter_values = {
                            key: value.value if isinstance(value, (CreditRule, ResolutionRule)) else value
                            for key, value in config.__dict__.items()
                        }
                        st.json(parameter_values)
                        st.info(
                            "Compare Mechanism M with B2 for the effect of incentives, and "
                            "with B4 for the remaining gap to the oracle."
                        )

    st.markdown("### One-parameter comparison cards")
    st.markdown(
        "Each card runs 3–5 values of one parameter while holding all other sidebar values fixed."
    )
    if st.button(
        "📈📈 Run all comparison cards",
        key="run_all_comparisons",
        use_container_width=True,
    ):
        progress = st.progress(0.0)
        for index, card in enumerate(COMPARISON_CARDS, start=1):
            comparison = run_comparison_card(card, base_config)
            st.session_state.comparison_card_results[card.key] = comparison
            progress.progress(index / len(COMPARISON_CARDS))
        progress.empty()
        st.success(f"Completed {len(COMPARISON_CARDS)} comparison cards.")

    comparison_columns = st.columns(2)
    for index, card in enumerate(COMPARISON_CARDS):
        with comparison_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {card.title}")
                st.write(card.explanation)
                st.caption(
                    f"Parameter: `{card.parameter}` | Values: "
                    + ", ".join(str(value) for value in card.values)
                )
                if card.base_overrides:
                    st.caption(
                        "Fixed card settings: "
                        + ", ".join(
                            f"`{key}={value}`" for key, value in card.base_overrides.items()
                        )
                    )
                if st.button(
                    f"📈 Compare {card.title}",
                    key=f"run_comparison_{card.key}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Running {card.title}..."):
                        comparison = run_comparison_card(card, base_config)
                    st.session_state.comparison_card_results[card.key] = comparison

                if card.key in st.session_state.comparison_card_results:
                    comparison = st.session_state.comparison_card_results[card.key]
                    display_columns = [
                        card.parameter,
                        "Mechanism Welfare",
                        "Mechanism Regret",
                        "Mechanism Resolution Rate",
                        "Mechanism Incentive Overhead",
                        "B2 Polling Regret",
                        "Mechanism Mean Absolute Shading",
                        "Mechanism Tampering Rate",
                        "Mechanism Coalition Coverage",
                        "Mechanism Coalition Padding",
                    ]
                    st.dataframe(
                        comparison[display_columns],
                        hide_index=True,
                        use_container_width=True,
                    )
                    chart = px.line(
                        comparison,
                        x=card.parameter,
                        y=[
                            "Mechanism Welfare",
                            "Mechanism Regret",
                            "Mechanism Resolution Rate",
                        ],
                        markers=True,
                        title=f"Effect of {card.parameter}",
                    )
                    st.plotly_chart(chart, use_container_width=True)


def mean_pivot(
    frame: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
) -> pd.DataFrame:
    """Average nuisance-factor duplicates before reshaping a research heatmap."""
    averaged = frame.groupby([index, columns], as_index=False)[values].mean()
    return averaged.pivot(index=index, columns=columns, values=values)


def research_grid_cache_key(
    card_key: str,
    base_config: SimulationConfig,
    base_seed: int,
    num_replicates: int,
    parameter_grid: dict[str, list[Any]],
) -> str:
    """Build an exact, readable cache key for a completed RQ grid."""
    grid_signature = {
        name: [
            value.value if hasattr(value, "value") else value
            for value in values
        ]
        for name, values in sorted(parameter_grid.items())
    }
    return repr({
        "card": card_key,
        "base_seed": base_seed,
        "replicates": num_replicates,
        "rounds": base_config.num_rounds,
        "agents": base_config.num_agents,
        "grid": grid_signature,
    })


def start_research_grid_worker(
    card,
    base_config: SimulationConfig,
    base_seed: int,
    num_replicates: int,
    grid_inputs: dict[str, list[Any]],
    cache_key: str,
) -> dict[str, Any]:
    """Start one RQ grid in a daemon worker and return a plain shared status dict."""
    status = {
        "running": True,
        "cancel_requested": False,
        "progress": 0.0,
        "message": "Starting grid search...",
        "result": None,
        "error": None,
        "cache_key": cache_key,
        "result_rendered": False,
    }

    def progress_callback(frac: float, message: str) -> None:
        status["progress"] = float(frac)
        status["message"] = message

    def cancel_check() -> bool:
        return bool(status["cancel_requested"])

    def worker() -> None:
        try:
            result = run_research_question_card(
                card,
                base_config,
                num_replicates=num_replicates,
                seed=base_seed,
                parameter_grid=grid_inputs,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            status["result"] = result
            status["progress"] = 1.0
            status["message"] = "Grid search complete."
        except SimulationCancelled:
            status["message"] = "Grid search stopped by user."
        except Exception as exc:
            status["error"] = str(exc)
            status["message"] = f"Grid search failed: {exc}"
        finally:
            status["running"] = False

    thread = threading.Thread(
        target=worker,
        name=f"rq-grid-{card.key}",
        daemon=True,
    )
    thread.start()
    return status


@st.fragment(run_every=0.5)
def render_research_status_fragment(card, status: dict[str, Any]) -> None:
    """Poll one RQ worker and provide an independent Stop button."""
    st.progress(
        float(status["progress"]),
        text=f"{status['message']} ({int(status['progress'] * 100)}%)",
    )
    stop_slot, status_slot = st.columns([1, 4])
    with stop_slot:
        if status["running"] and st.button(
            f"⏹ Stop {card.rq}",
            key=f"stop_research_{card.key}",
            use_container_width=True,
            type="secondary",
        ):
            status["cancel_requested"] = True
            status["message"] = f"{card.rq} stopping after active runs finish..."
            st.rerun()
    with status_slot:
        if status["error"]:
            st.error(status["error"])
        elif status["running"]:
            st.caption(status["message"])
        else:
            st.success(status["message"])

    if status["running"] is False and status["result"] is not None:
        cache_key = status["cache_key"]
        st.session_state.research_question_grid_cache[cache_key] = status["result"]
        st.session_state.research_question_results[card.key] = status["result"]
        if not status["result_rendered"]:
            status["result_rendered"] = True
            st.rerun(scope="app")


def summarize_rq1(frame: pd.DataFrame) -> list[str]:
    phi = frame.groupby("deflector_ratio", as_index=False)[
        ["B2 Regret", "B2 Miss Loss", "B2 Excess Cost"]
    ].mean()
    ell = frame.groupby("deflector_ell", as_index=False)[
        ["B2 Regret", "B2 Miss Loss", "B2 Excess Cost"]
    ].mean()
    low_phi, high_phi = phi.iloc[0], phi.iloc[-1]
    low_ell, high_ell = ell.iloc[0], ell.iloc[-1]
    return [
        f"Across nuisance-factor levels, mean B2 regret changes from "
        f"{low_phi['B2 Regret']:.3f} to {high_phi['B2 Regret']:.3f} "
        f"as ϕ moves from {low_phi['deflector_ratio']} to {high_phi['deflector_ratio']}.",
        f"At the lowest and highest tested cost magnitudes, mean miss loss is "
        f"{low_ell['B2 Miss Loss']:.3f} and {high_ell['B2 Miss Loss']:.3f}; "
        f"excess cost is {low_ell['B2 Excess Cost']:.3f} and "
        f"{high_ell['B2 Excess Cost']:.3f}.",
    ]


def summarize_rq2(frame: pd.DataFrame) -> list[str]:
    primary = frame.groupby(["score_scale", "audit_rate"], as_index=False)[
        ["Regret Recovery Fraction", "Incentive Overhead", "Tampering Rate", "Mean Absolute Shading M"]
    ].mean()
    best = primary.loc[primary["Regret Recovery Fraction"].idxmax()]
    return [
        f"The highest mean recovery in this completed grid is "
        f"{best['Regret Recovery Fraction']:.3f} at λ={best['score_scale']} "
        f"and α={best['audit_rate']}, with mean incentive overhead "
        f"{best['Incentive Overhead']:.3f}.",
        f"At that point, mean tampering is {best['Tampering Rate']:.3f} "
        f"and mean absolute shading is {best['Mean Absolute Shading M']:.3f}.",
    ]


def summarize_rq3(frame: pd.DataFrame) -> list[str]:
    grouped = frame.groupby(["rule", "credit_rule"], as_index=False)[
        [
            "Mechanism Coalition Coverage",
            "Mechanism Coalition Padding",
            "Required-Owner Underreporting",
            "Non-Required-Owner Overreporting",
        ]
    ].mean()
    conclusions: list[str] = []
    for rule_name in grouped["rule"].dropna().unique():
        subset = grouped[grouped["rule"] == rule_name]
        if len(subset) < 2:
            continue
        shapley = subset[subset["credit_rule"] == "shapley"]
        equal = subset[subset["credit_rule"] == "equal_split"]
        if shapley.empty or equal.empty:
            continue
        s, e = shapley.iloc[0], equal.iloc[0]
        conclusions.append(
            f"{rule_name}: coverage differs from {s['Mechanism Coalition Coverage']:.3f} "
            f"(Shapley) to {e['Mechanism Coalition Coverage']:.3f} (Equal Split); "
            f"padding differs from {s['Mechanism Coalition Padding']:.3f} to "
            f"{e['Mechanism Coalition Padding']:.3f}."
        )
    return conclusions


def build_fixed_factor_table(base_config: SimulationConfig, card_key: str) -> pd.DataFrame:
    """Build a string-only factor table that is safe for Streamlit Arrow serialization."""
    fixed_factors = {
        "Population regime": (
            "Custom deflector-only population"
            if card_key == "rq1_misalignment"
            else str(base_config.agent_regime)
        ),
        "Credit rule": str(base_config.credit_rule.value),
        "Resolution rule": str(base_config.rule.value),
        "Calibration κ": str(base_config.calibration_kappa),
        "Other mechanism/evidence parameters": (
            "Fixed at the current sidebar values unless listed as a grid factor"
        ),
    }
    return pd.DataFrame(
        [{"Factor": str(key), "Treatment": str(value)} for key, value in fixed_factors.items()]
    )


def render_research_questions_tab(base_config: SimulationConfig):
    """Tab 8: deterministic grid-search cards for RQ1-RQ3."""
    st.subheader("🔬 Research Questions")
    st.caption(
        "Run the three proposal questions as reproducible grids. Shared replicate seeds "
        "make parameter comparisons less noisy; means and standard deviations are retained."
    )

    if "research_question_results" not in st.session_state:
        st.session_state.research_question_results = {}
    if "research_question_grid_cache" not in st.session_state:
        st.session_state.research_question_grid_cache = {}
    if "research_question_heatmap_cache" not in st.session_state:
        st.session_state.research_question_heatmap_cache = {}
    if "research_question_statuses" not in st.session_state:
        st.session_state.research_question_statuses = {}

    for index, card in enumerate(RESEARCH_QUESTION_CARDS):
        with st.container(border=True):
            st.markdown(f"## {card.rq}: {card.title}")
            st.write(card.question)
            st.markdown("**Method**")
            st.write(card.explanation)

            with st.expander("Methodology and controlled factors", expanded=False):
                st.markdown(
                    """
                    **Design.** This is a full-factorial grid: every listed level of each factor
                    is crossed with every level of every other factor. Replicates receive
                    deterministic, distinct seeds at every grid point.

                    **Primary factors.** The parameters under study use 10 explicit levels.
                    **Nuisance factors.** Population size, simulation horizon, and evidence or
                    audit conditions use 3–4 levels to quantify sensitivity instead of hiding it.

                    **Seed protocol.** Replicate *r* uses `base_seed + r`; the same seed set is
                    reused across all grid points, reducing comparison noise.

                    **Uncertainty.** `Mean` and `Std` are across independent seeded runs. `Std`
                    is seed sensitivity, not within-round uncertainty.
                    """
                )
                st.dataframe(
                    build_fixed_factor_table(base_config, card.key),
                    hide_index=True,
                    use_container_width=True,
                )

            replicate_key = f"rq_replicates_{card.key}"
            seed_key = f"rq_seed_{card.key}"
            control_columns = st.columns(2)
            with control_columns[0]:
                base_seed = st.number_input(
                    "Base seed",
                    min_value=0,
                    max_value=100000,
                    value=base_config.seed,
                    step=1,
                    key=seed_key,
                )
            with control_columns[1]:
                num_replicates = st.number_input(
                    "Seeded replicates",
                    min_value=1,
                    max_value=50,
                    value=card.default_replicates,
                    step=1,
                    key=replicate_key,
                )
            grid_inputs: dict[str, list[Any]] = {
                name: list(values) for name, values in card.parameter_grid.items()
            }
            with st.expander("Grid levels", expanded=False):
                st.caption(
                    "Primary factors use 10 levels. Nuisance factors use 3 levels to expose "
                    "population, horizon, and measurement sensitivity."
                )
                level_columns = st.columns(3)
                numeric_parameters = [
                    parameter_name
                    for parameter_name in card.parameter_grid
                    if parameter_name not in {"credit_rule", "rule"}
                ]
                for index, parameter_name in enumerate(numeric_parameters):
                    default_values = card.parameter_grid[parameter_name]
                    labels = {
                        "deflector_ratio": "Deflector fraction ϕ",
                        "deflector_ell": "Deflector cost ℓ",
                        "score_scale": "Scoring scale λ",
                        "audit_rate": "Audit rate α",
                        "coalition_size_k": "Coalition size k",
                        "num_agents": "Population size N",
                        "num_rounds": "Simulation horizon T",
                        "missing_evidence_prob": "Missing evidence γ",
                        "audit_penalty": "Audit penalty F",
                        "audit_detection_delta": "Audit detection δ",
                    }
                    factor_role = (
                        "Primary"
                        if parameter_name in card.primary_parameters
                        else "Nuisance"
                        if parameter_name in card.nuisance_parameters
                        else "Fixed treatment"
                    )
                    with level_columns[index % 3]:
                        level_text = st.text_input(
                            f"[{factor_role}] {labels.get(parameter_name, parameter_name)}",
                            value=", ".join(str(value) for value in default_values),
                            key=f"rq_levels_{card.key}_{parameter_name}",
                        )
                    try:
                        parsed = [
                            int(value.strip()) if str(value).strip().isdigit()
                            else float(value.strip())
                            for value in level_text.split(",")
                            if value.strip()
                        ]
                        if parsed:
                            grid_inputs[parameter_name] = parsed
                    except ValueError:
                        st.error(f"Invalid levels for {parameter_name}; defaults will be used.")

            if card.key == "rq1_misalignment":
                num_points = len(grid_inputs.get("deflector_ratio", [])) + len(grid_inputs.get("deflector_ell", []))
                grid_size = num_points
                st.caption(
                    f"Fixed base seed: {int(base_seed)} · "
                    f"2 parameter sweeps ({num_points} parameter points) × {int(num_replicates)} seeds = {num_points * int(num_replicates)} runs"
                )
            else:
                grid_size = int(np.prod([len(values) for values in grid_inputs.values()]))
                st.caption(
                    f"Fixed base seed: {int(base_seed)} · "
                    f"{grid_size} combinations × {int(num_replicates)} seeds"
                )
            estimated_runs = grid_size * int(num_replicates)
            if estimated_runs > 50000:
                st.warning(
                    f"This design requires {estimated_runs:,} simulation runs. "
                    "It is research-scale work; reduce levels or replicates for faster screening."
                )

            if st.button(
                f"▶ Run {card.rq}",
                key=f"run_research_{card.key}",
                use_container_width=True,
                type="primary",
            ):
                cache_key = research_grid_cache_key(
                    card.key,
                    base_config,
                    int(base_seed),
                    int(num_replicates),
                    grid_inputs,
                )
                if cache_key in st.session_state.research_question_grid_cache:
                    grid_results = st.session_state.research_question_grid_cache[cache_key]
                    st.session_state.research_question_statuses[card.key] = {
                        "running": False,
                        "cancel_requested": False,
                        "progress": 1.0,
                        "message": f"{card.rq} loaded from cached grid.",
                        "result": grid_results,
                        "error": None,
                        "cache_key": cache_key,
                        "result_rendered": False,
                    }
                    st.session_state.research_question_results[card.key] = grid_results
                else:
                    status = start_research_grid_worker(
                        card,
                        base_config,
                        int(base_seed),
                        int(num_replicates),
                        grid_inputs,
                        cache_key,
                    )
                    st.session_state.research_question_statuses[card.key] = status

            status = st.session_state.research_question_statuses.get(card.key)
            if status:
                render_research_status_fragment(card, status)
            if card.key in st.session_state.research_question_results:
                grid_results = st.session_state.research_question_results[card.key].copy()
                st.divider()

                if card.key == "rq1_misalignment":
                    if "sweep" in grid_results.columns:
                        df_phi = grid_results[grid_results["sweep"] == "phi"].sort_values("deflector_ratio")
                        df_ell = grid_results[grid_results["sweep"] == "ell"].sort_values("deflector_ell")
                    else:
                        df_phi = grid_results.sort_values("deflector_ratio")
                        df_ell = grid_results.sort_values("deflector_ell")

                    st.markdown("### 📊 Regret Decomposition: Polling (B2) vs Mechanism $\\mathcal{M}$")
                    st.caption(
                        "Additive Identity: **Regret = Miss Loss ($\\Delta V$) + Excess Cost ($\\Delta C$)**. "
                        "Points reflect mean across seeded replicates; tooltips display $\\pm$ 1 std seed sensitivity."
                    )

                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=(
                            "(a) Varying Misaligned Fraction ϕ (Fixed ℓ = 3.0)",
                            "(b) Varying Deflector Cost Magnitude ℓ (Fixed ϕ = 0.5)",
                        ),
                        horizontal_spacing=0.08,
                    )

                    c_regret = "#1f77b4"  # Blue
                    c_miss = "#d62728"    # Crimson Red
                    c_cost = "#2ca02c"    # Forest Green

                    # Panel 1: phi sweep
                    if not df_phi.empty:
                        # Mechanism M
                        fig.add_trace(
                            go.Scatter(
                                x=df_phi["deflector_ratio"],
                                y=df_phi["Mechanism Regret"],
                                mode="lines+markers",
                                line=dict(color=c_regret, width=2.5),
                                marker=dict(symbol="circle", size=8),
                                name="Regret (Mechanism M)",
                                legendgroup="m_regret",
                                customdata=df_phi["Mechanism Regret Std"] if "Mechanism Regret Std" in df_phi else [0] * len(df_phi),
                                hovertemplate="ϕ: %{x:.2f}<br>M Regret: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df_phi["deflector_ratio"],
                                y=df_phi["Mechanism Miss Loss"],
                                mode="lines+markers",
                                line=dict(color=c_miss, width=2.5),
                                marker=dict(symbol="circle", size=8),
                                name="Miss Loss ΔV (Mechanism M)",
                                legendgroup="m_miss",
                                customdata=df_phi["Mechanism Miss Loss Std"] if "Mechanism Miss Loss Std" in df_phi else [0] * len(df_phi),
                                hovertemplate="ϕ: %{x:.2f}<br>M Miss Loss: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df_phi["deflector_ratio"],
                                y=df_phi["Mechanism Excess Cost"],
                                mode="lines+markers",
                                line=dict(color=c_cost, width=2.5),
                                marker=dict(symbol="circle", size=8),
                                name="Excess Cost ΔC (Mechanism M)",
                                legendgroup="m_cost",
                                customdata=df_phi["Mechanism Excess Cost Std"] if "Mechanism Excess Cost Std" in df_phi else [0] * len(df_phi),
                                hovertemplate="ϕ: %{x:.2f}<br>M Excess Cost: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )

                        # Polling B2
                        fig.add_trace(
                            go.Scatter(
                                x=df_phi["deflector_ratio"],
                                y=df_phi["B2 Regret"],
                                mode="lines+markers",
                                line=dict(color=c_regret, width=2.5, dash="dash"),
                                marker=dict(symbol="square", size=8),
                                name="Regret (Polling B2)",
                                legendgroup="b2_regret",
                                customdata=df_phi["B2 Regret Std"] if "B2 Regret Std" in df_phi else [0] * len(df_phi),
                                hovertemplate="ϕ: %{x:.2f}<br>B2 Regret: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df_phi["deflector_ratio"],
                                y=df_phi["B2 Miss Loss"],
                                mode="lines+markers",
                                line=dict(color=c_miss, width=2.5, dash="dash"),
                                marker=dict(symbol="square", size=8),
                                name="Miss Loss ΔV (Polling B2)",
                                legendgroup="b2_miss",
                                customdata=df_phi["B2 Miss Loss Std"] if "B2 Miss Loss Std" in df_phi else [0] * len(df_phi),
                                hovertemplate="ϕ: %{x:.2f}<br>B2 Miss Loss: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df_phi["deflector_ratio"],
                                y=df_phi["B2 Excess Cost"],
                                mode="lines+markers",
                                line=dict(color=c_cost, width=2.5, dash="dash"),
                                marker=dict(symbol="square", size=8),
                                name="Excess Cost ΔC (Polling B2)",
                                legendgroup="b2_cost",
                                customdata=df_phi["B2 Excess Cost Std"] if "B2 Excess Cost Std" in df_phi else [0] * len(df_phi),
                                hovertemplate="ϕ: %{x:.2f}<br>B2 Excess Cost: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )

                    # Panel 2: ell sweep
                    if not df_ell.empty:
                        # Mechanism M
                        fig.add_trace(
                            go.Scatter(
                                x=df_ell["deflector_ell"],
                                y=df_ell["Mechanism Regret"],
                                mode="lines+markers",
                                line=dict(color=c_regret, width=2.5),
                                marker=dict(symbol="circle", size=8),
                                showlegend=False,
                                customdata=df_ell["Mechanism Regret Std"] if "Mechanism Regret Std" in df_ell else [0] * len(df_ell),
                                hovertemplate="ℓ: %{x:.1f}<br>M Regret: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=2,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df_ell["deflector_ell"],
                                y=df_ell["Mechanism Miss Loss"],
                                mode="lines+markers",
                                line=dict(color=c_miss, width=2.5),
                                marker=dict(symbol="circle", size=8),
                                showlegend=False,
                                customdata=df_ell["Mechanism Miss Loss Std"] if "Mechanism Miss Loss Std" in df_ell else [0] * len(df_ell),
                                hovertemplate="ℓ: %{x:.1f}<br>M Miss Loss: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=2,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df_ell["deflector_ell"],
                                y=df_ell["Mechanism Excess Cost"],
                                mode="lines+markers",
                                line=dict(color=c_cost, width=2.5),
                                marker=dict(symbol="circle", size=8),
                                showlegend=False,
                                customdata=df_ell["Mechanism Excess Cost Std"] if "Mechanism Excess Cost Std" in df_ell else [0] * len(df_ell),
                                hovertemplate="ℓ: %{x:.1f}<br>M Excess Cost: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=2,
                        )

                        # Polling B2
                        fig.add_trace(
                            go.Scatter(
                                x=df_ell["deflector_ell"],
                                y=df_ell["B2 Regret"],
                                mode="lines+markers",
                                line=dict(color=c_regret, width=2.5, dash="dash"),
                                marker=dict(symbol="square", size=8),
                                showlegend=False,
                                customdata=df_ell["B2 Regret Std"] if "B2 Regret Std" in df_ell else [0] * len(df_ell),
                                hovertemplate="ℓ: %{x:.1f}<br>B2 Regret: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=2,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df_ell["deflector_ell"],
                                y=df_ell["B2 Miss Loss"],
                                mode="lines+markers",
                                line=dict(color=c_miss, width=2.5, dash="dash"),
                                marker=dict(symbol="square", size=8),
                                showlegend=False,
                                customdata=df_ell["B2 Miss Loss Std"] if "B2 Miss Loss Std" in df_ell else [0] * len(df_ell),
                                hovertemplate="ℓ: %{x:.1f}<br>B2 Miss Loss: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=2,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df_ell["deflector_ell"],
                                y=df_ell["B2 Excess Cost"],
                                mode="lines+markers",
                                line=dict(color=c_cost, width=2.5, dash="dash"),
                                marker=dict(symbol="square", size=8),
                                showlegend=False,
                                customdata=df_ell["B2 Excess Cost Std"] if "B2 Excess Cost Std" in df_ell else [0] * len(df_ell),
                                hovertemplate="ℓ: %{x:.1f}<br>B2 Excess Cost: %{y:.2f} ± %{customdata:.2f}<extra></extra>",
                            ),
                            row=1, col=2,
                        )

                    fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.5)", row=1, col=1)
                    fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.5)", row=1, col=2)
                    fig.update_xaxes(title_text="Misaligned Fraction ϕ", dtick=0.2, row=1, col=1)
                    fig.update_xaxes(title_text="Deflector Cost Magnitude ℓ", dtick=1.0, row=1, col=2)
                    fig.update_yaxes(title_text="Metric Value (Averaged Across Seeds)", row=1, col=1)
                    fig.update_yaxes(title_text="", row=1, col=2)
                    fig.update_layout(
                        height=520,
                        margin=dict(l=50, r=30, t=60, b=50),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.08,
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(255,255,255,0.7)",
                        ),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 📋 Empirical Data Tables")
                    tab_phi, tab_ell = st.tabs(["Sweep 1: Varying Fraction ϕ (Fixed ℓ = 3.0)", "Sweep 2: Varying Deflector Cost ℓ (Fixed ϕ = 0.5)"])
                    with tab_phi:
                        table_cols = [
                            c for c in [
                                "deflector_ratio", "B2 Regret", "B2 Regret Std", "Mechanism Regret", "Mechanism Regret Std",
                                "B2 Miss Loss", "Mechanism Miss Loss", "B2 Excess Cost", "Mechanism Excess Cost",
                                "B2 Resolution Rate", "Mechanism Resolution Rate", "Runs"
                            ] if c in df_phi.columns
                        ]
                        st.dataframe(df_phi[table_cols], hide_index=True, use_container_width=True)
                        st.download_button(
                            "📥 Download Sweep 1 CSV",
                            data=df_phi.to_csv(index=False),
                            file_name="rq1_phi_sweep.csv",
                            mime="text/csv",
                            key="dl_rq1_phi",
                        )
                    with tab_ell:
                        table_cols_ell = [
                            c for c in [
                                "deflector_ell", "B2 Regret", "B2 Regret Std", "Mechanism Regret", "Mechanism Regret Std",
                                "B2 Miss Loss", "Mechanism Miss Loss", "B2 Excess Cost", "Mechanism Excess Cost",
                                "B2 Resolution Rate", "Mechanism Resolution Rate", "Runs"
                            ] if c in df_ell.columns
                        ]
                        st.dataframe(df_ell[table_cols_ell], hide_index=True, use_container_width=True)
                        st.download_button(
                            "📥 Download Sweep 2 CSV",
                            data=df_ell.to_csv(index=False),
                            file_name="rq1_ell_sweep.csv",
                            mime="text/csv",
                            key="dl_rq1_ell",
                        )

                elif card.key == "rq2_incentive_recovery":
                    st.markdown("### Recovery and mechanism cost")
                    st.dataframe(
                        grid_results[
                            ["score_scale", "audit_rate", "Mechanism Regret",
                             "Mechanism Regret Std", "Regret Recovery Fraction",
                             "Regret Recovery Fraction Std", "Incentive Overhead",
                             "Tampering Rate", "Tampering Rate Std",
                             "Mean Absolute Shading M", "Mean Absolute Shading M Std", "Runs"]
                        ],
                        hide_index=True,
                        use_container_width=True,
                        height=520,
                    )
                    recovery_by_primary = grid_results.groupby(
                        ["score_scale", "audit_rate"], as_index=False
                    )[["Regret Recovery Fraction", "Incentive Overhead"]].mean()
                    best_row = recovery_by_primary.loc[
                        recovery_by_primary["Regret Recovery Fraction"].idxmax()
                    ]
                    st.success(
                        f"Highest mean recovery: {best_row['Regret Recovery Fraction']:.3f} "
                        f"at λ={best_row['score_scale']}, α={best_row['audit_rate']} "
                        f"(incentive overhead={best_row['Incentive Overhead']:.3f})."
                    )
                    with st.expander("Conclusion from this RQ2 grid", expanded=True):
                        for line in summarize_rq2(grid_results):
                            st.markdown(f"- {line}")
                    pivot = mean_pivot(
                        grid_results, "audit_rate", "score_scale",
                        "Regret Recovery Fraction",
                    )
                    fig = px.imshow(
                        pivot,
                        labels={"x": "Scoring scale λ", "y": "Audit rate α", "color": "Regret recovered"},
                        aspect="auto",
                        text_auto=".2f",
                        color_continuous_scale="Viridis",
                    )
                    fig.update_layout(height=560)
                    st.plotly_chart(fig, use_container_width=True)

                    diagnostic_columns = st.columns(2)
                    for column, title, metric in (
                        (diagnostic_columns[0], "Audit effect on tampering", "Tampering Rate"),
                        (diagnostic_columns[1], "Audit effect on report shading", "Mean Absolute Shading M"),
                    ):
                        with column:
                            diagnostic_pivot = mean_pivot(
                                grid_results, "audit_rate", "score_scale", metric
                            )
                            diagnostic_fig = px.imshow(
                                diagnostic_pivot,
                                labels={"x": "Scoring scale λ", "y": "Audit rate α", "color": title},
                                aspect="auto",
                                text_auto=".3f",
                                color_continuous_scale="RdYlGn_r",
                            )
                            diagnostic_fig.update_layout(
                                height=460, title=title,
                                margin=dict(l=20, r=20, t=40, b=20),
                            )
                            st.plotly_chart(diagnostic_fig, use_container_width=True)

                else:
                    for enum_column in ("credit_rule", "rule"):
                        if enum_column in grid_results:
                            grid_results[enum_column] = grid_results[enum_column].map(
                                lambda value: value.value if hasattr(value, "value") else str(value)
                            )
                    st.markdown("### Free-riding and team-padding outcomes")
                    st.dataframe(
                        grid_results[
                            ["credit_rule", "rule", "coalition_size_k",
                             "Required-Owner Underreporting",
                             "Required-Owner Underreporting Std",
                             "Non-Required-Owner Overreporting",
                             "Non-Required-Owner Overreporting Std",
                             "Mechanism Coalition Coverage",
                             "Mechanism Coalition Padding", "Runs"]
                        ],
                        hide_index=True,
                        use_container_width=True,
                        height=520,
                    )
                    with st.expander("Conclusion from this RQ3 grid", expanded=True):
                        rq3_lines = summarize_rq3(grid_results)
                        if rq3_lines:
                            for line in rq3_lines:
                                st.markdown(f"- {line}")
                        else:
                            st.info("Both credit rules must be present to summarize the RQ3 comparison.")
                    grouped = grid_results.groupby(
                        ["rule", "credit_rule"], as_index=False
                    ).agg(
                        Coverage=("Mechanism Coalition Coverage", "mean"),
                        Padding=("Mechanism Coalition Padding", "mean"),
                        Required_underreporting=("Required-Owner Underreporting", "mean"),
                        Nonrequired_overreporting=("Non-Required-Owner Overreporting", "mean"),
                    )
                    rq3_charts = [
                        (
                            "Coverage",
                            "Mechanism coalition coverage",
                            "Coverage (fraction)",
                        ),
                        (
                            "Padding",
                            "Average assigned non-required agents",
                            "Padding (agents)",
                        ),
                        (
                            "Required_underreporting",
                            "Required-owner under-reporting",
                            "Under-reporting magnitude",
                        ),
                        (
                            "Nonrequired_overreporting",
                            "Non-required-owner over-reporting",
                            "Over-reporting magnitude",
                        ),
                    ]
                    chart_columns = st.columns(2)
                    for chart_index, (metric, title, axis_title) in enumerate(rq3_charts):
                        with chart_columns[chart_index % 2]:
                            fig = px.bar(
                                grouped,
                                x="rule",
                                y=metric,
                                barmode="group",
                                color="credit_rule",
                                text_auto=".3f",
                                title=f"{title} by credit rule",
                                labels={
                                    "rule": "Resolution rule",
                                    metric: axis_title,
                                    "credit_rule": "Credit rule",
                                },
                            )
                            fig.update_layout(
                                height=420,
                                margin=dict(l=20, r=20, t=50, b=20),
                            )
                            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    render_header()
    config = configure_parameters()

    # Session state caching for simulation results
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None

    run_col1, run_col2 = st.columns([1, 4])
    with run_col1:
        run_sim = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

    if run_sim or st.session_state.sim_results is None:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def prog_cb(frac: float, msg: str):
            progress_bar.progress(frac)
            status_text.text(msg)

        results = run_full_simulation(config, progress_callback=prog_cb)
        st.session_state.sim_results = results
        progress_bar.empty()
        status_text.empty()
    else:
        results = st.session_state.sim_results

    # Render top KPIs
    render_kpi_cards(results)
    st.markdown("<br>", unsafe_allow_html=True)

    # A keyed radio navigation persists the selected page across Streamlit reruns.
    # This keeps button actions inside a page from returning the user to the first page.
    page = st.radio(
        "Application section",
        [
            "📊 Benchmark Results",
            "📈 Convergence & Dynamics",
            "🔬 Strategic Shading & Audits",
            "🕸️ Topology & Task Inspector",
            "🎛️ Sensitivity Sweeps",
            "📖 FAI Theoretical Guide",
            "🧪 Experiment Cards",
            "🔬 Research Questions",
        ],
        key="active_page",
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "📊 Benchmark Results":
        render_benchmark_tab(results)
    elif page == "📈 Convergence & Dynamics":
        render_dynamics_tab(results)
    elif page == "🔬 Strategic Shading & Audits":
        render_shading_tab(results)
    elif page == "🕸️ Topology & Task Inspector":
        render_topology_tab(results)
    elif page == "🎛️ Sensitivity Sweeps":
        render_sweep_tab(config)
    elif page == "📖 FAI Theoretical Guide":
        render_theory_tab()
    elif page == "🧪 Experiment Cards":
        render_experiment_cards_tab(config)
    elif page == "🔬 Research Questions":
        render_research_questions_tab(config)


if __name__ == "__main__":
    main()
