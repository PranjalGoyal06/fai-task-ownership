"""Direct verbalized-confidence experiment (E2-D)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.e1.dataset import MODULES, module
from src.e1.game import Condition, fit_strategy, select
from src.e1.models import predictors
from src.e1.replay import comparators, decisions, settle, tag
from src.mechanisms import AssignmentMechanism
from src.metrics import MetricsEvaluator
from src.types import (
    CreditRule,
    MechanismConfig,
    ResolutionRule,
    Task,
)
from src.e2.providers import LLMProvider, provider_from_environment


PROBABILITY_RESPONSE_SCHEMA = {
    "probability": "number from 0 to 1",
    "evidence": ["short factual statements about this module"],
    "citations": [
        {
            "path": "django/relative/path.py",
            "start_line": 1,
            "end_line": 1,
            "snippet": "exact pre-fix source text",
        }
    ],
}
DIRECT_MANIFEST_VERSION = 1


def base_rates(directory: Path) -> np.ndarray:
    """Return E1's unsmoothed module base rates from training issues only."""
    issues = json.loads((directory / "issues.json").read_text(encoding="utf-8"))
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    earlier = [issue for issue in issues if issue["split"] == "train"]
    if not earlier:
        raise ValueError("Training issues are required to calculate module base rates")
    counts = np.zeros(len(MODULES), dtype=int)
    for issue in earlier:
        for owner in labels[issue["id"]]["owners"]:
            counts[owner] += 1
    return counts / len(earlier)


def local_snippets(
    repository: str,
    base_commit: str,
    module_id: int,
    issue_text: str,
    limit: int = 12,
) -> list[dict]:
    """Retrieve bounded pre-fix source snippets from one module only."""
    git = ["git", "--git-dir=" + str(repository)]
    paths = subprocess.check_output(
        git + ["ls-tree", "-r", "--name-only", base_commit, "django"],
        text=True,
        errors="replace",
    ).splitlines()
    paths = [
        path
        for path in paths
        if path.endswith(".py") and module(path) == module_id
    ]

    terms = {
        term.lower()
        for term in issue_text.replace("/", " ").replace("-", " ").split()
        if len(term) >= 5
    }
    snippets: list[dict] = []
    for path in sorted(paths):
        source = subprocess.check_output(
            git + ["show", f"{base_commit}:{path}"],
            text=True,
            errors="replace",
        )
        lines = source.splitlines()
        for line_number, line in enumerate(lines, start=1):
            lower = line.lower()
            if terms and any(term.lower() in lower for term in terms):
                start = max(1, line_number - 2)
                end = min(len(lines), line_number + 2)
                snippets.append(
                    {
                        "path": path,
                        "start_line": start,
                        "end_line": end,
                        "snippet": "\n".join(lines[start - 1:end]),
                    }
                )
                if len(snippets) >= limit:
                    return snippets
    return snippets


def validate_direct_response(
    raw_response: str,
    repository: str,
    base_commit: str,
    module_id: int,
    allowed_snippets: list[dict] | None = None,
) -> dict:
    """Parse and validate a direct verbalized-belief response."""
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM did not return valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")

    probability = result.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("LLM probability must be numeric")
    probability = float(probability)
    if not np.isfinite(probability) or probability < 0 or probability > 1:
        raise ValueError("LLM probability must be in [0, 1]")

    evidence = result.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    citations = result.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    accepted: list[dict] = []
    git = ["git", "--git-dir=" + str(repository)]
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        path = citation.get("path")
        if not isinstance(path, str) or module(path) != module_id:
            continue
        try:
            start = int(citation["start_line"])
            end = int(citation["end_line"])
            if start < 1 or end < start:
                continue
            source = subprocess.check_output(
                git + ["show", f"{base_commit}:{path}"],
                text=True,
                errors="replace",
            )
            lines = source.splitlines()
            if start > len(lines):
                continue
            end = min(end, len(lines))
            snippet = "\n".join(lines[start - 1:end])
        except (KeyError, TypeError, ValueError, subprocess.SubprocessError):
            continue
        if citation.get("snippet") != snippet:
            continue
        if allowed_snippets is not None:
            allowed = any(
                item["path"] == path
                and int(item["start_line"]) <= start
                and int(item["end_line"]) >= end
                for item in allowed_snippets
            )
            if not allowed:
                continue
        accepted.append(
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "snippet": snippet,
            }
        )

    return {
        "probability": probability,
        "evidence": [str(item) for item in evidence],
        "citations": accepted,
        "citations_rejected": len(citations) - len(accepted),
        "citation_validation": "source_match_only",
    }


def direct_prompt(
    issue: dict,
    module_id: int,
    prior: float,
    mode: str,
    snippets: list[dict] | None = None,
) -> tuple[str, str]:
    system = (
        "Estimate whether the historical issue will require a production code change "
        "in the specified module. Do not inspect or invent the gold patch. "
        "Return only valid JSON matching probability, evidence, and citations fields."
    )
    context = (
        f"Issue text:\n{issue['text']}\n\n"
        f"Module: {MODULES[module_id]}\n"
        f"Module base ownership rate from earlier issues: {prior:.6f}\n"
    )
    if mode == "no_tools":
        context += (
            "No source snippets are supplied. Use only the issue, module identity, "
            "and base ownership rate. Every citation must be an empty list."
        )
    else:
        context += (
            "Only the following pre-fix source snippets from this module are "
            "available. Do not cite any file or line outside this list:\n"
            f"{json.dumps(snippets or [], ensure_ascii=False)}"
        )
    context += (
        "\nOutput JSON exactly with this shape:\n"
        f"{json.dumps(PROBABILITY_RESPONSE_SCHEMA)}\n"
        "Set probability from 0 to 1. Keep evidence short and factual."
    )
    return system, context


def complete_belief_issues(beliefs: pd.DataFrame, split: str) -> set[str]:
    """Return issues with exactly one belief for every module in a split."""
    counts = beliefs.groupby(["split", "issue"])["module_id"].agg(
        row_count="size",
        module_count="nunique",
    )
    complete = counts[
        (counts.index.get_level_values("split") == split)
        & (counts["row_count"].to_numpy() == len(MODULES))
        & (counts["module_count"].to_numpy() == len(MODULES))
    ]
    return set(complete.index.get_level_values("issue"))


def summarize_replay(replay: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replay outcomes, with efficiency relative to the E1 B4 reference."""
    group_columns = ["split", "evidence", "config", "method", "regret_reference"]
    metrics = (
        "coverage",
        "padding",
        "any_padding",
        "welfare",
        "miss_loss",
        "excess_cost",
        "gap",
        "score",
        "audit_spend",
        "incentive_cost",
        "fine_paid",
        "credit_paid",
        "credit_to_extra",
        "shading",
        "gross_recovery_vs_strategic_b2",
        "net_recovery_vs_strategic_b2",
    )
    if replay.empty:
        return pd.DataFrame(columns=["split", "config", "method", "issue_count"])

    summaries = []
    for keys, group in replay.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["issue_count"] = len(group)
        for metric in metrics:
            if metric in group:
                row[f"mean_{metric}"] = float(group[metric].mean())
        reference_welfare = float((group["welfare"] + group["gap"]).mean())
        row["efficiency_vs_e1_lexical_b4"] = (
            float(group["welfare"].mean() / reference_welfare)
            if reference_welfare > 0
            else None
        )
        summaries.append(row)
    return pd.DataFrame(summaries)


def mechanism_outcomes(
    cases,
    ids: np.ndarray,
    outcomes: np.ndarray,
    mask: np.ndarray,
    condition: Condition,
    policy: np.ndarray,
    thresholds: np.ndarray,
    seed: int,
    fixed: dict[str, np.ndarray],
    evidence: str,
) -> pd.DataFrame:
    """Run one fitted reporting policy and return settled per-issue outcomes."""
    chosen, reports, caught, checked = decisions(
        cases,
        mask,
        condition,
        policy,
        thresholds,
        seed,
    )
    row = settle(ids, outcomes, chosen, fixed["B4"], condition.reward)
    row["evidence"] = evidence
    row["mode"] = "direct_llm_probability" if evidence == "direct_llm" else "e1_lexical"
    if condition == Condition():
        row["config"] = "B2-honest"
        row["method"] = "Honest routing"
        row["routing_label"] = (
            "B2-direct" if evidence == "direct_llm" else "B2-E1-lexical"
        )
    elif (
        condition.ell == -1.0
        and condition.phi == 2 / 3
        and condition.scale == 0
        and condition.audit == 0
        and condition.credit == "off"
    ):
        row["config"] = "B2-strategic"
        row["method"] = "Strategic B2"
    else:
        row["config"] = tag(condition)
        row["method"] = "Mechanism M"
    row["regret_reference"] = "E1 lexical B4 estimate"
    row["ell"] = condition.ell
    row["phi"] = condition.phi
    row["scale"] = condition.scale
    row["audit"] = condition.audit
    row["credit"] = condition.credit
    row["shading"] = (reports - cases.local[mask]).mean(axis=1)
    centered = -(outcomes - reports) ** 2 + (
        outcomes - cases.prior[None, :]
    ) ** 2
    row["score"] = condition.scale * centered.sum(axis=1)
    row["audit_spend"] = condition.audit_cost * checked.sum(axis=1)
    row["incentive_cost"] = row["score"] + row["audit_spend"]
    row["fine_paid"] = condition.fine * caught.sum(axis=1)
    row["credit_paid"] = 0.0
    row["credit_to_extra"] = 0.0

    if condition.credit != "off":
        credit_rule = (
            CreditRule.SHAPLEY
            if condition.credit == "shapley"
            else CreditRule.EQUAL_SPLIT
        )
        credit_mechanism = AssignmentMechanism(
            MechanismConfig(credit_rule=credit_rule)
        )
        bits = 1 << np.arange(len(MODULES))
        for issue_index, (outcome, chosen_mask) in enumerate(zip(outcomes, chosen)):
            truth = set(np.flatnonzero(outcome))
            assigned_set = set(np.flatnonzero(chosen_mask & bits))
            task = Task(
                issue_index,
                np.array([]),
                -1,
                truth,
                outcome,
                condition.reward,
                np.ones(len(MODULES)),
                ResolutionRule.CONJUNCTIVE,
            )
            payouts = credit_mechanism.divide_credit(task, assigned_set)
            row.at[issue_index, "credit_paid"] = sum(payouts.values())
            row.at[issue_index, "credit_to_extra"] = sum(
                payout
                for agent_id, payout in payouts.items()
                if agent_id not in truth
            )
    return row


def paired_effects(
    replay: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    complete_ids: dict[str, set[str]],
) -> pd.DataFrame:
    """Bootstrap direct-minus-E1 differences on exactly paired issue IDs."""
    rng = np.random.default_rng(20261122)
    comparisons: list[dict] = []
    metrics = (
        "brier_loss",
        "welfare",
        "coverage",
        "padding",
        "any_padding",
        "gross_recovery_vs_strategic_b2",
        "net_recovery_vs_strategic_b2",
    )

    for split_name in ("validation", "test"):
        ids = complete_ids[split_name]
        direct = prediction_rows[
            (prediction_rows["split"] == split_name)
            & prediction_rows["issue"].isin(ids)
        ].copy()
        if not direct.empty:
            direct["direct_brier"] = (
                direct["probability"] - direct["outcome"]
            ) ** 2
            direct["e1_lexical_brier"] = (
                direct["e1_lexical_probability"] - direct["outcome"]
            ) ** 2
            per_issue = direct.groupby("issue")[
                ["direct_brier", "e1_lexical_brier"]
            ].mean()
            values = (
                per_issue["direct_brier"] - per_issue["e1_lexical_brier"]
            ).to_numpy()
            comparisons.append(
                paired_summary_row(
                    "beliefs",
                    split_name,
                    "Ownership probability",
                    "brier_loss",
                    values,
                    rng,
                )
            )

    if not replay.empty:
        assignment_methods = {"Honest routing", "Strategic B2", "Mechanism M"}
        mechanisms = replay[
            (replay["method"].isin(assignment_methods))
            & replay["issue"].isin(
                complete_ids["validation"] | complete_ids["test"]
            )
        ]
        for (split_name, config, method), group in mechanisms.groupby(
            ["split", "config", "method"], sort=False
        ):
            by_evidence = {
                evidence: rows.set_index("issue")
                for evidence, rows in group.groupby("evidence")
            }
            if not {"direct_llm", "e1_lexical"}.issubset(by_evidence):
                continue
            paired_ids = by_evidence["direct_llm"].index.intersection(
                by_evidence["e1_lexical"].index
            )
            for metric in metrics[1:]:
                if metric not in group:
                    continue
                differences = (
                    by_evidence["direct_llm"].loc[paired_ids, metric].to_numpy(
                        dtype=float
                    )
                    - by_evidence["e1_lexical"].loc[paired_ids, metric].to_numpy(
                        dtype=float
                    )
                )
                comparisons.append(
                    paired_summary_row(
                        config,
                        split_name,
                        method,
                        metric,
                        differences,
                        rng,
                    )
                )
    return pd.DataFrame(comparisons)


def paired_summary_row(
    config: str,
    split: str,
    method: str,
    metric: str,
    differences: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    if len(differences):
        samples = rng.choice(
            differences,
            size=(2000, len(differences)),
            replace=True,
        ).mean(axis=1)
        mean = float(np.mean(differences))
        low, high = map(float, np.quantile(samples, [0.025, 0.975]))
    else:
        mean, low, high = None, None, None
    return {
        "config": config,
        "split": split,
        "method": method,
        "metric": metric,
        "count": len(differences),
        "direct_minus_e1_lexical": mean,
        "ci95_low": low,
        "ci95_high": high,
    }


def compare_modes(output_a: Path, output_b: Path) -> Path:
    """Compare no-tools and snippets results on the same complete issue cohort."""
    manifest_a = json.loads(
        (output_a / "e2d-manifest.json").read_text(encoding="utf-8")
    )
    manifest_b = json.loads(
        (output_b / "e2d-manifest.json").read_text(encoding="utf-8")
    )
    if {manifest_a.get("mode"), manifest_b.get("mode")} != {
        "no_tools",
        "snippets",
    }:
        raise ValueError("Mode comparison requires one no_tools and one snippets run")
    if manifest_a.get("issue_ids") != manifest_b.get("issue_ids"):
        raise ValueError("Mode runs must use the exact same selected issue IDs")
    if (
        manifest_a.get("provider") != manifest_b.get("provider")
        or manifest_a.get("model") != manifest_b.get("model")
    ):
        raise ValueError("Mode runs must use the same provider and model")

    mode_a = manifest_a["mode"]
    mode_b = manifest_b["mode"]
    predictions_a = pd.read_csv(output_a / "e2d-predictions.csv")
    predictions_b = pd.read_csv(output_b / "e2d-predictions.csv")
    complete_by_split = {}
    for split_name in ("validation", "test"):
        complete_a = complete_belief_issues(
            predictions_a,
            split_name,
        )
        complete_b = complete_belief_issues(predictions_b, split_name)
        complete_by_split[split_name] = complete_a & complete_b

    rng = np.random.default_rng(20261122)
    comparisons: list[dict] = []
    for split_name in ("validation", "test"):
        complete = complete_by_split[split_name]
        a = predictions_a[
            (predictions_a["split"] == split_name)
            & predictions_a["issue"].isin(complete)
        ].copy()
        b = predictions_b[
            (predictions_b["split"] == split_name)
            & predictions_b["issue"].isin(complete)
        ].copy()
        if not a.empty and not b.empty:
            a["loss_a"] = (a["probability"] - a["outcome"]) ** 2
            b["loss_b"] = (b["probability"] - b["outcome"]) ** 2
            per_issue_a = a.groupby("issue")["loss_a"].mean()
            per_issue_b = b.groupby("issue")["loss_b"].mean()
            shared = per_issue_a.index.intersection(per_issue_b.index)
            comparisons.append(
                paired_summary_row(
                    "beliefs",
                    split_name,
                    "Ownership probability",
                    "brier_loss",
                    (
                        per_issue_a.loc[shared] - per_issue_b.loc[shared]
                    ).to_numpy(),
                    rng,
                )
            )

    replay_a = pd.read_csv(output_a / "e2d-replay.csv")
    replay_b = pd.read_csv(output_b / "e2d-replay.csv")
    outcomes = (
        "welfare",
        "coverage",
        "padding",
        "any_padding",
        "gross_recovery_vs_strategic_b2",
        "net_recovery_vs_strategic_b2",
    )
    for split_name in ("validation", "test"):
        complete = complete_by_split[split_name]
        for config in sorted(
            set(replay_a["config"].dropna())
            & set(replay_b["config"].dropna())
        ):
            for method in sorted(
                set(
                    replay_a.loc[
                        (replay_a["split"] == split_name)
                        & (replay_a["config"] == config)
                        & (replay_a["evidence"] == "direct_llm"),
                        "method",
                    ]
                )
                & set(
                    replay_b.loc[
                        (replay_b["split"] == split_name)
                        & (replay_b["config"] == config)
                        & (replay_b["evidence"] == "direct_llm"),
                        "method",
                    ]
                )
            ):
                a = replay_a[
                    (replay_a["split"] == split_name)
                    & (replay_a["config"] == config)
                    & (replay_a["method"] == method)
                    & (replay_a["evidence"] == "direct_llm")
                    & replay_a["issue"].isin(complete)
                ].set_index("issue")
                b = replay_b[
                    (replay_b["split"] == split_name)
                    & (replay_b["config"] == config)
                    & (replay_b["method"] == method)
                    & (replay_b["evidence"] == "direct_llm")
                    & replay_b["issue"].isin(complete)
                ].set_index("issue")
                shared = a.index.intersection(b.index)
                for metric in outcomes:
                    if metric not in a or metric not in b:
                        continue
                    comparisons.append(
                        paired_summary_row(
                            config,
                            split_name,
                            method,
                            metric,
                            (
                                a.loc[shared, metric].to_numpy(dtype=float)
                                - b.loc[shared, metric].to_numpy(dtype=float)
                            ),
                            rng,
                        )
                    )

    result_path = output_a / "e2d-mode-comparison.csv"
    result = pd.DataFrame(comparisons)
    result.rename(
        columns={"direct_minus_e1_lexical": "mode_a_minus_mode_b"},
        inplace=True,
    )
    result["mode_a"] = mode_a
    result["mode_b"] = mode_b
    result.to_csv(result_path, index=False)
    return result_path


def run(
    directory: Path,
    output: Path,
    repository: str | None = None,
    splits: tuple[str, ...] = ("validation", "test"),
    max_issues: int | None = None,
    mode: str = "no_tools",
    provider: LLMProvider | None = None,
) -> None:
    """Generate one cached verbalized belief per selected issue/module."""
    provider = provider or provider_from_environment()
    issues = json.loads((directory / "issues.json").read_text(encoding="utf-8"))
    if max_issues is not None:
        if max_issues < 1:
            raise ValueError("max_issues must be positive")
        selected_splits: dict[str, int] = {}
        selected = []
        for issue in issues:
            split = issue["split"]
            if split not in splits:
                continue
            split_count = selected_splits.get(split, 0)
            if split_count >= max_issues:
                continue
            selected.append(issue)
            selected_splits[split] = split_count + 1
    else:
        selected = [issue for issue in issues if issue["split"] in splits]
    if not selected:
        raise ValueError("No issues found in the selected split(s)")

    priors = base_rates(directory)
    cache = output / "e2d" / "beliefs"
    cache.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    cache_hits = 0
    inference_seconds = 0.0

    for issue_index, issue in enumerate(selected, start=1):
        print(
            f"Direct belief {issue_index}/{len(selected)}: {issue['id']}",
            flush=True,
        )
        for module_id in range(len(MODULES)):
            print(
                f"  Module {module_id + 1}/{len(MODULES)}: {MODULES[module_id]}",
                flush=True,
            )
            record_path = cache / f"{issue['id']}__{module_id}__{mode}.json"
            snippets = None
            if mode == "snippets":
                if repository is None:
                    raise ValueError("repository is required for snippets mode")
                snippets = local_snippets(
                    repository,
                    issue["base"],
                    module_id,
                    issue["text"],
                )

            system, prompt = direct_prompt(
                issue,
                module_id,
                float(priors[module_id]),
                mode,
                snippets,
            )
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "issue_id": issue["id"],
                        "base": issue["base"],
                        "module_id": module_id,
                        "mode": mode,
                        "provider": type(provider).__name__,
                        "model": getattr(provider, "model", type(provider).__name__),
                        "endpoint": getattr(provider, "base_url", None),
                        "cutoff": os.getenv("E2_MODEL_CUTOFF", "unknown"),
                        "system": system,
                        "prompt": prompt,
                        "temperature": 0.0,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

            if record_path.exists():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                if cached.get("signature") == signature:
                    result = cached
                else:
                    result = None
            else:
                result = None

            if result is None:
                try:
                    call_started = time.perf_counter()
                    raw = provider.complete(system, prompt)
                    call_seconds = time.perf_counter() - call_started
                    inference_seconds += call_seconds
                    usage = getattr(provider, "last_usage", {})
                    result = validate_direct_response(
                        raw,
                        repository,
                        issue["base"],
                        module_id,
                        snippets if mode == "snippets" else [],
                    )
                    result.update(
                        {
                            "issue_id": issue["id"],
                            "split": issue["split"],
                            "module_id": module_id,
                            "module": MODULES[module_id],
                            "base": issue["base"],
                            "mode": mode,
                            "raw_response": raw,
                            "provider": getattr(provider, "model", type(provider).__name__),
                            "model_cutoff": os.getenv("E2_MODEL_CUTOFF", "unknown"),
                            "signature": signature,
                            "temperature": 0.0,
                            "inference_seconds": call_seconds,
                            "usage": usage,
                        }
                    )
                    record_path.write_text(
                        json.dumps(result, sort_keys=True, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    failure = {
                        "issue_id": issue["id"],
                        "module_id": module_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(
                        f"  Failed {issue['id']} / {MODULES[module_id]}: "
                        f"{failure['error']}",
                        flush=True,
                    )
                    continue
            else:
                cache_hits += 1

            rows.append(
                {
                    "issue": issue["id"],
                    "split": issue["split"],
                    "module": MODULES[module_id],
                    "module_id": module_id,
                    "probability": result["probability"],
                    "base_rate": float(priors[module_id]),
                    "citations_accepted": len(result["citations"]),
                    "citations_rejected": result["citations_rejected"],
                    "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": result.get("usage", {}).get("completion_tokens"),
                    "inference_seconds": result.get("inference_seconds"),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "e2d-beliefs.csv", index=False)
    (output / "e2d-failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )
    token_rows = [
        row
        for row in rows
        if row["prompt_tokens"] is not None
        and row["completion_tokens"] is not None
    ]
    (output / "e2d-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": DIRECT_MANIFEST_VERSION,
                "experiment": "E2-D direct verbalized confidence",
                "description": "One LLM call per issue/module; no head or LLM fine-tuning",
                "mode": mode,
                "provider": type(provider).__name__,
                "model": getattr(provider, "model", type(provider).__name__),
                "endpoint": getattr(provider, "base_url", None),
                "model_cutoff": os.getenv("E2_MODEL_CUTOFF", "unknown"),
                "temperature": 0.0,
                "splits": list(splits),
                "issue_ids": [issue["id"] for issue in selected],
                "issues": len(selected),
                "module_calls": len(selected) * len(MODULES),
                "completed_calls": len(rows) + len(failures),
                "successful_rows": len(rows),
                "failures": len(failures),
                "cache_hits": cache_hits,
                "inference_seconds": inference_seconds,
                "token_usage_calls": len(token_rows),
                "prompt_tokens": (
                    sum(int(row["prompt_tokens"]) for row in token_rows)
                    if token_rows
                    else None
                ),
                "completion_tokens": (
                    sum(int(row["completion_tokens"]) for row in token_rows)
                    if token_rows
                    else None
                ),
                "base_rates": priors.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Completed {len(rows)}/{len(selected) * len(MODULES)} module beliefs "
        f"({cache_hits} cache hits, {len(failures)} failures; "
        f"{inference_seconds:.1f}s inference this run).",
        flush=True,
    )


def replay(
    directory: Path,
    output: Path,
) -> None:
    """Replay direct beliefs through the E1 reporting and assignment machinery."""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "e2d-manifest.json"
    if not manifest_path.exists():
        raise ValueError("No E2-D manifest found; run extraction before replay")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DIRECT_MANIFEST_VERSION:
        raise ValueError(
            "E2-D output is missing the current manifest version; rerun extraction "
            "before replay"
        )
    try:
        beliefs = pd.read_csv(output / "e2d-beliefs.csv")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("No valid E2-D beliefs are available to replay") from exc
    required_columns = {"issue", "split", "module_id", "probability"}
    if not required_columns.issubset(beliefs.columns):
        raise ValueError("E2-D belief CSV is missing required columns")
    if len(beliefs) != manifest.get("successful_rows"):
        raise ValueError("Belief rows do not match the E2-D manifest")
    if beliefs.duplicated(["issue", "module_id"]).any():
        raise ValueError("E2-D beliefs contain duplicate issue/module rows")
    if not set(beliefs["issue"]).issubset(set(manifest["issue_ids"])):
        raise ValueError("E2-D beliefs include issues outside the manifest cohort")
    module_ids = pd.to_numeric(beliefs["module_id"], errors="raise").to_numpy()
    if (
        not np.equal(module_ids, np.floor(module_ids)).all()
        or np.any(module_ids < 0)
        or np.any(module_ids >= len(MODULES))
    ):
        raise ValueError("E2-D beliefs contain an invalid module ID")
    probabilities = pd.to_numeric(beliefs["probability"], errors="raise").to_numpy()
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("E2-D beliefs contain a probability outside [0, 1]")
    lexical_cases, _ = predictors(directory)
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))

    direct_matrix = np.array(lexical_cases.local, dtype=float, copy=True)
    direct_matrix[np.asarray(lexical_cases.split) == "validation"] = np.nan
    direct_matrix[np.asarray(lexical_cases.split) == "test"] = np.nan
    for split_name in ("validation", "test"):
        ids = np.asarray(lexical_cases.ids)[
            np.asarray(lexical_cases.split) == split_name
        ]
        split_beliefs = beliefs[beliefs["split"] == split_name]
        for row_index, issue_id in enumerate(ids):
            issue_rows = split_beliefs[split_beliefs["issue"] == issue_id]
            for _, belief in issue_rows.iterrows():
                global_index = np.flatnonzero(
                    np.asarray(lexical_cases.ids) == issue_id
                )[0]
                direct_matrix[global_index, int(belief["module_id"])] = float(
                    belief["probability"]
                )
    cases = replace(lexical_cases, local=direct_matrix)

    complete_validation_ids = complete_belief_issues(beliefs, "validation")
    complete_test_ids = complete_belief_issues(beliefs, "test")
    issue_ids = np.asarray(cases.ids)
    split_names = np.asarray(cases.split)
    validation_mask = (split_names == "validation") & np.isin(
        issue_ids, list(complete_validation_ids)
    )
    test_mask = (split_names == "test") & np.isin(
        issue_ids, list(complete_test_ids)
    )
    validation_ids = issue_ids[validation_mask]
    validation_y = np.zeros((len(validation_ids), len(MODULES)), dtype=int)
    for row_index, issue_id in enumerate(validation_ids):
        validation_y[row_index, labels[issue_id]["owners"]] = 1

    test_ids = issue_ids[test_mask]
    test_y = np.zeros((len(test_ids), len(MODULES)), dtype=int)
    for row_index, issue_id in enumerate(test_ids):
        test_y[row_index, labels[issue_id]["owners"]] = 1

    validation_complete = len(validation_ids) > 0

    rows: list[pd.DataFrame] = []
    diagnostics = []

    if len(test_ids):
        diagnostics.append(
            {
                "status": "test_replay",
                "complete_test_issues": len(test_ids),
                "mechanism_fit": (
                    "fit on complete validation direct forecasts"
                    if validation_complete
                    else "skipped: no complete validation direct forecasts"
                ),
                "regret_reference": (
                    "E1 B4 lexical pooled estimate; not a direct-evidence oracle"
                ),
            }
        )

    if not validation_complete:
        diagnostics.append(
            {
                "status": "mechanism_replay_not_run",
                "reason": "No complete validation direct-belief matrix",
                "complete_validation_issues": len(validation_ids),
                "available_validation_issues": int(
                    np.count_nonzero(split_names == "validation")
                ),
            }
        )

    conditions = []
    if validation_complete:
        conditions.append(Condition())
        for scale, audit in (
            (2.0, 0.0),
            (8.0, 0.0),
            (0.0, 0.25),
            (2.0, 0.25),
        ):
            conditions.append(Condition(scale=scale, audit=audit))
        for ell, phi in ((-1.0, 1 / 3), (-1.0, 2 / 3)):
            for scale, audit in (
                (0.0, 0.0),
                (2.0, 0.0),
                (8.0, 0.0),
                (0.0, 0.25),
                (2.0, 0.25),
            ):
                conditions.append(Condition(ell=ell, phi=phi, scale=scale, audit=audit))
        for credit in ("equal", "shapley"):
            conditions.append(
                Condition(
                    ell=-1.0,
                    phi=2 / 3,
                    scale=2.0,
                    audit=0.25,
                    credit=credit,
                )
            )

    evidence_cases = (("direct_llm", cases), ("e1_lexical", lexical_cases))
    for condition in conditions:
        fitted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for evidence_name, evidence_case in evidence_cases:
            policy, thresholds, diagnostic = fit_strategy(
                evidence_case.local[validation_mask],
                validation_y,
                evidence_case.degree[validation_mask],
                evidence_case.prior,
                condition,
                331,
            )
            config = (
                "B2-honest"
                if condition == Condition()
                else (
                    "B2-strategic"
                    if condition.ell == -1.0
                    and condition.phi == 2 / 3
                    and condition.scale == 0
                    and condition.audit == 0
                    and condition.credit == "off"
                    else tag(condition)
                )
            )
            diagnostic.update(
                {
                    "evidence": evidence_name,
                    "config": config,
                    "regret_reference": "E1 B4 lexical pooled estimate",
                }
            )
            diagnostics.append(diagnostic)
            fitted[evidence_name] = (policy, thresholds)

        for split_name, mask in (
            ("validation", validation_mask),
            ("test", test_mask),
        ):
            if not np.any(mask):
                continue
            ids = np.asarray(cases.ids)[mask]
            outcomes = validation_y if split_name == "validation" else test_y
            seed = 331 if split_name == "validation" else 332
            for evidence_name, evidence_case in evidence_cases:
                fixed = comparators(evidence_case, mask, condition.reward)
                policy, thresholds = fitted[evidence_name]
                outcome_rows = mechanism_outcomes(
                    evidence_case,
                    ids,
                    outcomes,
                    mask,
                    condition,
                    policy,
                    thresholds,
                    seed,
                    fixed,
                    evidence_name,
                )
                outcome_rows["split"] = split_name
                rows.append(outcome_rows)

    if not validation_complete and len(test_ids):
        for evidence_name, evidence_case in evidence_cases:
            fixed = comparators(evidence_case, test_mask, 8)
            chosen = select(evidence_case.local[test_mask], 8)
            honest = settle(test_ids, test_y, chosen, fixed["B4"], 8)
            honest["evidence"] = evidence_name
            honest["mode"] = (
                "direct_llm_probability" if evidence_name == "direct_llm" else "e1_lexical"
            )
            honest["config"] = "B2-honest"
            honest["method"] = "Honest routing"
            honest["routing_label"] = (
                "B2-direct" if evidence_name == "direct_llm" else "B2-E1-lexical"
            )
            honest["split"] = "test"
            honest["regret_reference"] = "E1 lexical B4 estimate"
            honest["shading"] = 0.0
            honest["score"] = 0.0
            honest["audit_spend"] = 0.0
            honest["incentive_cost"] = 0.0
            honest["fine_paid"] = 0.0
            honest["credit_paid"] = 0.0
            honest["credit_to_extra"] = 0.0
            rows.append(honest)

    for split_name, mask in (
        ("validation", validation_mask),
        ("test", test_mask),
    ):
        if not np.any(mask):
            continue
        ids = np.asarray(cases.ids)[mask]
        outcomes = validation_y if split_name == "validation" else test_y
        for evidence_name, evidence_case in evidence_cases:
            fixed = comparators(evidence_case, mask, 8)
            for baseline_name, selection in fixed.items():
                baseline = settle(ids, outcomes, selection, fixed["B4"], 8)
                baseline["evidence"] = evidence_name
                baseline["mode"] = (
                    "direct_llm_probability"
                    if evidence_name == "direct_llm"
                    else "e1_lexical"
                )
                baseline["config"] = (
                    "B4-E1-lexical" if baseline_name == "B4" else baseline_name
                )
                baseline["method"] = baseline_name
                baseline["routing_label"] = baseline_name
                baseline["split"] = split_name
                baseline["regret_reference"] = "E1 lexical B4 estimate"
                baseline["shading"] = 0.0
                baseline["score"] = 0.0
                baseline["audit_spend"] = 0.0
                baseline["incentive_cost"] = 0.0
                baseline["fine_paid"] = 0.0
                baseline["credit_paid"] = 0.0
                baseline["credit_to_extra"] = 0.0
                rows.append(baseline)

    replay = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    recovery_columns = (
        "gross_recovery_vs_strategic_b2",
        "net_recovery_vs_strategic_b2",
    )
    for column in recovery_columns:
        replay[column] = np.nan
    strategic_mask = replay["config"] == "B2-strategic"
    strategic = replay[strategic_mask].set_index(["split", "evidence", "issue"])
    recovery_mask = pd.Series(False, index=replay.index)
    if {"ell", "phi", "credit"}.issubset(replay.columns):
        recovery_mask = (
            (replay["ell"] == -1.0)
            & (replay["phi"] == 2 / 3)
            & (replay["credit"] == "off")
            & (replay["method"].isin(["Mechanism M", "Strategic B2"]))
        )
    for row_index in replay.index[recovery_mask]:
        row = replay.loc[row_index]
        key = (row["split"], row["evidence"], row["issue"])
        if key not in strategic.index:
            continue
        gross = float(row["welfare"] - strategic.loc[key, "welfare"])
        replay.at[row_index, "gross_recovery_vs_strategic_b2"] = gross
        replay.at[row_index, "net_recovery_vs_strategic_b2"] = (
            gross - float(row["incentive_cost"])
        )
    replay.to_csv(output / "e2d-replay.csv", index=False)
    summarize_replay(replay).to_csv(output / "e2d-summary.csv", index=False)
    predictions = beliefs.copy()
    outcome_rows = [
        {
            "issue": issue_id,
            "module_id": owner,
            "value": 1,
        }
        for issue_id, record in labels.items()
        for owner in record["owners"]
    ]
    prediction_rows = predictions.merge(
        pd.DataFrame(outcome_rows),
        how="left",
        on=["issue", "module_id"],
    )
    prediction_rows["outcome"] = (
        prediction_rows["value"].fillna(0).astype(int)
    )
    prediction_rows.drop(columns=["value"], inplace=True, errors="ignore")
    lexical_lookup = {
        (issue_id, module_id): float(lexical_cases.local[issue_index, module_id])
        for issue_index, issue_id in enumerate(lexical_cases.ids)
        for module_id in range(len(MODULES))
    }
    prediction_rows["e1_lexical_probability"] = [
        lexical_lookup[(issue_id, int(module_id))]
        for issue_id, module_id in zip(
            prediction_rows["issue"],
            prediction_rows["module_id"],
        )
    ]
    prediction_rows.to_csv(output / "e2d-predictions.csv", index=False)

    calibration_rows = []
    for split_name in ("validation", "test"):
        split_predictions = prediction_rows[
            prediction_rows["split"] == split_name
        ]
        for evidence_name, probability_column in (
            ("direct_llm", "probability"),
            ("e1_lexical", "e1_lexical_probability"),
        ):
            groups = [("overall", split_predictions)] + [
                (
                    module_name,
                    split_predictions[split_predictions["module_id"] == module_id],
                )
                for module_id, module_name in enumerate(MODULES)
            ]
            for module_name, module_predictions in groups:
                probabilities = module_predictions[probability_column].to_numpy(
                    dtype=float
                )
                outcomes = module_predictions["outcome"].to_numpy(dtype=int)
                row = {
                    "split": split_name,
                    "module": module_name,
                    "evidence": evidence_name,
                    "count": len(outcomes),
                    "positive_count": int(outcomes.sum()),
                    "negative_count": int(len(outcomes) - outcomes.sum()),
                    "brier_loss": (
                        float(np.mean((probabilities - outcomes) ** 2))
                        if len(outcomes)
                        else None
                    ),
                    "ece_10_bin": (
                        MetricsEvaluator.compute_ece(
                            probabilities,
                            outcomes,
                            n_bins=10,
                        )
                        if len(outcomes)
                        else None
                    ),
                    "base_rate": (
                        float(np.mean(outcomes)) if len(outcomes) else None
                    ),
                    "roc_auc": (
                        float(roc_auc_score(outcomes, probabilities))
                        if np.unique(outcomes).size == 2
                        else None
                    ),
                }
                calibration_rows.append(row)
    pd.DataFrame(calibration_rows).to_csv(
        output / "e2d-calibration.csv",
        index=False,
    )
    paired_effects(
        replay,
        prediction_rows,
        {
            "validation": complete_validation_ids,
            "test": complete_test_ids,
        },
    ).to_csv(output / "e2d-paired-summary.csv", index=False)

    (output / "e2d-replay-diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository")
    parser.add_argument(
        "--mode",
        choices=("no_tools", "snippets"),
        default="no_tools",
    )
    parser.add_argument("--max-issues", type=int)
    parser.add_argument("--split", action="append", choices=("validation", "test"))
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay cached direct beliefs after extraction",
    )
    parser.add_argument(
        "--compare-mode",
        type=Path,
        help="Compare this replay with the other mode's output directory",
    )
    args = parser.parse_args()
    if args.compare_mode is not None:
        print(compare_modes(args.output, args.compare_mode))
        raise SystemExit(0)
    if args.replay:
        replay(args.directory, args.output)
        raise SystemExit(0)
    splits = tuple(args.split) if args.split else ("validation", "test")
    run(
        args.directory,
        args.output,
        args.repository,
        splits,
        args.max_issues,
        args.mode,
    )
