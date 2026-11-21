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
        f"Issue ID: {issue['id']}\n"
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
    group_columns = ["split", "config", "method", "regret_reference"]
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
            record_path = cache / f"{issue['id']}__{module_id}.json"
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
    cases, _ = predictors(directory)
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))

    direct_matrix = np.array(cases.local, dtype=float, copy=True)
    direct_matrix[np.asarray(cases.split) == "validation"] = np.nan
    direct_matrix[np.asarray(cases.split) == "test"] = np.nan
    for split_name in ("validation", "test"):
        ids = np.asarray(cases.ids)[np.asarray(cases.split) == split_name]
        split_beliefs = beliefs[beliefs["split"] == split_name]
        for row_index, issue_id in enumerate(ids):
            issue_rows = split_beliefs[split_beliefs["issue"] == issue_id]
            for _, belief in issue_rows.iterrows():
                global_index = np.flatnonzero(np.asarray(cases.ids) == issue_id)[0]
                direct_matrix[global_index, int(belief["module_id"])] = float(
                    belief["probability"]
                )
    cases = replace(cases, local=direct_matrix)

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

    complete_test_positions = [
        position
        for position, issue_id in enumerate(test_ids)
        if issue_id in complete_test_ids
    ]
    if complete_test_positions:
        complete_test_mask = np.zeros(len(cases.ids), dtype=bool)
        complete_test_global_indices = [
            int(np.flatnonzero(np.asarray(cases.ids) == test_ids[position])[0])
            for position in complete_test_positions
        ]
        complete_test_mask[complete_test_global_indices] = True
        complete_test_ids_array = np.asarray(cases.ids)[complete_test_mask]
        complete_test_y = np.vstack(
            [
                np.asarray(
                    [int(module_id in labels[issue_id]["owners"]) for module_id in range(len(MODULES))],
                    dtype=int,
                )
                for issue_id in complete_test_ids_array
            ]
        )
        fixed = comparators(cases, complete_test_mask, 8)
        direct_chosen = select(
            direct_matrix[complete_test_mask],
            8,
        )
        direct_row = settle(
            complete_test_ids_array,
            complete_test_y,
            direct_chosen,
            fixed["B4"],
            8,
        )
        direct_row["evidence"] = "verbalized_confidence"
        direct_row["mode"] = "direct_llm_probability"
        direct_row["split"] = "test"
        direct_row["config"] = "B2-direct"
        direct_row["method"] = "B2 direct"
        direct_row["shading"] = 0.0
        direct_row["score"] = 0.0
        direct_row["audit_spend"] = 0.0
        direct_row["incentive_cost"] = 0.0
        direct_row["fine_paid"] = 0.0
        direct_row["credit_paid"] = 0.0
        direct_row["credit_to_extra"] = 0.0
        direct_row["regret_reference"] = "E1 lexical B4 estimate"
        rows.append(direct_row)
        for baseline_name, selection in fixed.items():
            baseline = settle(
                complete_test_ids_array,
                complete_test_y,
                selection,
                fixed["B4"],
                8,
            )
            baseline["evidence"] = "verbalized_confidence"
            baseline["mode"] = "direct_llm_probability"
            baseline["split"] = "test"
            baseline["config"] = (
                "B4-E1-lexical" if baseline_name == "B4" else baseline_name
            )
            baseline["method"] = baseline_name
            baseline["shading"] = 0.0
            baseline["score"] = 0.0
            baseline["audit_spend"] = 0.0
            baseline["incentive_cost"] = 0.0
            baseline["fine_paid"] = 0.0
            baseline["credit_paid"] = 0.0
            baseline["credit_to_extra"] = 0.0
            baseline["regret_reference"] = "E1 lexical B4 estimate"
            rows.append(baseline)
        diagnostics.append(
            {
                "status": "test_only_direct_replay",
                "complete_test_issues": len(complete_test_ids),
                "mechanism_fit": "skipped: no complete validation direct beliefs",
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
        for scale, audit in ((2.0, 0.0), (8.0, 0.0), (2.0, 0.25)):
            conditions.append(Condition(scale=scale, audit=audit))
        for ell, phi in ((-1.0, 1 / 3), (-1.0, 2 / 3)):
            for scale, audit in ((0.0, 0.0), (2.0, 0.0), (8.0, 0.0), (2.0, 0.25)):
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

    for c in conditions:
        policy, thresholds, diagnostic = fit_strategy(
            cases.local[validation_mask],
            validation_y,
            cases.degree[validation_mask],
            cases.prior,
            c,
            331,
        )
        diagnostic.update({"config": tag(c)})
        diagnostic["regret_reference"] = (
            "E1 B4 lexical pooled estimate; not a direct-evidence oracle"
        )
        diagnostics.append(diagnostic)

        for split_name, mask in (
            ("validation", validation_mask),
            ("test", test_mask),
        ):
            if not np.any(mask):
                continue
            ids = np.asarray(cases.ids)[mask]
            y = validation_y if split_name == "validation" else test_y
            seed = 331 if split_name == "validation" else 332
            fixed = comparators(cases, mask, c.reward)
            chosen, reports, caught, checked = decisions(
                cases,
                mask,
                c,
                policy,
                thresholds,
                seed,
            )
            row = settle(ids, y, chosen, fixed["B4"], c.reward)
            row["evidence"] = "verbalized_confidence"
            row["mode"] = "direct_llm_probability"
            row["split"] = split_name
            row["config"] = tag(c)
            row["method"] = "Mechanism M"
            row["regret_reference"] = "E1 lexical B4 estimate"
            row["shading"] = (
                reports - cases.local[mask]
            ).mean(axis=1)
            centered = (
                -(y - reports) ** 2
                + (y - cases.prior[None, :]) ** 2
            )
            row["score"] = c.scale * centered.sum(axis=1)
            row["audit_spend"] = c.audit_cost * checked.sum(axis=1)
            row["incentive_cost"] = row["score"] + row["audit_spend"]
            row["fine_paid"] = c.fine * caught.sum(axis=1)
            row["credit_paid"] = 0.0
            row["credit_to_extra"] = 0.0

            if c.credit != "off":
                credit_rule = (
                    CreditRule.SHAPLEY
                    if c.credit == "shapley"
                    else CreditRule.EQUAL_SPLIT
                )
                credit_mechanism = AssignmentMechanism(
                    MechanismConfig(credit_rule=credit_rule)
                )
                for issue_index, (outcome, chosen_mask) in enumerate(zip(y, chosen)):
                    truth = set(np.flatnonzero(outcome))
                    assigned_set = set(
                        np.flatnonzero(chosen_mask & (1 << np.arange(len(MODULES))))
                    )
                    task = Task(
                        issue_index,
                        np.array([]),
                        -1,
                        truth,
                        outcome,
                        c.reward,
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
            rows.append(row)

            if c == Condition():
                for baseline_name, selection in fixed.items():
                    baseline = settle(ids, y, selection, fixed["B4"], c.reward)
                    baseline["evidence"] = "verbalized_confidence"
                    baseline["mode"] = "direct_llm_probability"
                    baseline["split"] = split_name
                    baseline["config"] = (
                        "B4-E1-lexical" if baseline_name == "B4" else baseline_name
                    )
                    baseline["method"] = baseline_name
                    baseline["shading"] = 0.0
                    baseline["score"] = 0.0
                    baseline["audit_spend"] = 0.0
                    baseline["incentive_cost"] = 0.0
                    baseline["fine_paid"] = 0.0
                    baseline["credit_paid"] = 0.0
                    baseline["credit_to_extra"] = 0.0
                    baseline["regret_reference"] = "E1 lexical B4 estimate"
                    rows.append(baseline)

    replay = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
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
    prediction_rows.to_csv(output / "e2d-predictions.csv", index=False)

    calibration_rows = []
    for split_name in ("validation", "test"):
        split_predictions = prediction_rows[
            prediction_rows["split"] == split_name
        ]
        groups = [("overall", split_predictions)] + [
            (
                module_name,
                split_predictions[split_predictions["module_id"] == module_id],
            )
            for module_id, module_name in enumerate(MODULES)
        ]
        for module_name, module_predictions in groups:
            probabilities = module_predictions["probability"].to_numpy(dtype=float)
            outcomes = module_predictions["outcome"].to_numpy(dtype=int)
            row = {
                "split": split_name,
                "module": module_name,
                "count": len(outcomes),
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
                "base_rate": float(np.mean(outcomes)) if len(outcomes) else None,
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
    args = parser.parse_args()
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
