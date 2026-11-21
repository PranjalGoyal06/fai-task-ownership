"""Paired lexical/semantic replay for the grounded E2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.e1.dataset import MODULES
from src.e1.game import Condition, fit_strategy
from src.e1.replay import comparators, decisions, settle, tag
from src.e2.models import paired_predictors
from src.metrics import MetricsEvaluator


def paired_conditions() -> list[Condition]:
    """Primary E2 conditions: paired control, incentives, score and budget profiles."""
    conditions: list[Condition] = []
    conditions.append(Condition())
    for ell, phi in ((-1.0, 1 / 3), (-1.0, 2 / 3), (1.0, 1 / 3), (1.0, 2 / 3)):
        for scale, audit in (
            (0.0, 0.0),
            (2.0, 0.0),
            (8.0, 0.0),
            (0.0, 0.25),
            (2.0, 0.25),
        ):
            conditions.append(
                Condition(ell=ell, phi=phi, scale=scale, audit=audit)
            )
    conditions.append(Condition(ell=-1.0, phi=2 / 3, scale=2.0, audit=0.25, credit="equal"))
    conditions.append(Condition(ell=-1.0, phi=2 / 3, scale=2.0, audit=0.25, credit="shapley"))
    return conditions


def run(
    directory: Path,
    output: Path,
    issues_per_split: int = 50,
) -> None:
    lex_cases, sem_cases, validation_y = paired_predictors(
        directory,
        issues_per_split=issues_per_split,
    )
    labels = json.loads((directory / "labels.json").read_text())
    partitions = {
        "validation": lex_cases.split == "validation",
        "test": lex_cases.split == "test",
    }
    rows: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    selected_targets = np.zeros((len(lex_cases.ids), len(MODULES)), dtype=int)
    for issue_index, issue_id in enumerate(lex_cases.ids):
        selected_targets[issue_index, labels[issue_id]["owners"]] = 1

    for evidence_name, cases in (("lexical", lex_cases), ("semantic", sem_cases)):
        validation = partitions["validation"]
        for condition in paired_conditions():
            policy, thresholds, diagnostic = fit_strategy(
                cases.local[validation],
                validation_y,
                cases.degree[validation],
                cases.prior,
                condition,
                331,
            )
            diagnostic.update(
                {
                    "evidence": evidence_name,
                    "config": tag(condition),
                    "issue_count": int(validation.sum()),
                }
            )
            diagnostics.append(diagnostic)

            for split_name, mask in partitions.items():
                fixed = comparators(cases, mask, condition.reward)
                seed = 331 if split_name == "validation" else 332
                chosen, reports, caught, checked = decisions(
                    cases,
                    mask,
                    condition,
                    policy,
                    thresholds,
                    seed,
                )
                ids = np.asarray(cases.ids)[mask]
                y = np.zeros((len(ids), len(MODULES)), dtype=int)
                for issue_index, issue_id in enumerate(ids):
                    y[issue_index, labels[issue_id]["owners"]] = 1
                row = settle(ids, y, chosen, fixed["B4"], condition.reward)
                row["evidence"] = evidence_name
                row["config"] = tag(condition)
                row["split"] = split_name
                row["method"] = "Mechanism M"
                row["audit_spend"] = condition.audit_cost * checked.sum(axis=1)
                row["contradictions"] = caught.sum(axis=1)
                row["audits"] = checked.sum(axis=1)
                row["shading"] = (reports - cases.local[mask]).mean(axis=1)
                centered = (
                    -(y - reports) ** 2 + (y - cases.prior[None, :]) ** 2
                )
                row["score"] = condition.scale * centered.sum(axis=1)
                row["incentive_cost"] = (
                    row["score"] + row["audit_spend"]
                )
                row["fine_paid"] = condition.fine * caught.sum(axis=1)
                row["credit_paid"] = np.where(
                    (row["coverage"].to_numpy() > 0) & (condition.credit != "off"),
                    condition.reward,
                    0.0,
                )
                if condition.credit == "equal":
                    assigned_count = np.maximum(1, row["assigned"].map(int).map(int.bit_count))
                    row["credit_to_extra"] = (
                        condition.reward
                        * row["padding"].to_numpy()
                        / assigned_count.to_numpy()
                    ) * (row["coverage"].to_numpy() > 0)
                else:
                    row["credit_to_extra"] = 0.0
                row["net_welfare_before_extraction"] = (
                    row["welfare"] - row["incentive_cost"]
                )
                rows.append(row)

                if condition == Condition():
                    ids_array = np.asarray(cases.ids)[mask]
                    for baseline_name, selection in fixed.items():
                        baseline = settle(
                            ids_array,
                            y,
                            selection,
                            fixed["B4"],
                            condition.reward,
                        )
                        baseline["evidence"] = evidence_name
                        baseline["config"] = baseline_name
                        baseline["split"] = split_name
                        baseline["method"] = baseline_name
                        baseline["audits"] = 0
                        baseline["contradictions"] = 0
                        baseline["audit_spend"] = 0.0
                        baseline["score"] = 0.0
                        baseline["incentive_cost"] = 0.0
                        baseline["fine_paid"] = 0.0
                        baseline["credit_paid"] = 0.0
                        baseline["credit_to_extra"] = 0.0
                        baseline["net_welfare_before_extraction"] = baseline["welfare"]
                        baseline["shading"] = 0.0
                        rows.append(baseline)

    output.mkdir(parents=True, exist_ok=True)
    replay = pd.concat(rows, ignore_index=True)
    replay.to_csv(output / "paired-replay.csv", index=False)

    prediction_rows = []
    calibration_rows = []
    masks = {
        "train": lex_cases.split == "train",
        "validation": lex_cases.split == "validation",
        "test": lex_cases.split == "test",
    }
    for evidence_name, cases in (("lexical", lex_cases), ("semantic", sem_cases)):
        for split_name, mask in masks.items():
            for module_id, module_name in enumerate(MODULES):
                probabilities = cases.local[mask, module_id]
                outcomes = selected_targets[mask, module_id]
                for issue_id, probability, outcome in zip(
                    np.asarray(cases.ids)[mask],
                    probabilities,
                    outcomes,
                ):
                    prediction_rows.append(
                        {
                            "issue": issue_id,
                            "split": split_name,
                            "module": module_name,
                            "evidence": evidence_name,
                            "outcome": int(outcome),
                            "probability": float(probability),
                        }
                    )
                calibration_rows.append(
                    {
                        "split": split_name,
                        "module": module_name,
                        "evidence": evidence_name,
                        "count": len(outcomes),
                        "brier_loss": float(np.mean((probabilities - outcomes) ** 2)),
                        "ece_10_bin": MetricsEvaluator.compute_ece(
                            probabilities,
                            outcomes,
                            n_bins=10,
                        ),
                        "roc_auc": (
                            float(roc_auc_score(outcomes, probabilities))
                            if np.unique(outcomes).size == 2
                            else float("nan")
                        ),
                        "base_rate": float(np.mean(outcomes)),
                    }
                )
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        output / "calibration.csv",
        index=False,
    )

    paired_rows = []
    test_mechanisms = replay[
        (replay["split"] == "test") & (replay["method"] == "Mechanism M")
    ]
    bootstrap_rng = np.random.default_rng(20261121)
    paired_metrics = (
        "welfare",
        "coverage",
        "padding",
        "any_padding",
        "miss_loss",
        "excess_cost",
        "incentive_cost",
        "fine_paid",
        "credit_paid",
        "net_welfare_before_extraction",
    )
    for config in test_mechanisms["config"].drop_duplicates():
        lexical_rows = test_mechanisms[
            (test_mechanisms["config"] == config)
            & (test_mechanisms["evidence"] == "lexical")
        ].set_index("issue")
        semantic_rows = test_mechanisms[
            (test_mechanisms["config"] == config)
            & (test_mechanisms["evidence"] == "semantic")
        ].set_index("issue")
        shared_ids = lexical_rows.index.intersection(semantic_rows.index)
        for metric in paired_metrics:
            differences = (
                semantic_rows.loc[shared_ids, metric].to_numpy(dtype=float)
                - lexical_rows.loc[shared_ids, metric].to_numpy(dtype=float)
            )
            samples = bootstrap_rng.choice(
                differences,
                size=(2000, len(differences)),
                replace=True,
            ).mean(axis=1)
            paired_rows.append(
                {
                    "config": config,
                    "metric": metric,
                    "count": len(differences),
                    "semantic_minus_lexical": float(np.mean(differences)),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                }
            )
    pd.DataFrame(paired_rows).to_csv(
        output / "paired-summary.csv",
        index=False,
    )
    (output / "e2-diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    issue_ids = {
        split_name: [
            issue_id
            for issue_id, issue_split in zip(lex_cases.ids, lex_cases.split)
            if issue_split == split_name
        ]
        for split_name in ("train", "validation", "test")
    }
    cohort_sha256 = hashlib.sha256(
        json.dumps(issue_ids, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (output / "cohort.json").write_text(
        json.dumps(
            {
                "selection": (
                    f"first {issues_per_split} issues per frozen E1 split, "
                    "in chronological order"
                ),
                "pilot_issue_included_as_training_issue": "django__django-5158",
                "issues_per_split": issues_per_split,
                "evidence_model": "gemini-3.5-flash-lite",
                "model_cutoff": "unknown",
                "issue_ids": issue_ids,
                "cohort_sha256": cohort_sha256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--issues-per-split", type=int, default=50)
    args = parser.parse_args()
    run(args.directory, args.output, args.issues_per_split)
