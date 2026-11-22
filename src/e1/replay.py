"""Replay every policy on the same cases, then reveal ownership for settlement."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.e1.dataset import MODULES, dump
from src.e1.game import BITS, Condition, fit_strategy, information, population, select, submit
from src.e1.models import predictors
from src.mechanisms import AssignmentMechanism
from src.types import CreditRule, MechanismConfig, ResolutionRule, Task

SEED = 331


def comparators(cases, subset, reward):
    local = cases.local[subset]
    count = local.shape[1]
    votes = []
    for i in range(count):
        public = np.tile(cases.prior, (len(local), 1))
        public[:, i] = local[:, i]
        votes.append(public.argmax(axis=1))
    tally = np.eye(count)[votes].sum(axis=0)
    masks = np.arange(1 << count)
    costs = np.array([i.bit_count() for i in range(1 << count)])
    known = cases.joint_masks[subset, :, None]
    covered = (masks & known) == known
    objective = reward * (cases.joint_weights[subset, :, None] * covered).sum(axis=1) - costs
    return {"B1": select(cases.descriptions[subset], reward), "B3": 1 << tally.argmax(axis=1),
            "B4": objective.argmax(axis=1), "central": select(cases.central[subset], reward)}


def conditions():
    for ell, phi in [(0., 0.), (-1., 1/3), (-1., 2/3), (1., 1/3), (1., 2/3)]:
        base = Condition(ell=ell, phi=phi)
        for scale in (0., 2., 8.):
            for audit in (0., .25):
                yield replace(base, scale=scale, audit=audit)
        for credit in ("equal", "shapley"):
            yield replace(base, scale=2., audit=.25, credit=credit)
    for reward in (5., 12.):
        yield Condition(ell=-1., phi=2/3, reward=reward)
    for delta in (.1, .8):
        yield Condition(ell=-1., phi=2/3, audit=.25, delta=delta)


def tag(c):
    return (f"ell{c.ell:g}_phi{c.phi:.3f}_score{c.scale:g}_audit{c.audit:g}"
            f"_{c.credit}_R{c.reward:g}_delta{c.delta:g}_F{c.fine:g}")


def decisions(cases, mask, c, policy, thresholds, seed):
    p = cases.local[mask]
    types, checked, exposed = population(seed, len(p), c.phi, len(MODULES))
    info = information(p, thresholds, types)
    r, effective, caught = submit(policy, p, info, (checked, exposed), cases.degree[mask],
                                  cases.prior, c)
    return select(effective, c.reward), r, caught, checked < c.audit


def settle(ids, y, chosen, b4, reward):
    count = y.shape[1]
    sizes = np.array([i.bit_count() for i in range(1 << count)])
    targets = y @ (1 << np.arange(count))
    covered = (chosen & targets) == targets
    oracle_covered = (b4 & targets) == targets
    size = sizes[chosen]
    extra = sizes[chosen & ~targets]
    miss = reward * (oracle_covered.astype(float) - covered)
    excess = size - sizes[b4]
    return pd.DataFrame(dict(issue=ids, owners=targets, k=sizes[targets], assigned=chosen,
                             coverage=covered.astype(int), padding=extra, any_padding=extra > 0,
                             welfare=reward * covered - size, miss_loss=miss, excess_cost=excess,
                             gap=miss + excess, messages=count))


def run(directory, output):
    output.mkdir(parents=True, exist_ok=True)
    cases, validation_y = predictors(directory)
    labels = json.loads((directory / "labels.json").read_text())
    partitions = {s: cases.split == s for s in ("validation", "test")}
    validation = partitions["validation"]
    output_rows, diagnostics = [], []
    mechanism = AssignmentMechanism(MechanismConfig())
    centered_score = np.vectorize(mechanism.centered_brier_score)
    for c in conditions():
        policy, thresholds, diagnostic = fit_strategy(cases.local[validation], validation_y,
                                                     cases.degree[validation], cases.prior, c, SEED)
        diagnostic.update(asdict(c), config=tag(c))
        if c == Condition(ell=-1., phi=2/3, scale=2., audit=.25):
            alt, _, other = fit_strategy(cases.local[validation], validation_y,
                                         cases.degree[validation], cases.prior, c, SEED, start=11)
            diagnostic["alternate"] = dict(other, differing_cells=int((policy != alt).sum()))
            test = partitions["test"]
            main_decision = decisions(cases, test, c, policy, thresholds, SEED + 1)[0]
            alt_decision = decisions(cases, test, c, alt, thresholds, SEED + 1)[0]
            diagnostic["alternate"]["different_test_assignments"] = int((main_decision != alt_decision).sum())
        diagnostics.append(diagnostic)
        for split, mask in partitions.items():
            fixed = comparators(cases, mask, c.reward)
            seed = SEED if split == "validation" else SEED + 1
            chosen, r, caught, checked = decisions(cases, mask, c, policy, thresholds, seed)
            ids = np.array(cases.ids)[mask]
            y = np.array([[int(i in labels[key]["owners"]) for i in range(len(MODULES))] for key in ids])
            row = settle(ids, y, chosen, fixed["B4"], c.reward)
            centered = centered_score(r, y, cases.prior)
            row["score"] = c.scale * centered.sum(axis=1)
            row["audit_spend"] = c.audit_cost * checked.sum(axis=1)
            row["incentive_cost"] = row.score + c.audit_cost * c.audit * len(MODULES)
            row["fines"] = c.fine * caught.sum(axis=1)
            row["credit_paid"] = 0.
            row["credit_to_extra"] = 0.
            if c.credit != "off":
                rule = CreditRule.SHAPLEY if c.credit == "shapley" else CreditRule.EQUAL_SPLIT
                credit_mechanism = AssignmentMechanism(MechanismConfig(credit_rule=rule))
                for ix, (outcome, mask_value) in enumerate(zip(y, chosen)):
                    truth = set(np.flatnonzero(outcome))
                    assigned_set = set(np.flatnonzero(mask_value & BITS))
                    task = Task(ix, np.array([]), -1, truth, outcome, c.reward,
                                np.ones(len(MODULES)), ResolutionRule.CONJUNCTIVE)
                    payouts = credit_mechanism.divide_credit(task, assigned_set)
                    row.at[ix, "credit_paid"] = sum(payouts.values())
                    row.at[ix, "credit_to_extra"] = sum(v for i, v in payouts.items() if i not in truth)
            row["audits"] = checked.sum(axis=1)
            row["contradictions"] = caught.sum(axis=1)
            row["messages"] += (2 * checked * cases.degree[mask]).sum(axis=1)
            row["shading"] = (r - cases.local[mask]).mean(axis=1)
            row["method"] = "B2" if c.scale == c.audit == 0 else "mechanism"
            for field, value in asdict(c).items():
                row[field] = value
            row["config"], row["split"] = tag(c), split
            output_rows.append(row)
            if c == Condition():
                for name, selection in fixed.items():
                    baseline = settle(ids, y, selection, fixed["B4"], c.reward)
                    baseline["method"], baseline["split"], baseline["config"] = name, split, name
                    if name in ("B1", "central"):
                        baseline["messages"] = 0
                    output_rows.append(baseline)
    pd.concat(output_rows).to_csv(output / "replay.csv", index=False)
    dump(output / "diagnostics.json", diagnostics)
    test = partitions["test"]
    ids = np.array(cases.ids)[test]
    predictions = []
    for ix, key in enumerate(ids):
        for i, name in enumerate(MODULES):
            predictions.append(dict(issue=key, module=name, outcome=int(i in labels[key]["owners"]),
                                    local=cases.local[test][ix, i], central=cases.central[test][ix, i]))
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.directory, args.output)
