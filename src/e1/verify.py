"""Verify the label boundary by perturbing one actual held-out resolving patch."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from src.e1.dataset import MODULES, dump, owners
from src.e1.evidence import Snapshot
from src.e1.game import Condition, fit_strategy
from src.e1.models import predictors
from src.e1.replay import SEED, comparators, decisions, settle


def verify(directory, repository):
    original, training = predictors(directory)
    test = original.split == "test"
    val = original.split == "validation"
    labels = json.loads((directory / "labels.json").read_text())
    sample = next(r for r in json.loads((directory / "issues.json").read_text())
                  if r["split"] == "test" and len(labels[r["id"]]["owners"]) > 1)
    key = sample["id"]
    patch = parquet.read_table(directory / "test.parquet", columns=["patch"],
                               filters=[("instance_id", "=", key)]).column("patch")[0].as_py()
    assert owners(patch) == labels[key]["owners"]
    source = Snapshot(repository)
    try:
        before = source.evidence(dict(sample, patch=patch))
        changed_patch = "diff --git a/django/forms/widgets.py b/django/forms/widgets.py"
        after = source.evidence(dict(sample, patch=changed_patch))
        assert before == after
        cached = json.loads((directory / "evidence" / (key + ".json")).read_text())
        cached.pop("signature")
        assert json.loads(json.dumps(before)) == cached
    finally:
        source.close()
    truth = labels[key]["owners"]
    replacement = owners(changed_patch)
    assert truth != replacement
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        scratch = Path(temp)
        for name in ("issues.json", "evidence", "descriptions.json"):
            (scratch / name).symlink_to((directory / name).resolve())
        labels[key]["owners"] = replacement
        dump(scratch / "labels.json", labels)
        perturbed, training_after = predictors(scratch)
    for name in ("local", "central", "descriptions", "joint_masks", "joint_weights", "prior", "degree"):
        np.testing.assert_array_equal(getattr(original, name), getattr(perturbed, name))
    np.testing.assert_array_equal(training, training_after)
    c = Condition(ell=-1., phi=2/3, scale=2., audit=.25)
    outputs = []
    for data, outcomes in ((original, training), (perturbed, training_after)):
        policy, thresholds, diagnostics = fit_strategy(data.local[val], outcomes, data.degree[val],
                                                       data.prior, c, SEED)
        outputs.append((comparators(data, test, c.reward), decisions(data, test, c, policy,
                                                                     thresholds, SEED + 1)))
    for method in outputs[0][0]:
        np.testing.assert_array_equal(outputs[0][0][method], outputs[1][0][method])
    for left, right in zip(outputs[0][1], outputs[1][1]):
        np.testing.assert_array_equal(left, right)
    chosen = np.array([sum(1 << i for i in truth)])
    y = np.array([[int(i in target) for i in range(len(MODULES))] for target in (truth, replacement)])
    settlement = settle([key, key], y, chosen.repeat(2), chosen.repeat(2), c.reward)
    assert settlement.owners.nunique() == 2
    return dict(issue=key, base=sample["base"], original=truth, replacement=replacement,
                snapshots_equal=True, predictions_equal=True, all_baseline_assignments_equal=True,
                strategic_reports_and_assignments_equal=True, checked_test_issues=int(test.sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dump(args.output, verify(args.directory, args.repository))
