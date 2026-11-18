"""Repeat the primary scoring comparison with independent private-type draws."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.e1.dataset import MODULES
from src.e1.game import Condition, fit_strategy
from src.e1.models import predictors
from src.e1.replay import decisions, settle
from src.e1.summary import intervals


def repeat(directory, output):
    cases, y = predictors(directory)
    valid, test = cases.split == "validation", cases.split == "test"
    c = Condition(ell=-1., phi=2/3)
    ids = np.asarray(cases.ids)[test]
    labels = json.loads((directory / "labels.json").read_text())
    outcomes = np.array([[int(i in labels[key]["owners"]) for i in range(len(MODULES))] for key in ids])
    rows = []
    for seed in (331, 341, 351, 361, 371):
        metrics, statuses = [], []
        for cond in (c, replace(c, scale=8.)):
            policy, threshold, diag = fit_strategy(cases.local[valid], y, cases.degree[valid],
                                                   cases.prior, cond, seed)
            chosen, _, _, _ = decisions(cases, test, cond, policy, threshold, seed + 1)
            metrics.append(settle(ids, outcomes, chosen, chosen, cond.reward).welfare)
            statuses.append(diag["status"])
        gain = metrics[1] - metrics[0]
        lo, hi = intervals(gain)
        rows.append(dict(seed=seed, baseline_status=statuses[0], scoring_status=statuses[1],
                         gross=gain.mean(), low=lo, high=hi))
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repeat(args.directory, args.output)
