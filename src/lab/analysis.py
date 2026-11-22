"""Compare two policies on the same historical fixes."""
from dataclasses import dataclass

import pandas as pd

from src.e1.summary import intervals


@dataclass
class Comparison:
    totals: pd.DataFrame
    newly_covered: int
    newly_missed: int
    welfare_gain_interval: tuple[float, float]


def compare(selected: pd.DataFrame, polling: pd.DataFrame) -> Comparison:
    baseline = polling.loc[selected.index]
    table = {}
    for label, rows in [("Polling", baseline), ("Selected settings", selected)]:
        table[label] = {
            "Fixes fully covered": rows.coverage.sum(),
            "Module assignments": rows.assignments.sum(),
            "Extra module assignments": rows.padding.sum(),
            "Welfare per issue": rows.welfare.mean(),
            "Incentive cost per issue": rows.incentive_cost.mean(),
        }
    totals = pd.DataFrame(table)
    totals["Change"] = totals["Selected settings"] - totals["Polling"]
    delta = selected.coverage - baseline.coverage
    return Comparison(totals, int((delta > 0).sum()), int((delta < 0).sum()),
                      tuple(intervals(selected.welfare - baseline.welfare)))
