"""Create paired replay summaries and a validation-selected budget frontier."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.e1.dataset import MODULES
from src.e1.replay import SEED
from src.simulation_runner import SimulationConfig, run_full_simulation

METRICS = ["coverage", "padding", "any_padding", "welfare", "gap", "miss_loss",
           "excess_cost", "incentive_cost", "shading", "fines", "credit_paid",
           "credit_to_extra", "messages", "contradictions"]


def intervals(values, seed=SEED):
    values = np.asarray(values, dtype=float)
    sample = np.random.default_rng(seed).integers(len(values), size=(2000, len(values)))
    return np.quantile(values[sample].mean(axis=1), [.025, .975])


def summarize(run, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(run / "replay.csv")
    diag = pd.read_json(run / "diagnostics.json").set_index("config")
    test = frame[frame.split == "test"]
    rows = []
    for config, data in test.groupby("config", sort=False):
        for name, part in (("all", data), ("single", data[data.k == 1]), ("multi", data[data.k > 1])):
            record = dict(config=config, slice=name, n=len(part))
            record["status"] = diag.loc[config, "status"] if config in diag.index else "fixed"
            for col in METRICS:
                v = part[col].dropna()
                if len(v):
                    record[col] = v.mean()
                    record[col + "_low"], record[col + "_high"] = intervals(v)
                    if col in ("coverage", "any_padding"):
                        # Wilson bounds retain uncertainty for zero-success small slices.
                        freq, n, z = v.mean(), len(v), 1.96
                        denom = 1 + z*z/n
                        middle = (freq + z*z/(2*n)) / denom
                        width = z * np.sqrt(freq*(1-freq)/n + z*z/(4*n*n)) / denom
                        record[col + "_low"] = max(0., middle - width)
                        record[col + "_high"] = min(1., middle + width)
            oracle_welfare = (part.welfare + part.gap).mean()
            record["efficiency"] = part.welfare.mean() / oracle_welfare if oracle_welfare > 0 else np.nan
            rows.append(record)
    pd.DataFrame(rows).to_csv(output / "methods.csv", index=False)

    eligible = frame[(frame.credit == "off") & (frame.reward == 8) & (frame.delta == .4)]
    rows = []
    for (ell, phi), group in eligible.groupby(["ell", "phi"]):
        validation = group[group.split == "validation"].groupby("config")
        criteria = validation[["gap", "incentive_cost"]].mean()
        criteria = criteria[diag.loc[criteria.index, "status"].values == "converged"]
        held = group[group.split == "test"]
        base = held[(held.scale == 0) & (held.audit == 0)].set_index("issue")
        for budget in [0., .25, .5, 1., 2.]:
            affordable = criteria[criteria.incentive_cost <= budget]
            if affordable.empty:
                rows.append(dict(ell=ell, phi=phi, budget=budget, status="no eligible policy"))
                continue
            best = affordable.sort_values(["gap", "incentive_cost"]).index[0]
            policy = held[held.config == best].set_index("issue").loc[base.index]
            gain = policy.welfare - base.welfare
            net = gain - policy.incentive_cost
            lo, hi = intervals(gain)
            nlo, nhi = intervals(net)
            rows.append(dict(ell=ell, phi=phi, budget=budget, config=best,
                             validation_cost=affordable.loc[best, "incentive_cost"],
                             held_cost=policy.incentive_cost.mean(), gross=gain.mean(),
                             gross_low=lo, gross_high=hi, net=net.mean(), net_low=nlo, net_high=nhi))
    pd.DataFrame(rows).to_csv(output / "frontier.csv", index=False)
    predictions = pd.read_csv(run / "predictions.csv")
    rows = []
    for name in ["local", "central"]:
        for module, group in [("all", predictions), *predictions.groupby("module")]:
            bins = np.minimum(9, (10 * group[name]).astype(int))
            error = group[name] - group.outcome
            ece = error.groupby(bins).sum().abs().sum() / len(group)
            rows.append(dict(model=name, module=module, n=len(group), brier=(error ** 2).mean(), ece=ece))
    pd.DataFrame(rows).to_csv(output / "calibration.csv", index=False)


def synthetic(output):
    rows = []
    for seed in (331, 332, 333):
        config = SimulationConfig(num_rounds=200, num_agents=len(MODULES), seed=seed,
                                  agent_regime="custom", deflector_ratio=0., acquirer_ratio=2/3,
                                  acquirer_ell=-1., coalition_size_k=1, use_coalition_range=False,
                                  reward_per_owner=8., default_cost=1.)
        result = run_full_simulation(config)
        for name in ["Description (B1)", "Polling Router (B2)", "Voting (B3)"]:
            rows.append(dict(seed=seed, method=name, welfare=np.mean(result.raw_trackers[name]["welfare"])))
    pd.DataFrame(rows).to_csv(output / "synthetic.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.run, args.output)
    synthetic(args.output)
