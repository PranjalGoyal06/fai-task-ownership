"""Plot held-out welfare and padding with uncertainty under acquisition."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e1.game import Condition
from src.e1.replay import tag


def figure(directory):
    plt.rcParams["svg.hashsalt"] = "issue-replay"
    methods = pd.read_csv(directory / "methods.csv")
    base = dict(ell=-1., phi=2/3)
    configs = [("Polling", Condition(**base)), ("Scoring", Condition(**base, scale=8.)),
               ("Audits", Condition(**base, audit=.25)), ("Both", Condition(**base, scale=8., audit=.25))]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), layout="constrained")
    for axis, metric, title in zip(axes, ["welfare", "padding"], ["Welfare per issue", "Extra modules per issue"]):
        for offset, group, color, label in [(-.12, "single", "#30577b", "1 owner"),
                                             (.12, "multi", "#a34721", "Multiple owners")]:
            rows = methods[methods.slice == group].set_index("config").loc[[tag(c) for _, c in configs]]
            values = rows[metric].to_numpy()
            errors = np.stack([values - rows[metric + "_low"], rows[metric + "_high"] - values])
            axis.errorbar(values, np.arange(len(rows)) + offset, xerr=np.maximum(0, errors),
                          fmt="o", markersize=4, capsize=2, color=color, label=label)
        axis.set_yticks(np.arange(4), [name for name, _ in configs])
        axis.invert_yaxis()
        axis.axvline(0, lw=.6, color="gray")
        axis.set_xlabel(title)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.22, -.2), ncol=2, frameon=False, fontsize=8)
    fig.savefig(directory / "acquisition.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    figure(parser.parse_args().directory)
