"""Rebuild the portable Django replay inputs from the prepared E1 evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.e1.dataset import MODULES
from src.e1.models import predictors
from src.e1.replay import comparators


def prepare(inputs: Path, reports: Path, destination: Path):
    manifest = (reports / "manifest.json").read_bytes()
    cases, _ = predictors(inputs)
    cohort = pd.read_csv(reports / "cohort.csv").set_index("issue")
    published = pd.read_csv(reports / "predictions.csv.gz")
    held = published.pivot(index="issue", columns="module", values="local")
    issues = json.loads((inputs / "issues.json").read_text())
    if list(cohort.index) != cases.ids or list(cohort.split) != list(cases.split):
        raise ValueError("Input cohort differs from the published issue sequence")
    test_ids = list(cohort.index[cohort.split == "test"])
    np.testing.assert_allclose(cases.local[cases.split == "test"], held.loc[test_ids, list(MODULES)],
                               rtol=0, atol=1e-12)

    validation, snapshots = {}, {}
    for issue, local, degree in zip(issues, cases.local, cases.degree, strict=True):
        key = issue["id"]
        if issue["split"] == "validation":
            validation[key] = {"beliefs": local.tolist(), "degree": degree.tolist()}
        if issue["split"] == "test":
            cache = json.loads((inputs / "evidence" / (key + ".json")).read_text())
            if cache["id"] != key or cache["base"] != cohort.at[key, "base"]:
                raise ValueError(f"Pre-fix revision mismatch for {key}")
            snapshots[key] = {
                "title": issue["text"].strip().splitlines()[0],
                "base": cache["base"], "features": cache["features"],
                "citations": {module: entries[:3] for module, entries in cache["citations"].items()},
                "edges": [edge[:4] for edge in cache["edges"]],
            }
    pooled = {str(reward): comparators(cases, cases.split == "test", reward)["B4"].tolist()
              for reward in (5, 8, 12)}
    bundle = dict(version=1, pooled=pooled, pooled_ids=test_ids, manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                  modules=MODULES, prior=cases.prior.tolist(), validation=validation,
                  snapshots=snapshots)
    destination.write_bytes(gzip.compress(json.dumps(bundle, sort_keys=True).encode(), mtime=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("reports", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.inputs, args.reports, args.destination)
