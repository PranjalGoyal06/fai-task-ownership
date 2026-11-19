"""Load the verified, offline nine-module Django preset in manifest order."""
import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.e1.dataset import MODULES

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DjangoPreset:
    digest: str
    manifest: dict
    prior: np.ndarray
    validation_beliefs: np.ndarray
    validation_degree: np.ndarray
    validation_outcomes: np.ndarray
    issues: pd.DataFrame
    beliefs: np.ndarray
    degrees: np.ndarray
    snapshots: dict
    outcomes: np.ndarray
    pooled_assignments: dict[int, np.ndarray]
    methods: pd.DataFrame


def load_preset(root: Path = ROOT) -> DjangoPreset:
    reports = root / "reports/e1"
    raw = (root / "presets/django/replay-inputs.json.gz").read_bytes()
    extra = json.loads(gzip.decompress(raw))
    manifest_bytes = (reports / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (extra["version"] != 1 or extra["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
            or list(MODULES) != extra["modules"] or list(MODULES) != manifest["modules"]):
        raise ValueError("The replay inputs and E1 manifest describe different presets")

    def read_table(name):
        content = (reports / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["output_sha256"][name]:
            raise ValueError(f"Checksum mismatch for {name}")
        return pd.read_csv(io.BytesIO(content), compression="gzip" if name.endswith(".gz") else None)

    cohort = read_table("cohort.csv").set_index("issue")
    if not cohort.index.is_unique:
        raise ValueError("The frozen cohort has duplicate issue identifiers")
    validation = cohort[cohort.split == "validation"]
    issues = cohort[cohort.split == "test"].copy()
    predictions = read_table("predictions.csv.gz")

    def matrix(column):
        return predictions.pivot(index="issue", columns="module", values=column).loc[issues.index, list(MODULES)]

    snapshots = extra["snapshots"]
    if set(snapshots) != set(issues.index) or set(extra["validation"]) != set(validation.index):
        raise ValueError("The replay inputs and cohort have different issue sets")
    degrees = []
    for key, base in issues.base.items():
        snapshot = snapshots[key]
        if snapshot["base"] != base:
            raise ValueError(f"Unexpected pre-fix snapshot for {key}")
        neighbors = [set() for _ in MODULES]
        for source, target, *_ in snapshot["edges"]:
            neighbors[source].add(target)
            neighbors[target].add(source)
        degrees.append([len(adjacent) for adjacent in neighbors])
    issues["title"] = [snapshots[key]["title"] for key in issues.index]
    if extra["pooled_ids"] != list(issues.index):
        raise ValueError("Pooled benchmarks differ from the held-out issue order")
    pooled = {int(reward): np.asarray(masks) for reward, masks in extra["pooled"].items()}
    earlier = [extra["validation"][key] for key in validation.index]
    return DjangoPreset(
        digest=hashlib.sha256(raw).hexdigest(), manifest=manifest, prior=np.array(extra["prior"]),
        validation_beliefs=np.array([row["beliefs"] for row in earlier]),
        validation_degree=np.array([row["degree"] for row in earlier]),
        validation_outcomes=np.array([[int(m in names.split(",")) for m in MODULES] for names in validation.owners]),
        issues=issues, beliefs=matrix("local").to_numpy(), degrees=np.array(degrees),
        snapshots=snapshots, outcomes=matrix("outcome").to_numpy(dtype=int),
        pooled_assignments=pooled, methods=read_table("methods.csv"))
