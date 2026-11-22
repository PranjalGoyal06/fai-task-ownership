"""Train a repository-local replay from the approved ownership map and chronological cutoffs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src.e1.evidence import Snapshot
from src.e1.models import fit_cases
from src.e1.replay import comparators
from src.lab.layout import import_path
from src.lab.storage import LAB_CACHE, Store
from src.lab.repository import FileSnapshot
from src.lab.intake import CACHE, REPO, read, save
from src.lab.ownership import (Ownership, default_cutoffs, label_cases, partition, rule_counts,
                               source_paths, suggest_from_tree, suggested_groups, verify_rules)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def extract_one(source, case, ownership):
    issue = {k: case[k] for k in ("id", "text", "base")}
    if ownership.roots is not None:
        return source.evidence(issue, root=ownership.roots, modules=ownership.names, owner=ownership.owner,
                               mentions=ownership.mentions, import_path=lambda p: import_path(p, ownership.roots))
    return source.evidence(issue, root="sklearn", modules=ownership.names,
                           owner=ownership.owner, mentions=ownership.mentions)


def train(scan, groups, cutoffs, cache=CACHE, progress=print):
    if "roots" in scan:
        store = Store(LAB_CACHE if cache == CACHE else cache, scan["repository"])
        with store.active():
            return _train(scan, groups, cutoffs, store.directory, progress, store)
    return _train(scan, groups, cutoffs, cache, progress, None)


def _train(scan, groups, cutoffs, cache, progress, store):
    ownership = Ownership(groups, scan.get("roots"))
    if store:
        source = FileSnapshot(cache, scan["roots"])
        history = {p for ref in {c["base"] for c in scan["cases"]} for _, p in source.files(ref, None)}
        verify_rules(rule_counts(ownership, (), history))
    elif (cache / "source.git").is_dir():
        history = {p for ref in {c["base"] for c in scan["cases"]} for p in source_paths(cache / "source.git", ref)}
        verify_rules(rule_counts(ownership, (), history))
    eligible, unmapped = label_cases(scan, ownership)
    cases, unavailable = partition(eligible, cutoffs)
    counts = Counter(c["split"] for c in cases)
    if any(not counts[s] for s in ("train", "validation", "test")):
        raise ValueError("Adjust the map or dates. Each period needs at least one usable fix.")
    manifest = dict(preset="sklearn-2022", repository=REPO, revision=scan["revision"],
                    split_counts=dict(counts), modules=ownership.names, ownership=ownership.groups,
                    cutoffs=list(cutoffs), protocol=1, exclusions={**unmapped, **unavailable},
                    intake_exclusions=scan["excluded"], scan_counts=scan["counts"],
                    cohort=[{k: c[k] for k in ("id", "issue", "pr", "base", "merge", "opened", "merged",
                                               "paths", "compiled", "owners", "split")} for c in cases],
                    text_digests=[fingerprint(c["text"]) for c in cases])
    if store:
        manifest.update(preset="public-python-e1", repository=scan["repository"], protocol=4,
                        roots=scan["roots"], window=scan["window"])
    directory = cache / "runs" / fingerprint(manifest)
    destination = directory / "inputs.json.gz"
    if destination.exists():
        progress("Restoring the matching trained run")
        return destination
    source = FileSnapshot(cache, scan["roots"]) if store else Snapshot(cache / "source.git")
    features = []
    evidence_key = [manifest["protocol"], ownership.groups]
    if store:
        evidence_key.append(ownership.roots)
    evidence_cache = cache / "evidence" / fingerprint(evidence_key)
    try:
        for i, case in enumerate(cases):
            progress(f"Reading pre-fix evidence {i + 1}/{len(cases)} · {case['id']}")
            path = evidence_cache / (fingerprint([case["id"], case["base"], case["text"]]) + ".json")
            if not path.exists():
                (store.json if store else save)(path, extract_one(source, case, ownership))
            features.append(read(path))
    finally:
        source.close()
    earlier = {c["id"]: {"owners": c["owners"]} for c in cases if c["split"] != "test"}
    docs = [" ".join([g["name"], *g["paths"]]) for g in ownership.groups]
    model, valid_y = fit_cases(cases, earlier, features, docs, progress)
    valid, test = model.split == "validation", model.split == "test"
    held = [c for c in cases if c["split"] == "test"]
    y = np.array([[int(i in c["owners"]) for i in range(len(groups))] for c in held])
    stats = []
    for i, name in enumerate(ownership.names):
        positives = sum(i in c["owners"] for c in cases if c["split"] == "train")
        fallback = positives in (0, counts["train"])
        calibrated = not fallback and len(np.unique(valid_y[:, i])) == 2
        p = model.local[test, i]
        bins = np.minimum((10 * p).astype(int), 9)
        calibration_error = sum(abs((p[bins == b] - y[bins == b, i]).sum()) for b in set(bins)) / len(y)
        stats.append(dict(agent=name, training_positives=positives, training_total=counts["train"],
                          validation_positives=int(valid_y[:, i].sum()), held_out_positives=int(y[:, i].sum()),
                          predictor="smoothed prior" if fallback else "logistic",
                          calibration="sigmoid" if calibrated else "unavailable",
                          brier=float(np.mean((p - y[:, i])**2)), calibration_error=float(calibration_error)))
    manifest["model_metrics"] = stats
    progress("Preparing pooled benchmarks and held-out traces")
    pooled = {reward: comparators(model, test, reward)["B4"].tolist() for reward in (5, 8, 12)}
    snapshots = {}
    for c, row in zip(cases, features, strict=True):
        if c["split"] == "test":
            snapshots[c["id"]] = dict(base=c["base"], features=row["features"], edges=[e[:4] for e in row["edges"]],
                                      citations={m: cs[:3] for m, cs in row["citations"].items()})
    result = dict(format=1, manifest=manifest, issues=[{k: c[k] for k in ("id", "title", "issue", "pr", "base")}
                                                      for c in held], prior=model.prior.tolist(),
                  validation_beliefs=model.local[valid].tolist(), validation_degree=model.degree[valid].tolist(),
                  validation_outcomes=valid_y.tolist(), beliefs=model.local[test].tolist(), degrees=model.degree[test].tolist(),
                  outcomes=y.tolist(), snapshots=snapshots, pooled=pooled)
    payload = gzip.compress(json.dumps(result, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
    if store:
        store.write(destination, payload)
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, delete=False) as tmp:
        tmp.write(payload)
    Path(tmp.name).replace(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--scan", type=Path, help="Frozen scan.json from a prepared public repository")
    parser.add_argument("--map", type=Path, help="JSON list of agents with names and paths")
    parser.add_argument("--cutoffs", nargs=2, metavar=("TRAIN_END", "VALIDATION_END"))
    opts = parser.parse_args()
    scan = read(opts.scan or opts.cache / "scan.json")
    if not scan["cases"]:
        parser.error("No usable fixes in this scan")
    if opts.map:
        groups = read(opts.map)
    elif "roots" in scan:
        directory = Store(LAB_CACHE if opts.cache == CACHE else opts.cache, scan["repository"]).directory
        source = FileSnapshot(directory, scan["roots"])
        groups = suggest_from_tree([p for _, p in source.files(scan["cases"][0]["base"], None)], scan["roots"])
    else:
        groups = suggested_groups(source_paths(opts.cache / "source.git", scan["cases"][0]["base"]))
    eligible, _ = label_cases(scan, Ownership(groups, scan.get("roots")))
    print(train(scan, groups, opts.cutoffs or default_cutoffs(eligible), opts.cache))
